import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.multinomial import Multinomial
from tqdm import trange
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import chain
from more_itertools import roundrobin

import numpy as np
import random
from functional.cross_entropy import cross_entropy
from data_generate.sampler import SuppQueryBatchSampler
from training.inner import inner_maml_amvi
from training.util import get_accuracy, kldiv_mvn_diagcov, get_zk_likelihood
from evaluate.util_eval import meta_evaluation_amvi


def metatrain_seqdataset_amvi(model, var_approx_list, var_beta, model_device, trainset, evalset, outer_kl_scale, 
        nstep_outer, nstep_inner, lr_inner, first_order, seqtask, num_task_per_itr, 
        task_by_supercls, num_way, num_shot, num_query_per_cls, eval_prev_task, eval_per_num_iter, 
        num_eval_task, eval_task_by_supercls, nstep_inner_eval, writer, task_idx, prev_glob_step, verbose, sample_batch):
    
    num_var_approx = len(var_approx_list)
    eta = np.zeros(num_var_approx)
    eta[task_idx%num_var_approx] = 1.0
    eta = torch.tensor(eta, device=model_device).float()

    for itr in trange(nstep_outer, desc='meta-train {}'.format(verbose if verbose is not None else task_idx)):
        
        # calculate variational distribution of beta
        if var_beta is not None:
            if itr < nstep_outer:
                eta = np.zeros(num_var_approx)
                eta[task_idx%num_var_approx] = 1.0
                eta = torch.tensor(eta, device=model_device).float()
            var_beta.update_posterior(eta)
        
        # prepare train data
        trainset = trainset if isinstance(trainset, list) else [trainset]
        trainloader = []
        num_task_per_dataset = Multinomial(num_task_per_itr, torch.tensor([1.] * len(trainset))).sample()\
            .to(dtype=torch.int64).tolist() if len(trainset) > 1 else [num_task_per_itr]
        
        for trset, n_task in zip(trainset, num_task_per_dataset):
            trainsampler = SuppQueryBatchSampler(
                dataset=trset, seqtask=seqtask, num_task=n_task, task_by_supercls=task_by_supercls,
                num_way=num_way, num_shot=num_shot, num_query_per_cls=num_query_per_cls
            )
            trainloader.append(DataLoader(trset, batch_sampler=trainsampler))
        

        # update variational parameters
        eta, negloglik_supp, negloglik_query, loss_outer, zk_likelihood, accuracy_query = \
            outer_step(var_approx_list, model, var_beta, trainloader, num_way, num_shot, num_query_per_cls,\
                    outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, model_device, sample_batch)

        # record variational posterior of z
        if writer is not None:
            for i, eta_i in enumerate(eta):
                writer.add_scalar(tag="posterior_z_{}".format(i), scalar_value=eta_i, \
                    global_step = prev_glob_step + itr)

        # meta-evaluation once every 'eval_per_num_iter' iterations
        if (itr + 1) % eval_per_num_iter == 0:
            if not eval_prev_task:
                evalset = [evalset[-1]]

            for ldr_idx, evset in enumerate(evalset):
                
                loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi(
                evset, num_task=num_eval_task, task_by_supercls=eval_task_by_supercls, num_way=num_way,
                num_shot=num_shot, num_query_per_cls=num_query_per_cls, model=model,
                variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1, nstep_inner=nstep_inner_eval,
                lr_inner=lr_inner, model_device=model_device, var_beta=var_beta, sample_batch=sample_batch
                )

                writer.add_scalar(
                    tag='loss_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss_meta_eval',
                    scalar_value=loss_eval, global_step=prev_glob_step + itr
                )
                writer.add_scalar(
                    tag='accuracy_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'accuracy_meta_eval',
                    scalar_value=accuracy_eval, global_step=prev_glob_step + itr
                )
                writer.add_scalar(
                    tag='loss95ci_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'loss95ci_meta_eval',
                    scalar_value=loss_95ci, global_step=prev_glob_step + itr
                )
                writer.add_scalar(
                    tag='acc95ci_meta_eval_task{}'.format(ldr_idx) if eval_prev_task else 'acc95ci_meta_eval',
                    scalar_value=acc_95ci, global_step=prev_glob_step + itr
                )

        # update lr scheduler
        for var_obj in var_approx_list:
            if var_obj.lr_scheduler is not None:
                if 'ReduceLROnPlateau' in var_obj.lr_scheduler.__class__.__name__:
                    var_obj.lr_scheduler.step(loss_outer)
                else:
                    var_obj.lr_scheduler.step()

    # update variational objective of beta
    if var_beta is not None:
        var_beta.update_posterior(eta)
        var_beta.update_prior()



def batch_loss_amvi(support_img, support_lbl, query_img, query_lbl, var_obj, model, 
                    nstep_inner, lr_inner, first_order, device,  cal_zk_likelihood=False, 
                    var_beta=None, kth=None, sample_batch=True):

    if not sample_batch:
        support_img = support_img.expand(1, *support_img.size())
        support_lbl = support_lbl.expand(1, *support_lbl.size())
        query_img = query_img.expand(1, *query_img.size())
        query_lbl = query_lbl.expand(1, *query_lbl.size())

        mean_inner, cov_inner = inner_maml_amvi(
            model=model, var_obj=var_obj, inputs=support_img, labels=support_lbl, 
            nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=first_order, 
            device=device, kl_scale=0.01, sample_batch=False
        )

        nll_supp = torch.tensor(0.0, device=device)
        nll_query = torch.tensor(0.0, device=device)
        acc_query = torch.tensor(0.0, device=device)
        zk_likelihood = torch.tensor(0.0, device=device)

        for _ in range(var_obj.num_mc_sample):
            out_supp = model(x=support_img, mean=mean_inner, cov=var_obj.exp_covar(cov_inner))
            nll_supp += cross_entropy(input=out_supp, target=support_lbl, reduction='mean')
            out_query = model(x=query_img, mean=mean_inner, cov=var_obj.exp_covar(cov_inner))
            nll_query += cross_entropy(input=out_query, target=query_lbl, reduction='mean')

            with torch.no_grad():
                acc_query += get_accuracy(labels=query_lbl, outputs=out_query)
                
            if cal_zk_likelihood:
                zk_likelihood += get_zk_likelihood(var_obj, var_obj.mean, var_obj.covar, mean_inner, \
                    cov_inner, var_beta, kth, device, query_img.size()[1])
        
        return nll_supp/var_obj.num_mc_sample, nll_query/var_obj.num_mc_sample, \
                acc_query/var_obj.num_mc_sample, zk_likelihood/var_obj.num_mc_sample

    else:
        support_img = support_img.expand(var_obj.num_mc_sample, *support_img.size())
        support_lbl = support_lbl.expand(var_obj.num_mc_sample, *support_lbl.size())
        query_img = query_img.expand(var_obj.num_mc_sample, *query_img.size())
        query_lbl = query_lbl.expand(var_obj.num_mc_sample, *query_lbl.size())

        mean_inner, cov_inner = inner_maml_amvi(
            model=model, var_obj=var_obj, inputs=support_img, labels=support_lbl, 
            nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=first_order, 
            device=device, kl_scale=0.01, sample_batch=True
        )

        out_supp = model(x=support_img, mean=mean_inner, cov=var_obj.exp_covar(cov_inner))
        nll_supp = cross_entropy(input=out_supp, target=support_lbl, reduction='mean')
        out_query = model(x=query_img, mean=mean_inner, cov=var_obj.exp_covar(cov_inner))
        nll_query = cross_entropy(input=out_query, target=query_lbl, reduction='mean')

        zk_likelihood = torch.tensor(0.0, device=device)
        
        with torch.no_grad():
            accuracy_query = get_accuracy(labels=query_lbl, outputs=out_query)

            if cal_zk_likelihood:
                zk_likelihood = get_zk_likelihood(var_obj, var_obj.mean, var_obj.covar, mean_inner, \
                    cov_inner, var_beta, kth, device, query_img.size()[1], var_obj.num_mc_sample)

    return nll_supp, nll_query, accuracy_query, zk_likelihood


def batch_gradient_amvi(var_obj, model, images, labels, num_way, num_shot, 
                        nstep_inner, lr_inner, first_order, var_beta, i, num_batch, sample_batch):
    
    device = var_obj.device

    supp_idx = num_way * num_shot
    ims = images.to(device=device)
    lbls = labels.to(device=device)
    support_img, query_img = ims[:supp_idx, :], ims[supp_idx:, :]
    support_lbl, query_lbl = lbls[:supp_idx], lbls[supp_idx:]

    negloglik_supp, negloglik_query, accuracy_query, zk_likelihood = \
        batch_loss_amvi(support_img, support_lbl, query_img, query_lbl,  
                        var_obj, model, nstep_inner, lr_inner, first_order, device, 
                        cal_zk_likelihood=var_beta is not None, var_beta=var_beta, kth=i, 
                        sample_batch=sample_batch)
    
    negloglik_supp = negloglik_supp / num_batch
    negloglik_query = negloglik_query / num_batch
    accuracy_query = accuracy_query / num_batch
    zk_likelihood = zk_likelihood / num_batch

    nll_querysupp_one_task = negloglik_supp + negloglik_query
    var_obj.optimizer.zero_grad()
    nll_querysupp_one_task.backward()

    grad_wrt_mean = OrderedDict([
        (name, param.grad) for name, param in var_obj.mean.items()
    ])

    grad_wrt_covar = OrderedDict([
        (name, param.grad) for (name, param) in var_obj.covar.items()
    ])

    return grad_wrt_mean, grad_wrt_covar, negloglik_supp, negloglik_query, zk_likelihood, accuracy_query

                    

def outer_gradient_amvi(var_approx_list, model, var_beta, dataloader, num_way, num_shot, num_query_per_cls, 
                        outer_kl_scale, nstep_inner, lr_inner, first_order, num_task_per_itr, sample_batch):

    # for gradient accumulation
    grad_wrt_mean_list = []
    grad_wrt_covar_list = []

    for var_obj in var_approx_list:
        grad_wrt_mean_list.append(OrderedDict([
            (name, torch.zeros_like(mu)) for name, mu in var_obj.mean.items()
        ]))
        grad_wrt_covar_list.append(OrderedDict([
            (name, torch.zeros_like(cov)) for name, cov in var_obj.covar.items()
        ]))

    # for graphs
    negloglik_supp_list = []
    negloglik_query_list =[]
    loss_outer_list = []
    zk_likelihood_list = []
    acc_query_list = []

    if isinstance(outer_kl_scale, list):
        assert len(var_approx_list) == len(outer_kl_scale)
    else:
        outer_kl_scale = [outer_kl_scale] * len(var_approx_list)

    for i, var_obj in enumerate(var_approx_list):

        negloglik_query = torch.tensor(0., device=var_obj.device)
        negloglik_supp = torch.tensor(0., device=var_obj.device)
        loss_outer = torch.tensor(0., device=var_obj.device)
        zk_likelihood = torch.tensor(0., device=var_obj.device)
        accuracy_query = torch.tensor(0., device=var_obj.device)
        nll_querysupp_one_task_divisor = (num_shot + num_query_per_cls) * num_way * var_obj.num_mc_sample

        for batch_idx, (images, labels) in enumerate(chain(*dataloader), 0):

            grad_wrt_mean, grad_wrt_covar, nll_supp, nll_query, zk_ll, acc_query = \
                batch_gradient_amvi(var_obj, model, images, labels, num_way, num_shot, 
                                    nstep_inner, lr_inner, first_order, var_beta, i, num_task_per_itr, sample_batch)

            with torch.no_grad():
                for acc_g_mu, acc_g_cov, g_mu, g_cov in \
                    zip(grad_wrt_mean_list[i].values(), grad_wrt_covar_list[i].values(),
                        grad_wrt_mean.values(), grad_wrt_covar.values()):
                            
                    acc_g_mu += g_mu
                    acc_g_cov += g_cov
                    # zero gradients after accumulating mean and cov gradients
                    g_mu.zero_()
                    g_cov.zero_()

                negloglik_query += nll_query
                negloglik_supp += nll_supp
                loss_outer += (nll_supp + nll_query)
                zk_likelihood += zk_ll
                accuracy_query += acc_query
        
        negloglik_query_list.append(negloglik_query)
        negloglik_supp_list.append(negloglik_supp)
        loss_outer_list.append(loss_outer)
        zk_likelihood_list.append(zk_likelihood)
        acc_query_list.append(accuracy_query)

        # kl-div term
        kldiv = outer_kl_scale[i] * kldiv_mvn_diagcov(
            mean_p=var_obj.mean, cov_p=var_obj.exp_covar(var_obj.covar),
            mean_q=var_obj.mean_old, cov_q=var_obj.exp_covar(var_obj.covar_old)
        ) / (nll_querysupp_one_task_divisor * num_task_per_itr)

        with torch.no_grad():
            loss_outer += kldiv

        # accumulate gradient for kldiv term
        var_obj.optimizer.zero_grad()
        kldiv.backward()
        kldiv_gradient_wrt_mean = OrderedDict([(name, mu.grad) for name, mu in var_obj.mean.items()])
        kldiv_gradient_wrt_covar = OrderedDict([(name, cov.grad) for name, cov in var_obj.covar.items()])

        # add nll and kldiv gradients together
        with torch.no_grad():
            for acc_g_mu, acc_g_cov, kldiv_g_mu, kldiv_g_cov in \
                zip(grad_wrt_mean_list[i].values(), grad_wrt_covar_list[i].values(), \
                    kldiv_gradient_wrt_mean.values(), kldiv_gradient_wrt_covar.values()):
                acc_g_mu += kldiv_g_mu
                acc_g_cov += kldiv_g_cov
                kldiv_g_mu.zero_()
                kldiv_g_cov.zero_()


    return grad_wrt_mean_list, grad_wrt_covar_list, negloglik_supp_list, negloglik_query_list, \
            loss_outer_list, zk_likelihood_list, acc_query_list

        
def outer_step(var_approx_list, model, var_beta, dataloader, num_way, num_shot, 
                num_query_per_cls, outer_kl_scale, nstep_inner, lr_inner, first_order, 
                num_task_per_itr, model_device, sample_batch):

    grad_wrt_mean_list, grad_wrt_covar_list, negloglik_supp_list, negloglik_query_list, \
        loss_outer_list, zk_likelihood_list, acc_query_list = outer_gradient_amvi(var_approx_list, \
            model, var_beta, dataloader, num_way, num_shot, num_query_per_cls, outer_kl_scale, nstep_inner, \
                lr_inner, first_order, num_task_per_itr, sample_batch)
        
    # calculate posterior of z
    with torch.no_grad():
        ll_query = -1.0 * torch.cat([x.view(-1).to(device=model_device) for x in negloglik_query_list], dim=0)
        ll_zk = torch.cat([x.view(-1).to(device=model_device) for x in zk_likelihood_list])
        eta = torch.exp(ll_query+ll_zk - torch.logsumexp(ll_query+ll_zk, dim=0))
             
    # update mean and covar
    for i, eta_i in enumerate(eta):
        var_obj = var_approx_list[i]
        eta_i = eta_i.to(device=var_obj.device)
        total_grad_wrt_mean = [eta_i * grad for grad in grad_wrt_mean_list[i].values()]
        total_grad_wrt_covar = [eta_i * grad for grad in grad_wrt_covar_list[i].values()]
        var_obj.optimizer.step_grad(gradient=total_grad_wrt_mean + total_grad_wrt_covar)
    

    with torch.no_grad():
        negloglik_supp = torch.sum(
            eta * torch.cat([x.view(-1).to(device=model_device) for x in negloglik_supp_list], dim=0)
        )
        negloglik_query = torch.sum(eta * (-1.0 * ll_query))
        loss_outer = torch.sum(
            eta * torch.cat([x.view(-1).to(device=model_device) for x in loss_outer_list], dim=0)
        )
        zk_likelihood = torch.sum(eta * ll_zk)
        accuracy_query = torch.sum(
            eta * torch.cat([x.view(-1).to(device=model_device) for x in acc_query_list], dim=0)
        )
    
    return eta, negloglik_supp, negloglik_query, loss_outer, zk_likelihood, accuracy_query
    