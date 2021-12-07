import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict

from data_generate.sampler import SuppQueryBatchSampler
from functional.cross_entropy import cross_entropy
from training.inner import inner_maml_amvi
from training.util import get_accuracy, get_zk_likelihood


def meta_evaluation_amvi(evalset, num_task, task_by_supercls, num_way, num_shot, num_query_per_cls, model, 
                        variational_obj_list, inner_on_mean, n_sample=1, nstep_inner=10, lr_inner=0.4, 
                        model_device=None, var_beta=None, sample_batch=True, return_eta=False):
    loss = []
    accuracy = []
    z_pos_list = []
    for _ in range(len(variational_obj_list)):
        z_pos_list.append([])

    evalsampler = SuppQueryBatchSampler(
        dataset=evalset, seqtask=False, num_task=num_task, task_by_supercls=task_by_supercls, num_way=num_way,
        num_shot=num_shot, num_query_per_cls=num_query_per_cls
    )
    evalloader = DataLoader(evalset, batch_sampler=evalsampler)

    cal_zk_likelihood = True if var_beta is not None else False

    for images, labels in evalloader:

        nll_query_list = []
        pred_logits_query_list = []
        zk_likelihood_list = []

        for kth, variational_obj in enumerate(variational_obj_list):

            device = variational_obj.device
            expand_dim = variational_obj.num_mc_sample if sample_batch else 1
            
            ims = images.to(device=device)
            lbls = labels.to(device=device)

            supp_idx = num_way * num_shot
            support_img, query_img = ims[:supp_idx, :], ims[supp_idx:, :]
            support_lbl, query_lbl = lbls[:supp_idx], lbls[supp_idx:]

            support_img = support_img.expand(expand_dim, *support_img.size())
            support_lbl = support_lbl.expand(expand_dim, *support_lbl.size())
            query_img = query_img.expand(expand_dim, *query_img.size())
            query_lbl = query_lbl.expand(expand_dim, *query_lbl.size())

            mean_inner, cov_inner = inner_maml_amvi_v2(
                model=model, var_obj=variational_obj, inputs=support_img, labels=support_lbl, 
                nstep_inner=nstep_inner, lr_inner=lr_inner, first_order=True, device=device,
                kl_scale=0.01, sample_batch=sample_batch
            )

            if sample_batch:

                output = model(x=query_img, mean=mean_inner, cov=variational_obj.exp_covar(cov_inner))
            
                with torch.no_grad():
                    nll = cross_entropy(input=output, target=query_lbl, reduction='mean')
                    logits = torch.softmax(output, dim=-1).mean(0)
                    zk_likelihood = get_zk_likelihood(variational_obj, variational_obj.mean, variational_obj.covar, \
                        mean_inner, cov_inner, var_beta, kth, device, query_img.size()[1], variational_obj.num_mc_sample) \
                            if cal_zk_likelihood else torch.tensor(0.0, device=device) 

                nll_query_list.append(nll.to(device=model_device))
                pred_logits_query_list.append(logits.to(device=model_device))
                zk_likelihood_list.append(zk_likelihood.to(device=model_device))
            
            else:

                output = model(x=query_img, mean=mean_inner, cov=variational_obj.exp_covar(cov_inner))
                with torch.no_grad():
                    nll = cross_entropy(input=output, target=query_lbl, reduction='mean') / variational_obj.num_mc_sample
                    logits = torch.softmax(output, dim=-1) / variational_obj.num_mc_sample

                for _ in range(variational_obj.num_mc_sample - 1):
                    output = model(x=query_img, mean=mean_inner, cov=variational_obj.exp_covar(cov_inner))
                    with torch.no_grad():
                        nll += cross_entropy(input=output, target=query_lbl, reduction='mean') / variational_obj.num_mc_sample
                        logits += torch.softmax(output, dim=-1) / variational_obj.num_mc_sample
                
                with torch.no_grad():
                    zk_likelihood = get_zk_likelihood(variational_obj, variational_obj.mean, variational_obj.covar, \
                        mean_inner, cov_inner, var_beta, kth, device, query_img.size()[1], variational_obj.num_mc_sample) \
                            if cal_zk_likelihood else torch.tensor(0.0, device=device) 
                
                nll_query_list.append(nll.to(device=model_device))
                pred_logits_query_list.append(logits.squeeze(0).to(device=model_device))
                zk_likelihood_list.append(zk_likelihood.to(device=model_device))


        # calculate posterior of z
        ll_query = torch.cat([-x.view(-1) for x in nll_query_list], dim=0)
        ll_zk = torch.cat([x.view(-1) for x in zk_likelihood_list])
        pos_z = torch.exp(ll_query+ll_zk - torch.logsumexp(ll_query+ll_zk, dim=0))
        pos_z_likelihood = torch.exp(ll_zk - torch.logsumexp(ll_zk, dim=0))

        nll_query = torch.tensor(0., device=model_device)
        pred_logits_query = torch.zeros_like(pred_logits_query_list[0], device=model_device)
        for k in range(len(nll_query_list)):
            nll_query += pos_z[k] * nll_query_list[k]
            pred_logits_query += pos_z[k] * pred_logits_query_list[k]
            z_pos_list[k].append(pos_z[k])

        preds = torch.argmax(pred_logits_query, dim = -1)
        acc = torch.mean(preds == query_lbl.to(device=model_device), dtype=torch.float32) * 100.
        loss.append(nll_query)
        accuracy.append(acc)

    loss_tensor = torch.stack(loss)
    acc_tensor = torch.stack(accuracy)

    loss_mean = loss_tensor.mean()
    acc_mean = acc_tensor.mean()

    sqrt_nsample = (torch.tensor(num_task, dtype=torch.float32, device=model_device)).sqrt()
    loss_95ci = 1.96 * loss_tensor.std(unbiased=True) / sqrt_nsample
    acc_95ci = 1.96 * acc_tensor.std(unbiased=True) / sqrt_nsample

    z_pos_mean_list = []
    z_pos_95ci_list = []
    for k in range(len(variational_obj_list)):
        z_pos_tensor = torch.stack(z_pos_list[k])
        z_pos_mean = z_pos_tensor.mean()
        z_pos_95ci = 1.96 * z_pos_tensor.std(unbiased=True) / sqrt_nsample
        z_pos_mean_list.append(z_pos_mean.item())
        z_pos_95ci_list.append(z_pos_95ci.item())

    if return_eta:
        return loss_mean.item(), loss_95ci.item(), acc_mean.item(), acc_95ci.item(), z_pos_mean_list, z_pos_95ci_list
    else:
        return loss_mean.item(), loss_95ci.item(), acc_mean.item(), acc_95ci.item()
