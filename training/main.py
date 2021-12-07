import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import datetime
import os
import json
import random
from platform import system
from warnings import filterwarnings

import optim
from config.configuration import get_run_name
from data_generate.dataset import FewShotImageDataset
from data_generate.sampler import SuppQueryBatchSampler
from training import model as models
from training.variational import VariationalApprox, var_approx_beta
from training.boml import metatrain_seqdataset_amvi
from training.util import enlist_transformation

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(config, run_spec, seed=0):

    torch.manual_seed(seed)

    start_datetime = datetime.datetime.now()
    experiment_date = '{:%Y-%m-%d_%H:%M:%S}'.format(start_datetime)
    config['experiment_parent_dir'] = os.path.join(config['run_dir'], get_run_name(config['dataset_ls']))
    config['experiment_dir'] = os.path.join(config['experiment_parent_dir'],
                                            '{}_{}_{}'.format(run_spec, experiment_date, seed))

    os.system("echo 'running {}_{} seed {}'".format(run_spec, experiment_date, seed)) if system() == 'Linux' \
        else print('running {}_{} seed {}'.format(run_spec, experiment_date, seed))

    # save config json file
    if not os.path.exists(config['experiment_dir']):
        os.makedirs(config['experiment_dir'])
    with open(os.path.join(
            config['experiment_dir'],
            'config{}_{}.json'.format(0 if config['completed_task_idx'] is None
                                      else config['completed_task_idx'] + 1, run_spec)
    ), 'w') as outfile:
        outfile.write(json.dumps(config, indent=4))

    # define result directory and previous result directory if applicable
    # define tensorboard writer
    if config['completed_task_idx'] is not None:
        completed_result_dir = os.path.join(
            os.path.join(os.path.join(config['run_dir'], get_run_name(config['dataset_ls'])),
                         config['completed_exp_name']),
            'result'
        )
    else:
        completed_result_dir = None
    writer = SummaryWriter(os.path.join(config['experiment_dir'], 'logtb'))
    result_dir = os.path.join(config['experiment_dir'], 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # define model
    model = getattr(models, config['net'])(**config['net_kwargs']).to(device=config['device'])
    # detach model meta-parameters
    for param in model.meta_parameters():
        param.requires_grad = False

    # define variational object
    var_approx_list = []
    cuda_devices = ["cuda:{}".format(i) for i in config["k"]]
    for i, device in enumerate(cuda_devices):

        if config["completed_task_idx"] is not None:
            mean_init = torch.load(
                os.path.join(completed_result_dir, 'mean{}_varapprox{}.pt'.format(config["completed_task_idx"], i)),
                map_location="cpu"
            )
            covar_init = torch.load(
                os.path.join(completed_result_dir, 'covar{}_varapprox{}.pt'.format(config["completed_task_idx"], i)),
                map_location="cpu"
            )
            approx = VariationalApprox(
                device, config["num_mc_sample"], mean_init=mean_init, 
                covar_init=covar_init, init_optim_lrsch=False
            )
        else:

            model_temp = getattr(models, config['net'])(**config['net_kwargs'])
            approx = VariationalApprox(
                device, config["num_mc_sample"], model=model_temp, init_optim_lrsch=False
            )

        approx.update_mean_cov()
        var_approx_list.append(approx)


    # var approximation of beta
    if len(config["k"]) > 1:

        alpha = config["alpha"] if config['completed_task_idx'] is None else \
            np.load(os.path.join(completed_result_dir, "alpha{}.npy".format(config['completed_task_idx'])))

        var_beta = var_approx_beta(alpha, len(config["k"]), config["device"])
    else:
        var_beta = None
    
    if config['completed_task_idx'] is not None:
        prev_glob_step \
            = torch.load(os.path.join(completed_result_dir, 'prev_glob_step{}.pt'.format(config['completed_task_idx'])))
        evalset = torch.load(
            os.path.join(completed_result_dir, 'evalset{}.pt'.format(config['completed_task_idx'])))
    else:
        prev_glob_step = 0
        evalset = []

    # run partial num of datasets or all
    num_dataset_to_run = len(config['dataset_ls']) if config['num_dataset_to_run'] == 'all' \
        else config['num_dataset_to_run']

    for task_idx, task in enumerate(config['dataset_ls'][:num_dataset_to_run], 0):
        if config['completed_task_idx'] is not None and config['completed_task_idx'] >= task_idx:
            pass
        else:
            # split directory for this dataset
            split_dir = os.path.join(os.path.join(config['data_dir'], task, config['split_folder']))

            # define optimiser and lr scheduler
            for var_approx in var_approx_list:
                var_approx.optimizer = getattr(optim, config[task]['optim_outer_name']) \
                    (list(var_approx.mean.values()) + list(var_approx.covar.values()), **config[task]['optim_outer_kwargs'])
                if config[task]['lr_sch_outer_name'] is None:
                    var_approx.lr_scheduler = None
                else:
                    var_approx.lr_scheduler = getattr(lr_scheduler, config[task]['lr_sch_outer_name']) \
                        (var_approx.optimizer, **config[task]['lr_sch_outer_kwargs'])

            # define transformation of images
            transformation = transforms.Compose(
                enlist_transformation(img_resize=config['img_resize'], is_grayscale=config['is_grayscale'],
                                      device=config['device'], img_normalise=config[task]['img_normalise'])
            )

            trainset = FewShotImageDataset(
                task_list=np.load(os.path.join(split_dir, 'metatrain.npy'), allow_pickle=True).tolist(),
                supercls=config[task]['supercls'], img_lvl=int(config[task]['supercls']) + 1, transform=transformation,
                relabel=None, device=config['device'], cuda_img_tensor=config['cuda_img_tensor'],
                verbose='{} trainset'.format(task)
            )

            # define & append meta-evaluation dataset and dataloader
            evalset.append(FewShotImageDataset(
                task_list=np.load(os.path.join(split_dir, 'metatest.npy'), allow_pickle=True).tolist(),
                supercls=config[task]['eval_supercls'], img_lvl=int(config[task]['eval_supercls']) + 1,
                transform=transformation, relabel=None, device=config['device'],
                cuda_img_tensor=config['cuda_img_tensor'], verbose='{} evalset'.format(task)
            ))

            # meta-training
            metatrain_seqdataset_amvi(
                model=model, var_approx_list=var_approx_list, var_beta=var_beta, model_device=config["device"], 
                trainset=trainset, evalset=evalset, outer_kl_scale=config[task]['outer_kl_scale'],
                nstep_outer=config[task]['nstep_outer'], nstep_inner=config[task]['nstep_inner'],
                lr_inner=config[task]['lr_inner'], first_order=config[task]['first_order'], seqtask=config['seqtask'],
                num_task_per_itr=config[task]['num_task_per_itr'], task_by_supercls=config[task]['task_by_supercls'],
                num_way=config['net_kwargs']['num_way'], num_shot=config[task]['num_shot'],
                num_query_per_cls=config[task]['num_query_per_cls'], eval_prev_task=True,
                eval_per_num_iter=config[task]['eval_per_num_iter'], num_eval_task=config[task]['num_eval_task'],
                eval_task_by_supercls=config[task]['eval_task_by_supercls'],
                nstep_inner_eval=config[task]['nstep_inner_eval'], writer=writer, task_idx=task_idx,
                prev_glob_step=prev_glob_step, verbose=task, sample_batch=config["sample_batch"]
            )

            # update global step
            prev_glob_step += config[task]['nstep_outer']

            # update mean and covariance of meta-parameters
            for var_approx in var_approx_list:
                var_approx.update_mean_cov()

            # save mean, covariance, mean_old and covar_old
            torch.save(prev_glob_step, f=os.path.join(result_dir, 'prev_glob_step{}.pt'.format(task_idx)))
            torch.save(evalset, f=os.path.join(result_dir, 'evalset{}.pt'.format(task_idx)))
            for i,var_approx in enumerate(var_approx_list):
                torch.save(var_approx.mean, f=os.path.join(result_dir, 'mean{}_varapprox{}.pt'.format(task_idx, i)))
                torch.save(var_approx.covar, f=os.path.join(result_dir, 'covar{}_varapprox{}.pt'.format(task_idx, i)))
            
            if var_beta is not None:
                gamma = np.vstack([var_beta.prior_alpha1.detach().cpu().numpy(), \
                    var_beta.prior_alpha2.detach().cpu().numpy()])
                np.save(os.path.join(result_dir, "alpha{}.npy".format(task_idx)), gamma)

        torch.cuda.empty_cache()

    # check how long it ran
    run_time_print = '\ncompleted in {}'.format(datetime.datetime.now() - start_datetime)
    os.system('echo "{}"'.format(run_time_print)) if system() == 'Linux' else print(run_time_print)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('VC-BML Sequential Dataset')
    parser.add_argument('--config_path', type=str, help='Path of .json file to import config from')
    args = parser.parse_args()
    # load config file
    jsonfile = open(str(args.config_path))
    config = json.loads(jsonfile.read())
    # train
    train(config=config, run_spec=os.path.splitext(os.path.split(args.config_path)[-1])[0], seed=random.getrandbits(24))
