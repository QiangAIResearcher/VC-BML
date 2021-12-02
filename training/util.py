import torch
import torchvision.transforms as transforms

import os
import random
from collections import OrderedDict
from PIL import Image

from data_generate import transformations as transfm

def split_path(path):
    # remove trailing '/' if any
    split_ls = os.path.normpath(path).split('/')
    if '' in split_ls:
        split_ls.remove('')
    if '.' in split_ls:
        split_ls.remove('.')
    return split_ls


def concat_param(weight, bias):
    if bias is not None:
        return torch.cat([weight, bias.unsqueeze(-1)], dim=-1)
    else:
        return weight


def enlist_transformation(img_resize=None, resize_interpolation='BILINEAR', is_grayscale=False, device=None, img_normalise=True):
    transform_ls = []
    if img_resize is not None:
        transform_ls.append(transforms.Resize(size=(img_resize, img_resize), interpolation=getattr(Image, resize_interpolation)))
    if is_grayscale:
        transform_ls.append(transforms.Grayscale())
    transform_ls.append(transforms.ToTensor())
    if device is not None:
        transform_ls.append(transfm.ToDevice(device=device))
    if img_normalise:
        transform_ls.append(transfm.NormaliseMinMax())
    return transform_ls


def enlist_montecarlo_param(param, num_mc_sample):
    return [OrderedDict([(name, params[i, ...]) for (name, params) in param.items()]) for i in range(num_mc_sample)]


def concat_montecarlo_param(list):
    return OrderedDict([(name, torch.stack([list[i][name] for i in range(len(list))], dim=0)) for name in list[0].keys()])


def kldiv_mvn_diagcov(mean_p, cov_p, mean_q, cov_q):
    kl_layer_ls = []
    for mu_p, sig_p, mu_q, sig_q in zip(mean_p.values(), cov_p.values(), mean_q.values(), cov_q.values()):
        mean_diff = mu_q - mu_p
        sig_q_inv = 1 / sig_q
        kl_layer = torch.log(sig_q).sum() - torch.log(sig_p).sum() - mu_p.numel() + (sig_q_inv * sig_p).sum() \
                   + ((mean_diff * sig_q_inv) * mean_diff).sum()
        kl_layer_ls.append(kl_layer)
    return sum(kl_layer_ls) / 2


def get_accuracy(labels, outputs=None, inputs=None, model=None, param=None):
    
    if outputs is None:
        outputs = model(inputs, param=param)
    preds = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
    acc = torch.mean(preds == labels, dtype=torch.float32) * 100.
    return acc

def get_zk_likelihood(var_obj, prior_mean, prior_covar, pos_mean, pos_covar, var_beta, kth, device, batch_size, num_samples=1):

    kl = 0.01 * kldiv_mvn_diagcov(
        mean_p = pos_mean, cov_p=var_obj.exp_covar(pos_covar),
        mean_q=prior_mean, cov_q=var_obj.exp_covar(prior_covar)
    ) / (batch_size * num_samples)

    exp_pi = torch.tensor(1.0, device=device)
    for i in range(kth):
        exp_pi = exp_pi * (1 - var_beta.var_gamma1[i].to(device=device) / \
                (var_beta.var_gamma1[i].to(device=device) + var_beta.var_gamma2[i].to(device=device)))
    if kth < var_beta.k - 1:
        exp_pi = exp_pi * var_beta.var_gamma1[kth].to(device=device) / \
                (var_beta.var_gamma1[kth].to(device=device) + var_beta.var_gamma2[kth].to(device=device))

    return exp_pi - kl

