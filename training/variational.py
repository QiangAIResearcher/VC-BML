
from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np

import torch
import optim
import torch.optim.lr_scheduler as lr_scheduler


class Prior(ABC):
    def __init__(self, config):
        self.config = config
        self.device = config['device']

    @abstractmethod
    def add_task(self):
        pass

    @abstractmethod
    def record_usage(self, usage, index=None):
        pass

    @abstractmethod
    def nl_prior(self, normalize=False):
        pass


class VariationalApprox(object):

    def __init__(self, device, num_mc_sample, model=None, mean_init=None, covar_init=None, 
                init_optim_lrsch=True, optim_outer_name=None, optim_outer_kwargs=None, 
                lr_sch_outer_name=None, lr_sch_outer_kwargs=None):

        assert (model is not None) or (mean_init is not None and covar_init is not None)

        self.num_mc_sample = num_mc_sample
        self.device = device

        self.mean = self.init_mean(model, mean_init)
        self.covar = self.init_covariance(model, covar_init)

        if model is not None:
            self.detach_model_params(model)
        else:
            self.detach_mean_covar_params(mean_init, covar_init)
        
        # define optimiser and lr scheduler
        if init_optim_lrsch:
            self.optimizer = getattr(optim, optim_outer_name)\
                (list(self.mean.values()) + list(self.covar.values()), **optim_outer_kwargs)

            if lr_sch_outer_name is None:
                self.lr_scheduler = None
            else:
                self.lr_scheduler = getattr(lr_scheduler, lr_sch_outer_name)\
                    (self.optimizer, **lr_sch_outer_kwargs)
        else:
            self.optimizer = None
            self.lr_scheduler = None
        
        

    def init_mean(self, model, mean_init):

        if model is not None:
            return OrderedDict([
                (name, param.clone().detach().to(device=self.device).requires_grad_(True)) \
                    for (name, param) in model.meta_named_parameters()
            ])
        else:
            return OrderedDict([
                (name, param.clone().detach().to(device=self.device).requires_grad_(True)) \
                    for (name, param) in mean_init.items()
            ])


    def init_covariance(self, model, covar_init):

        if model is not None:
            return OrderedDict([
                (name, param.new_full(param.size(), fill_value=-10.).to(device=self.device).requires_grad_(True)) \
                    for (name, param) in model.meta_named_parameters()
            ])
        else:
            return OrderedDict([
                (name, param.clone().detach().to(device=self.device).requires_grad_(True)) \
                    for (name, param) in covar_init.items()
            ])

    def exp_covar(self, covar):
        return OrderedDict([(name, torch.exp(cov)) for name, cov in covar.items()])


    def sample_params(self, n_sample=None, detach_mean_cov=False):
        params = OrderedDict()
        for (name, mean), cov in zip(self.mean.items(), self.exp_covar(self.covar).values()):
            if n_sample == 1:
                params_sample_size = [*mean.size()]
            elif n_sample is None:
                params_sample_size = [self.num_mc_sample, *mean.size()]
            else:
                params_sample_size = [n_sample, *mean.size()]

            params[name] = \
                (mean.detach() + cov.detach().sqrt()
                 * torch.randn(*params_sample_size, dtype=mean.dtype, device=mean.device)).requires_grad_(True) \
                    if detach_mean_cov \
                else mean + cov.sqrt() * torch.randn(*params_sample_size, dtype=mean.dtype, device=mean.device)

        return params
    
    def detach_model_params(self, model):
        for param in model.meta_parameters():
            param.requires_grad = False
    
    def detach_mean_covar_params(self, mean, covar):
        for mean_tensor in mean.values():
            mean_tensor.requires_grad = False
        for covar_tensor in covar.values():
            covar_tensor.requires_grad = False

    def update_mean_cov(self):
        self.mean_old = OrderedDict([(name, mu.clone().detach()) for (name, mu) in self.mean.items()])
        self.covar_old = OrderedDict([(name, cov.clone().detach()) for (name, cov) in self.covar.items()])


class var_approx_beta(object):

    def __init__(self, alpha, k, device):
        
        if isinstance(alpha, int):
            self.prior_alpha1 = torch.tensor(np.ones([k-1]), device=device).float()
            self.prior_alpha2 = torch.tensor(np.tile([alpha], [k-1]), device=device).float()
        else:
            assert alpha.shape[1] == k-1
            self.prior_alpha1 = torch.tensor(alpha[0], device=device).float()
            self.prior_alpha2 = torch.tensor(alpha[1], device=device).float()

        self.k = k
        self.device = device
    
    def update_posterior(self, eta):

        self.var_gamma1 = self.prior_alpha1 +  eta[:self.k-1]

        s = torch.sum(eta)
        sum_backward = []
        for x in eta[:-1]:
            s = s - x 
            sum_backward.append(s.view(-1))

        self.var_gamma2 = self.prior_alpha2 + torch.cat(sum_backward, dim=0).float()
    

    def update_prior(self):

        self.prior_alpha1 = self.var_gamma1
        self.prior_alpha2 = self.var_gamma2
    
    def set_gamma(self, gamma1, gamma2):
        
        self.var_gamma1 = gamma1
        self.var_gamma2 = gamma2