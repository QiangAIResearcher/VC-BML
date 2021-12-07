import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.module import MetaModuleMonteCarlo

class MetaLinearMonteCarlo(nn.Linear, MetaModuleMonteCarlo):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None, mean=None, cov=None):
        # either params or (mean, var) must be not None
        # input shape (n_sample, batch, in_feat) or (batch, in_feat) if params=None
        if params is not None:
            return F.linear(input, params['weight'], params.get('bias', None))
        # mean and var same shape as ori params
        else:
            inp_sample = input.reshape(input.size(0) * input.size(1), *input.size()[2:])

            preact_mean = F.linear(inp_sample, mean['weight'], mean.get('bias', None))
            preact_cov = F.linear(inp_sample ** 2, cov['weight'], cov.get('bias', None))

            preact = preact_mean + torch.sqrt(preact_cov) * torch.randn_like(preact_mean)
            return preact.reshape(input.size(0), input.size(1), *preact.size()[1:])
