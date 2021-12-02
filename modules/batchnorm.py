import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from modules.module import MetaModuleMonteCarlo

class _MetaBatchNormMonteCarlo(_BatchNorm, MetaModuleMonteCarlo):
    def forward(self, input, params=None, mean=None, cov=None):
        self._check_input_dim(input)

        if params is not None:
            # exponential_average_factor is self.momentum set to
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            weight = params.get('weight', None)
            bias = params.get('bias', None)

            return F.batch_norm(
                input, self.running_mean, self.running_var, weight, bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            # input shape: (num_sample, batch_size, channel, height, width) or (num_sample, batch_size, num_features)
            # get dimensions to mean and var over (ie. all except num_sample and channel dims)
            stats_calc_dim = tuple(i for i in range(input.dim()) if i != 0 and i != 2)
            # shape (n_sample, batch, ch, height, width)
            norm = (input - input.mean(stats_calc_dim, keepdim=True)) / torch.sqrt(input.var(stats_calc_dim, keepdim=True) + self.eps)
            # get view dimensions to expand mean and var so that ndim matches
            stats_view_dim = [mean['weight'].size(0) if i == 2 else 1 for i in range(input.dim())]

            bn_mean = norm * mean['weight'].view(*stats_view_dim) + mean['bias'].view(*stats_view_dim)
            bn_var = norm ** 2 * cov['weight'].view(*stats_view_dim) + cov['bias'].view(*stats_view_dim)
            return bn_mean + torch.sqrt(bn_var) * torch.randn_like(bn_mean)


class MetaBatchNorm1dMonteCarlo(_MetaBatchNormMonteCarlo):
    __doc__ = nn.BatchNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D (non-sampling) or 3D (sampling) but got {}D input'.format(input.dim()))


class MetaBatchNorm2dMonteCarlo(_MetaBatchNormMonteCarlo):
    __doc__ = nn.BatchNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError('expected 4D (non-sampling) or 5D (sampling) but got {}D input'.format(input.dim()))
