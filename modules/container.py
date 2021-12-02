import torch.nn as nn
from torchmeta.modules.module import MetaModule

from modules.module import MetaModuleMonteCarlo


class MetaSequential(nn.Sequential, MetaModuleMonteCarlo, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None, mean=None, cov=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModuleMonteCarlo):
                input = module(input, params=self.get_subdict(params, name),
                               mean=self.get_subdict(mean, name), cov=self.get_subdict(cov, name))
            elif isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                    '(inheriting from `nn.Module`), a `MetaModule` or a `MetaModuleMonteCarlo`. '
                    'Got type: `{0}`'.format(type(module)))
        return input