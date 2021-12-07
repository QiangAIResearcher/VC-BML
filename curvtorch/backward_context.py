import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaConv1d, MetaConv2d, MetaConv3d, MetaLinear, MetaBatchNorm1d, MetaBatchNorm2d

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from curvtorch.util_curv import iter_modules

class CurvatureContext(object, metaclass=ABCMeta):
    def __init__(self, net):
        self.net = net

    @abstractmethod
    def conv_forward_hook(self, module, inputs, output):
        pass

    @abstractmethod
    def linear_forward_hook(self, module, inputs, output):
        pass

    @abstractmethod
    def unitbn1d_forward_hook(self, module, inputs, output):
        pass

    @abstractmethod
    def unitbn2d_forward_hook(self, module, inputs, output):
        pass

    def __enter__(self):
        self._hooks = []
        for module in iter_modules(self.net):
            if isinstance(module, (MetaLinear, nn.Linear)):
                self._hooks.append(module.register_forward_hook(self.linear_forward_hook))
            elif isinstance(module, (MetaConv1d, MetaConv2d, MetaConv3d, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self._hooks.append(module.register_forward_hook(self.conv_forward_hook))
            elif isinstance(module, (MetaBatchNorm1d, nn.BatchNorm1d)):
                self._hooks.append(module.register_forward_hook(self.unitbn1d_forward_hook))
            elif isinstance(module, (MetaBatchNorm2d, nn.BatchNorm2d)):
                self._hooks.append(module.register_forward_hook(self.unitbn2d_forward_hook))
        return self

    def __exit__(self, type, value, traceback):
        for hook in self._hooks:
            hook.remove()
        del self._hooks


class KroneckerContext(CurvatureContext):
    def __init__(self, net):
        super().__init__(net)

    def linear_forward_hook(self, module, inputs, output):
        def backward_hook(grad):
            activation = inputs[0]
            if module.bias is not None:
                activation = torch.cat([activation, torch.ones_like(activation[:, :1])], dim=1)
            self.activation_outprod_denom[module] = float(activation.size(0))
            self.preact_grad_outprod_denom[module] = float(grad.size(0))
            with torch.no_grad():
                self.activation_outprod[module] = torch.mm(activation.t(), activation)
                self.preact_grad_outprod[module] = torch.mm(grad.t(), grad)
        output.register_hook(backward_hook)

    def conv_forward_hook(self, module, inputs, output):
        def backward_hook(grad):
            act_unfold = \
                F.unfold(inputs[0], module.kernel_size, module.dilation, module.padding, module.stride)\
                    .transpose(1, -1)
            self.activation_outprod_denom[module] = float(act_unfold.size(0))
            activation = act_unfold.reshape(-1, act_unfold.size(-1))
            if module.bias is not None:
                activation = torch.cat([activation, torch.ones_like(activation[:, :1])], dim=1)
            grad = grad.permute(0, 2, 3, 1).reshape(-1, module.out_channels)
            self.preact_grad_outprod_denom[module] = float(grad.size(0))
            with torch.no_grad():
                self.activation_outprod[module] = torch.mm(activation.t(), activation)
                self.preact_grad_outprod[module] = torch.mm(grad.t(), grad)
        output.register_hook(backward_hook)

    def unitbn1d_forward_hook(self, module, inputs, output):
        def backward_hook(grad):
            inp_norm = F.batch_norm(
                inputs[0], running_mean=torch.mean(inputs[0], 0).detach_(), running_var=torch.var(inputs[0], 0).detach_(),
                momentum=module.momentum, eps=module.eps
            )
            with torch.no_grad():
                weight_grad = grad * inp_norm
                off_diags = weight_grad * grad
                # dim; batch_size x n_out_ch x 2 x 2
                unitbd = torch.stack(
                    [torch.stack([weight_grad ** 2, off_diags], dim=-1), torch.stack([off_diags, grad ** 2], dim=-1)],
                    dim=-1
                )
                self.batchnorm_grad_outprod[module] = torch.sum(unitbd, dim=0)
                self.batchnorm_grad_outprod_denom[module] = float(unitbd.size(0))
        output.register_hook(backward_hook)

    def unitbn2d_forward_hook(self, module, inputs, output):
        def backward_hook(grad):
            inp_norm = F.batch_norm(
                inputs[0], running_mean=torch.mean(inputs[0], (0, -2, -1)).detach_(),
                running_var=torch.var(inputs[0], (0, -2, -1)).detach_(), momentum=module.momentum, eps=module.eps
            )
            with torch.no_grad():
                weight_grad = grad * inp_norm
                off_diags = weight_grad * grad
                # dim: batch_size x n_out_ch x n_patches x 2 x 2
                unitbd = torch.stack(
                    [torch.stack([weight_grad ** 2, off_diags], dim=-1), torch.stack([off_diags, grad ** 2], dim=-1)],
                    dim=-1
                )
                self.batchnorm_grad_outprod[module] = torch.sum(unitbd, dim=[0, 2, 3])
                self.batchnorm_grad_outprod_denom[module] = float(unitbd.size(0) * unitbd.size(2) * unitbd.size(3))
        output.register_hook(backward_hook)

    def __enter__(self):
        self.activation_outprod = OrderedDict()
        self.preact_grad_outprod = OrderedDict()
        self.activation_outprod_denom = OrderedDict()
        self.preact_grad_outprod_denom = OrderedDict()
        # for batchnorm fisher
        self.batchnorm_grad_outprod = OrderedDict()
        self.batchnorm_grad_outprod_denom = OrderedDict()
        return super().__enter__()
