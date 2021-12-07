import torch.nn as nn
from torchmeta.modules import \
    MetaLinear, MetaConv1d, MetaConv2d, MetaConv3d, MetaBatchNorm1d, MetaBatchNorm2d, MetaBatchNorm3d

def kronecker(a, b):
    return a.unsqueeze(-2).unsqueeze(-1).mul(b.unsqueeze(-3).unsqueeze(-2)).flatten(-4, -3).flatten(-2)

def iter_named_modules(net):
    mods_to_iter = (MetaLinear, MetaConv1d, MetaConv2d, MetaConv3d,
                    MetaBatchNorm1d, MetaBatchNorm2d, MetaBatchNorm3d,
                    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for name, mod in net.named_modules():
        if isinstance(mod, mods_to_iter):
            yield name, mod

def iter_modules(net):
    mods_to_iter = (MetaLinear, MetaConv1d, MetaConv2d, MetaConv3d,
                    MetaBatchNorm1d, MetaBatchNorm2d, MetaBatchNorm3d,
                    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for mod in net.modules():
        if isinstance(mod, mods_to_iter):
            yield mod
