import torch
import torch.nn as nn
import numpy as np
import math
from torchmeta.modules import (MetaModule, MetaConv2d, MetaLinear, MetaBatchNorm1d, MetaBatchNorm2d)
from abc import ABCMeta, abstractmethod

from modules import MetaModuleMonteCarlo, MetaSequential, \
    MetaLinearMonteCarlo, MetaConv2dMonteCarlo, \
    MetaBatchNorm1dMonteCarlo, MetaBatchNorm2dMonteCarlo, MaxPool2dMonteCarlo 


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, maxpool_kernel_size=2):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(maxpool_kernel_size)
    )

def linear(in_dim, out_dim, bias=True):
    return MetaSequential(
        MetaLinear(in_dim, out_dim, bias=bias),
        MetaBatchNorm1d(out_dim, momentum=1., track_running_stats=False),
        nn.ReLU()
    )

def conv2dmc(in_channels, out_channels, kernel_size=3, stride=1, padding=1, maxpool_kernel_size=2):
    return MetaSequential(
        MetaConv2dMonteCarlo(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        MetaBatchNorm2dMonteCarlo(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        MaxPool2dMonteCarlo(maxpool_kernel_size)
    )

def linearmc(in_dim, out_dim, bias=True):
    return MetaSequential(
        MetaLinearMonteCarlo(in_dim, out_dim, bias=bias),
        MetaBatchNorm1dMonteCarlo(out_dim, momentum=1., track_running_stats=False),
        nn.ReLU()
    )


# use the initialisation that is same as default for Linear, Conv and BatchNorm
@torch.no_grad()
def init_param(m):
    if isinstance(m, (nn.Conv2d, MetaConv2d, MetaConv2dMonteCarlo, nn.Linear, MetaLinear, MetaLinearMonteCarlo)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, MetaBatchNorm1d, MetaBatchNorm2d,
                        MetaBatchNorm1dMonteCarlo, MetaBatchNorm2dMonteCarlo)):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class _NeuralNet(metaclass=ABCMeta):
    def __init__(self, num_way=5, img_dim=28, hidden_dims=(256, 128, 64, 64)):
        super(_NeuralNet, self).__init__()
        self.layer_dims = (img_dim * img_dim,) + hidden_dims + (num_way,)
        fc_modules = self.fc_modules_list()
        self.fc = MetaSequential(*fc_modules)

    @abstractmethod
    def fc_modules_list(self):
        pass


class _ConvNet(metaclass=ABCMeta):
    def __init__(self, num_in_ch=1, num_conv_layer=4, num_filter=64, kernel_size=3, maxpool_kernel_size=2,
                 stride=1, padding=1, fc_in_dim=1, num_fc_layer=1, num_fc_hidden=5):
        super(_ConvNet, self).__init__()
        self.conv_channel_ls = (num_in_ch,) + (num_filter,) * num_conv_layer
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * num_conv_layer
        else:
            self.kernel_size = kernel_size

        self.fc_dim_ls = (self.conv_channel_ls[-1] * fc_in_dim * fc_in_dim,) + (num_fc_hidden,) * num_fc_layer

        conv_modules = self.conv_modules_list(maxpool_kernel_size=maxpool_kernel_size, stride=stride, padding=padding)
        fc_modules = self.fc_modules_list()

        self.conv = MetaSequential(*conv_modules)
        if len(fc_modules) != 0:
            self.fc = MetaSequential(*fc_modules)
        else:
            self.fc = None

    @abstractmethod
    def conv_modules_list(self, maxpool_kernel_size, stride, padding):
        pass

    @abstractmethod
    def fc_modules_list(self):
        pass


class NeuralNet(_NeuralNet, MetaModule):
    def __init__(self, num_way=5, img_dim=28, hidden_dims=(256, 128, 64, 64)):
        super(NeuralNet, self).__init__(num_way, img_dim, hidden_dims)

    def fc_modules_list(self):
        return [linear(self.layer_dims[i - 1], self.layer_dims[i]) for i in range(1, len(self.layer_dims))]

    def forward(self, x, param=None):
        x = x.view(x.size(0), -1)
        output = self.fc(x, params=self.get_subdict(param, 'fc'))
        return output


class ConvNet(_ConvNet, MetaModule):
    def __init__(self, num_way=5, num_in_ch=1, num_conv_layer=4, num_filter=64, kernel_size=3, maxpool_kernel_size=2,
                 stride=1, padding=1, fc_in_dim=1, num_fc_layer=1, num_fc_hidden=5):
        super(ConvNet, self).__init__(num_in_ch, num_conv_layer, num_filter, kernel_size, maxpool_kernel_size,
                 stride, padding, fc_in_dim, num_fc_layer, num_fc_hidden)
        self.classifier = MetaLinear(self.fc_dim_ls[-1], num_way, bias=True)

    def conv_modules_list(self, maxpool_kernel_size, stride, padding):
        return [
            conv2d(self.conv_channel_ls[i - 1], self.conv_channel_ls[i], kernel_size=self.kernel_size[i - 1],
                   maxpool_kernel_size=maxpool_kernel_size, stride=stride, padding=padding)
            for i in range(1, len(self.conv_channel_ls))
        ]

    def fc_modules_list(self):
        return[linear(self.fc_dim_ls[i - 1], self.fc_dim_ls[i]) for i in range(1, len(self.fc_dim_ls))]

    def forward(self, x, param=None):
        conv = self.conv(x, params=self.get_subdict(param, 'conv'))
        conv = conv.view(conv.size(0), -1)
        if self.fc is not None:
            fc = self.fc(conv, params=self.get_subdict(param, 'fc'))
            output = self.classifier(fc, params=self.get_subdict(param, 'classifier'))
        else:
            output = self.classifier(conv, params=self.get_subdict(param, 'classifier'))
        return output


class Convolution(_ConvNet, MetaModule):

    def __init__(self, num_way=5, num_in_ch=1, num_conv_layer=4, num_filter=64, kernel_size=3, maxpool_kernel_size=2,
                 stride=1, padding=1, fc_in_dim=1, num_fc_layer=0, num_fc_hidden=32):
        super(Convolution, self).__init__(num_in_ch, num_conv_layer, num_filter, kernel_size, maxpool_kernel_size,
                 stride, padding, fc_in_dim, num_fc_layer, num_fc_hidden)
        # self.classifier = MetaLinear(self.fc_dim_ls[-1], num_way, bias=True)
        self.output_dim = self.fc_dim_ls[-1]
        self.num_way = num_way

    def conv_modules_list(self, maxpool_kernel_size, stride, padding):
        return [
            conv2d(self.conv_channel_ls[i - 1], self.conv_channel_ls[i], kernel_size=self.kernel_size[i - 1],
                   maxpool_kernel_size=maxpool_kernel_size, stride=stride, padding=padding)
            for i in range(1, len(self.conv_channel_ls))
        ]

    def fc_modules_list(self):
        return[linear(self.fc_dim_ls[i - 1], self.fc_dim_ls[i]) for i in range(1, len(self.fc_dim_ls))]

    def forward(self, x, param=None):
        output = self.conv(x, params=self.get_subdict(param, 'conv'))
        output = output.view(output.size(0), -1)

        if self.fc is not None:
            output = self.fc(output, params=self.get_subdict(param, 'fc'))

        return output


class NeuralNetMonteCarlo(_NeuralNet, MetaModuleMonteCarlo):
    def __init__(self, num_way=5, img_dim=28, hidden_dims=(256, 128, 64, 64)):
        super(NeuralNetMonteCarlo, self).__init__(num_way, img_dim, hidden_dims)

    def fc_modules_list(self):
        return [linearmc(self.layer_dims[i - 1], self.layer_dims[i]) for i in range(1, len(self.layer_dims))]

    def forward(self, x, param=None, mean=None, cov=None):
        x = x.view(x.size(0), -1) if param is not None else x.view(x.size(0), x.size(1), -1)
        output = self.fc(x, params=self.get_subdict(param, 'fc'),
                         mean=self.get_subdict(mean, 'fc'), cov=self.get_subdict(cov, 'fc'))
        return output


class ConvNetMonteCarlo(_ConvNet, MetaModuleMonteCarlo):
    def __init__(self, num_way=5, num_in_ch=1, num_conv_layer=4, num_filter=64, kernel_size=3, maxpool_kernel_size=2,
                 stride=1, padding=1, fc_in_dim=1, num_fc_layer=1, num_fc_hidden=5):
        super(ConvNetMonteCarlo, self).__init__(num_in_ch, num_conv_layer, num_filter, kernel_size, maxpool_kernel_size,
                                      stride, padding, fc_in_dim, num_fc_layer, num_fc_hidden)
        self.classifier = MetaLinearMonteCarlo(self.fc_dim_ls[-1], num_way, bias=True)

    def conv_modules_list(self, maxpool_kernel_size, stride, padding):
        return [
            conv2dmc(self.conv_channel_ls[i - 1], self.conv_channel_ls[i], kernel_size=self.kernel_size[i - 1],
                   maxpool_kernel_size=maxpool_kernel_size, stride=stride, padding=padding)
            for i in range(1, len(self.conv_channel_ls))
        ]

    def fc_modules_list(self):
        return [linearmc(self.fc_dim_ls[i - 1], self.fc_dim_ls[i]) for i in range(1, len(self.fc_dim_ls))]

    def forward(self, x, param=None, mean=None, cov=None):
        conv = self.conv(x, params=self.get_subdict(param, 'conv'),
                         mean=self.get_subdict(mean, 'conv'), cov=self.get_subdict(cov, 'conv'))
        
        conv = conv.view(conv.size(0), -1) if param is not None else conv.view(conv.size(0), conv.size(1), -1)
        
        if self.fc is not None:
            fc = self.fc(conv, params=self.get_subdict(param, 'fc'),
                         mean=self.get_subdict(mean, 'fc'), cov=self.get_subdict(cov, 'fc'))
            output = self.classifier(
                fc, params=self.get_subdict(param, 'classifier'),
                mean=self.get_subdict(mean, 'classifier'), cov=self.get_subdict(cov, 'classifier'))
        else:
            output = self.classifier(
                conv, params=self.get_subdict(param, 'classifier'),
                mean=self.get_subdict(mean, 'classifier'), cov=self.get_subdict(cov, 'classifier'))
        return output

        