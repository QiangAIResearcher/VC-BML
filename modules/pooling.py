import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import _MaxPoolNd

class MaxPool2dMonteCarlo(_MaxPoolNd):
    __doc__ = nn.MaxPool2d.__doc__

    def forward(self, input):
        if input.dim() == 5:
            # input shape (n_sample, batch, ch, inp_dim_1, inp_dim_2)
            maxpool = F.max_pool2d(input.reshape(-1, *input.size()[2:]), self.kernel_size, self.stride,
                                   self.padding, self.dilation, self.ceil_mode,
                                   self.return_indices)
            return maxpool.reshape(input.size(0), input.size(1), *maxpool.size()[1:])
        elif input.dim() == 4:
            return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode,
                                self.return_indices)
        else:
            raise NotImplementedError('Input should be 4-D if params is not None, or 5-D if mean & cov is not None, but got {}D'.format(input.dim()))