import torch.nn.functional as F

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='sum'):
    # input shape: (n_sample, batch, num_way)
    # target shape: (n_sample, batch)
    inp_reshape = input.reshape(input.size(0) * input.size(1), *input.size()[2:])
    targ_reshape = target.reshape(target.size(0) * target.size(1), *target.size()[2:])

    return F.cross_entropy(
            input=inp_reshape, target=targ_reshape, weight=weight, ignore_index=ignore_index, reduction=reduction)\
            .reshape(target.size(0), target.size(1)).reshape(target.size(0), target.size(1), *target.size()[2:]) \
            if reduction == 'none' else \
        F.cross_entropy(input=inp_reshape, target=targ_reshape, weight=weight, ignore_index=ignore_index,
                        reduction=reduction)