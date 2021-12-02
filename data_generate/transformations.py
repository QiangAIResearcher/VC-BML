import torch
import torchvision.transforms.functional as TF

class ToDevice(object):
    def __init__(self, device=None):
        self.device = device

    def __call__(self, tensor_pic):
        return tensor_pic.to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + '(device={0})'.format(self.device)

class NormaliseMinMax(object):
    def __call__(self, tensor):
        self.min = tensor.min()
        self.max = tensor.max()
        return (tensor - self.min) / (self.max - self.min)

    def __repr__(self):
        return self.__class__.__name__ + '(min={0}, max={1})'.format(self.min, self.max)

class ChangeBlackToColour(object):
    def __init__(self, colour):
        self.colour = colour

    def __call__(self, tensor):
        mask = tensor == 1.
        tensor_expand = tensor.repeat_interleave(dim=0, repeats=3)
        return tensor_expand.masked_scatter(mask, torch.tensor(self.colour).unsqueeze(-1).repeat_interleave(dim=1, repeats=mask.sum()))

    def __repr__(self):
        return self.__class__.__name__ + '(rgb_colour={})'.format(self.colour)

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image): # takes PIL image
        return TF.rotate(img=image, angle=self.angle)