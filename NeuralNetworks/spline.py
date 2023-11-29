import math

import torch
from torch import nn
import numpy as np


def get_mgrid(self, sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid


class SplineLayer(nn.Module):
    def __init(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.parameter(weights)

        # initialize weights
        nn.init.kaiming_uniform(self.weights, a=math.sqrt(5))

    def forward(self, x):
        pass

class SplineNetwork(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.control_points = get_mgrid(N)
        self.weights