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


class Spline(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.control_points = get_mgrid(N)
        self.weights