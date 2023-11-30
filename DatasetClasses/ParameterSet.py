import math
from torch import arange
from torch import linspace
from torch.utils.data import Dataset


class CoordSet(Dataset):
    def __init__(self, L, circle=True):
        """L : amount of points on projection plane"""
        if circle:
            self.coords = linspace(-1, 1, steps=L)
        else:
            self.coords = linspace(-math.sqrt(2), math.sqrt(2), steps=L)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], idx


class AngleSet(Dataset):
    def __init__(self, L, rad=True):
        """L : amount of angles between 0 and pi"""
        self.pi = math.pi
        if rad:
            self.angles = linspace(0, self.pi, steps=L)
        else:
            self.angles = arange(0, L, 1)

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        return self.angles[idx], idx
