from torch.utils.data import Dataset
from torch import linspace

class CoordSet(Dataset):
    def __init__(self, L, circle=True):
        """ L : amount of points on projection plane"""
        if circle:
            self.coords = linspace(-1,1, L)
        else:
            self.coords = linspace(-1.414, 1.414, L)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx]

class AngleSet(Dataset):
    def __init__(self, L):
        """ L : amount of angles between 0 and pi"""
        self.pi = 3.1459
        self.angles = linspace(0, self.pi, L)

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        return self.angles[idx]
