import h5py
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from RadonTransform.radon_transform import batch_radon_siren


class LodopabImage(Dataset):
    """Loads a single image from the LoDoPaB-CT dataset and makes pixel batching possible"""

    def __init__(self, resolution, set="ground_truth_train", pos1="000", pos2=0):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.device = "cpu"
        self.resolution = resolution
        self.image_path = f"../dataset/{set}/{set}_{pos1}.hdf5"
        self.image = self.read_hdf5(self.image_path)[pos2, :, :]
        # Shape: [362, 362]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(resolution, antialias=True),
                # transforms.Pad(
                # math.ceil((math.sqrt(2) * resolution - resolution)) // 2
                # ),
            ]
        )

        self.image = self.transform(self.image)  # Shape: [1, res, res]
        self.image = self.image.squeeze().detach().numpy()  # Shape: [res, res]

        diagonal = math.sqrt(2) * max(self.image.shape)
        pad = [int(math.ceil(diagonal - s)) for s in self.image.shape]
        new_center = [(s + p) // 2 for s, p in zip(self.image.shape, pad)]
        old_center = [s // 2 for s in self.image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(
            self.image, pad_width, mode="constant", constant_values=0
        )  # Shape: [512, 512]
        self.image = torch.from_numpy(padded_image).unsqueeze(0)  # Shape: [1, 512, 512]
        print(pad_width)

        self.pixels = self.image.permute(1, 2, 0).view(-1, 1)

        self.padded_resolution = self.image.shape[1]
        self.coords = self.get_mgrid(self.padded_resolution, dim=2)

        self.radon = self.get_radon_transform()

        self.image = self.image.to(self.device)
        self.radon = self.radon.to(self.device)

    def __len__(self):
        return self.image.shape[1] * self.image.shape[2]

    def __getitem__(self, idx):
        return self.coords[idx, :], self.pixels[idx]

    def read_hdf5(self, path):
        file = h5py.File(path, "r")
        with file as f:
            group_key = list(f.keys())[0]
            data = f[group_key][()]
        file.close()
        return data

    def get_mgrid(self, sidelen, dim=2):
        """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int"""
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        mgrid.to(self.device)
        return mgrid

    def get_2d_np(self):
        return self.image.detach().numpy()[0, :, :]

    def sample_image(self, grid, image=None):
        """
        Sample image with coordinates in [-1,1]^2 using bilinear interpolation
        :param grid: meshgrid with shape (N, H_out, W_out, 2)
        :param image: shape (N, C, H_in, W_in)
        :return: pixel value at coordinates shape (N, Height, Width, C) and input grid
        """
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if image is None:
            image = self.image  # Shape: [1, res, res]
            image = image.unsqueeze(0)  # Shape: [1, 1, res, res]
            image = image.expand(grid.shape[0], -1, -1, -1)  # Shape: [N, 1, res, res]
            image = image.to(self.device)

        sampled_image = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        sampled_image = sampled_image.permute(0, 2, 3, 1)
        sampled_image = sampled_image.to(self.device)

        return sampled_image, grid

    def sample_radon(self, grid, image=None):
        """
        Sample image with coordinates in [-1,1]^2 using bilinear interpolation
        :param grid: meshgrid with shape (N, H_out, W_out, 2)
        :param image: shape (N, C, H_in, W_in)
        :return: pixel value at coordinates shape (N, Height, Width, C) and input grid
        """
        if image is None:
            image = self.radon  # Shape: [padded_res, 180]
            image = image.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, padded_es, 180]
            image = image.expand(grid.shape[0], -1, -1, -1)  # Shape: [N, 1, res, res]
            image = image.to(self.device)

        sampled_image = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        sampled_image = sampled_image.permute(0, 2, 3, 1)
        sampled_image = sampled_image.to(self.device)
        return sampled_image, grid

    def get_radon_transform(self):
        print(self.padded_resolution)
        z = torch.linspace(
            -math.sqrt(1),
            math.sqrt(1),
            steps=self.padded_resolution,
            device=self.device,
        )
        f = self.sample_image
        L = self.padded_resolution
        theta = torch.arange(0, 180, step=1, device=self.device) + 90

        radon_transform = batch_radon_siren(z, f, L, theta, self.device)

        return radon_transform

    def get_padded_resolution(self):
        return self.padded_resolution
