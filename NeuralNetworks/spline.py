import math
import torch
from pykeops.torch import LazyTensor
from torch import nn


def get_mgrid(sidelen, dim=2):
    """
    Generates a flattened grid of (x,y) coordinates in a range of -1 to 1, Shape: (sidelen, dim)
    :param sidelen: sidelength of grid. Grid will have sidelen ** 2 points
    :param dim: dimension of coordiantes
    :return: Grid with shape: (sidelen, dim)
    """
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid


class SplineNetwork(nn.Module):
    def __init__(self, N, circle=False):
        """
        :param N: size of sidelength. Grid size will be N*N
        """
        super().__init__()

        self.N = N
        self.weights = torch.zeros(N, N, requires_grad=True)
        self.weights = self.weights.reshape(-1, 1)
        self.weights = nn.Parameter(self.weights)
        # self.weights.data.uniform_(-1 / N, 1 / N)

        self.control_points = get_mgrid(sidelen=N, dim=2).view(-1, 2)

        if circle:
            self.control_points = self.control_points / math.sqrt(2)
        self.control_points.requires_grad = False

    def forward(self, x):
        """
        :param x: input with shape (batch, 2)
        :return: cubic interpolation function: sum w_i conv(||x-X||/h), shape (batch)
        """
        K = 9

        device = x.device

        self.weights = self.weights.to(device)
        self.control_points = self.control_points.to(device)

        old_shape = x.shape
        is3d = False
        if x.dim() == 3:
            is4d = True
            x = x.reshape(-1, 2)

        X_i = LazyTensor(x, axis=0)
        X_j = LazyTensor(self.control_points, axis=1)

        # Calculate symbolic (euclidean) distance matrix
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        # Find indices of K nearest neighbors
        indices = D_ij.argKmin(K, dim=1)  # Shape: (batch, K)
        indices = indices.to(device)

        # access the K nearest neighbors
        neighbors = self.control_points[indices]  # Shape: (batch, K, 2)

        # Prepare input for convolutional kernel function
        h = (x[0] - x[1]).norm()
        pairwise_norm = torch.linalg.norm(
            x.unsqueeze(2) - neighbors.permute(0, 2, 1), dim=1
        )
        input = pairwise_norm / h  # Shape: (batch, K)

        conv_out = self.cubic_conv(input)  # Shape: (batch, K)

        # Perform row-wise dot product between weight and conv_out vectors
        output = torch.sum(
            self.weights[indices] * conv_out.unsqueeze(-1), dim=1
        )  # Shape: (batch)
        if is3d:
            output = output.reshape(old_shape[0], old_shape[1], 1)

        return output, x

    def cubic_conv(self, s):
        """
        Calculate convolutional kernel.

        :param s: ||x-X||/h, shape: (batch, K)
        :return: kernel operation on s, shape: (batch, K)
        """
        device = s.device
        result = torch.zeros(s.shape, device=device)

        cond1 = (0 <= torch.abs(s)) & (torch.abs(s) < 1)
        cond2 = (1 < torch.abs(s)) & (torch.abs(s) < 2)
        cond3 = 2 < torch.abs(s)

        result[cond1] = (
            3 / 2 * torch.abs(s[cond1]) ** 3 - 5 / 2 * torch.abs(s[cond1]) ** 2 + 1
        )
        result[cond2] = (
            -1 / 2 * torch.abs(s[cond2]) ** 3
            + 5 / 2 * torch.abs(s[cond2]) ** 2
            - 4 * torch.abs(s[cond2])
            + 2
        )
        # result[cond3] = 0

        return result


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

n = 1000
# model = SplineNetwork(n).to(device)

X = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
