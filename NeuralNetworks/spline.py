import math
import torch
from matplotlib import pyplot as plt
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
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="xy"), dim=-1)
    return mgrid


class SplineNetwork(nn.Module):
    def __init__(self, N, circle=False):
        """
        :param N: size of sidelength. Grid size will be N*N
        """
        super().__init__()

        self.N = N
        self.weights = torch.zeros(N, N, requires_grad=True)
        # self.weights = LodopabImage(N, pad=False).image
        # self.weights = self.weights.cpu()
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
        K = 16

        device = x.device

        self.weights = self.weights.to(device)
        self.control_points = self.control_points.to(device)

        old_shape = x.shape
        is3d = False
        if x.dim() == 3:
            is3d = True
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
        h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()
        pairwise_norm = x.unsqueeze(2) - neighbors.permute(0, 2, 1)

        input_x = pairwise_norm[:, 0, :] / h_x  # Shape: (batch, K)
        input_y = pairwise_norm[:, 1, :] / h_y  # Shape: (batch, K)

        conv_out_x = self.cubic_conv(input_x)  # Shape: (batch, K)
        conv_out_y = self.cubic_conv(input_y)  # Shape: (batch, K)

        # Perform row-wise dot product between weight and conv_out vectors
        conv_out_prod = conv_out_x * conv_out_y
        weights = self.weights[indices].squeeze()

        output = torch.sum(weights * conv_out_prod, dim=1)

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

    def integrate_line(self, z, theta):
        """
        Integrate the spline network along a line
        :param z: Position on the detection plane
        :param theta: Angle of rotation in degrees
        :return: Integral along line specified by z and theta
        """

        # 0. Find line slope and normal of slope
        # 1. Find all control points that are close to the line
        #   - project all control points to normal of slope to find distance
        h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()

        threshold_x = 2 * h_x
        threshold_y = 2 * h_y

        # Transform degree into radians and compute rotation matrix
        phi = torch.tensor(theta * math.pi / 180)
        s = torch.sin(phi)
        c = torch.cos(phi)

        rot = torch.stack(
            [torch.stack([c, s]), torch.stack([-s, c])]
        ).T  # .T yields counterclockwise rotation

        # Calculate the normal to the line (same direction as projection plane)
        normal = torch.tensor([[0.0], [z]], dtype=torch.float32)
        normal = torch.matmul(rot, normal)

        control_points = self.control_points

        projections = (
            torch.matmul(control_points, normal)
            / (torch.linalg.norm(normal).unsqueeze(-1) ** 2)
        ) * normal.T

        distances = torch.abs(projections - normal.T)
        indices = torch.all(
            distances <= torch.tensor([threshold_x, threshold_y]), dim=1
        ).nonzero()

        # 2. Find extrema of line, x- x+ s.t line is x- + t(x+ - x-) for t in [0,1]
        line_slope = torch.tensor([normal[1], -normal[0]])
        line_bias = normal.squeeze()

        if theta == 0 or theta == 90 or theta == 180:
            t_min = torch.tensor([-1, z])
            t_max = torch.tensor([1, z])
        elif 0 < theta < 90:
            tx = (-1 - line_bias[0]) / line_slope[0]
            t_min = line_slope * tx + line_bias
            ty = (1 - line_bias[0]) / line_slope[1]
            t_max = line_slope * ty + line_bias
        elif 90 < theta < 180:
            tx = (1 - line_bias[0]) / line_slope[0]
            t_min = line_slope * tx + line_bias
            ty = (-1 - line_bias[1]) / line_slope[1]
            t_max = line_slope * ty + line_bias

        # Normalize line slope and bias
        line_slope = t_max - t_min
        line_bias = t_min

        # 3. Integrate all control points close to the line
        integrals = torch.zeros(indices.shape)
        for i, ind in enumerate(indices):
            integrals[i] = (
                self.integrate_control_point(line_slope, line_bias, control_points[ind])
                * self.weights[ind]
            )

        integrals = torch.nan_to_num(integrals, nan=0.0)
        sum = integrals.sum()
        return sum

    def integrate_control_point(self, slope, bias, control_point):
        h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()

        a = slope[0]
        b = bias[0]
        c = slope[1]
        d = bias[1]

        x = control_point[0][0]
        y = control_point[0][1]

        x_bounds_1, y_bounds_1, x_bounds_2, y_bounds_2 = (
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )

        x_bounds_1[0] = torch.nn.functional.relu((-h_x + x - bias[0]) / slope[0])
        x_bounds_1[1] = torch.nn.functional.relu((h_x + x - bias[0]) / slope[0])
        x_bounds_2[0] = torch.nn.functional.relu((-2 * h_x + x - bias[0]) / slope[0])
        x_bounds_2[1] = torch.nn.functional.relu((2 * h_x + x - bias[0]) / slope[0])

        x_crossing = (x - bias[0]) / slope[0]

        y_bounds_1[0] = torch.nn.functional.relu((-h_y + y - bias[1]) / slope[1])
        y_bounds_1[1] = torch.nn.functional.relu((h_y + y - bias[1]) / slope[1])
        y_bounds_2[0] = torch.nn.functional.relu((-2 * h_y + y - bias[1]) / slope[1])
        y_bounds_2[1] = torch.nn.functional.relu((2 * h_y + y - bias[1]) / slope[1])

        y_crossing = (y - bias[1]) / slope[1]

        x_bounds_1[x_bounds_1 > 1] = 1
        x_bounds_2[x_bounds_2 > 1] = 1
        y_bounds_1[y_bounds_1 > 1] = 1
        y_bounds_2[y_bounds_2 > 1] = 1

        lower_max_2, ind_lower_max_2 = torch.max(
            torch.tensor([x_bounds_2[0], y_bounds_2[0]]), dim=0
        )
        lower_min_1 = torch.min(torch.tensor([x_bounds_1[0], y_bounds_1[0]]))
        lower_max_1 = torch.max(torch.tensor([x_bounds_1[0], y_bounds_1[0]]))
        upper_min_1 = torch.min(torch.tensor([x_bounds_1[1], y_bounds_1[1]]))
        upper_max_1 = torch.max(torch.tensor([x_bounds_1[1], y_bounds_1[1]]))
        upper_min_2 = torch.min(torch.tensor([x_bounds_2[1], y_bounds_2[1]]))
        upper_max_2 = torch.max(torch.tensor([x_bounds_2[1], y_bounds_2[1]]))
        integral = ()

        return integral


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = "cpu"
print(f"Using {device} device")
TEST_INTEGRATE = True
if TEST_INTEGRATE:
    n = 32
    model = SplineNetwork(n)

    t = torch.linspace(-0.5, -0.25, steps=30)
    theta = torch.arange(179, 180, step=1)

    sinogram = torch.zeros((len(t), len(theta)))
    for i, x in enumerate(t):
        print(i)
        for phi in theta:
            sinogram[i, phi] = model.integrate_line(x, phi)

    plt.imshow(sinogram.cpu().detach().numpy())
    plt.show()
