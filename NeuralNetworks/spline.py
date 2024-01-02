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
        h = (self.control_points[0] - self.control_points[1]).norm()
        threshold = 2 * h
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

        distances = torch.linalg.norm(projections - normal.T, dim=1)
        indices = (distances <= threshold).nonzero()

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
        h = torch.linalg.norm(self.control_points[0, :] - self.control_points[1, :])

        a = slope[0] ** 2 + slope[1] ** 2
        b = -2 * (
            slope[0] * (-bias[0] + control_point[0, 0])
            + slope[1] * (-bias[1] + control_point[0, 1])
        )
        c = (-bias[0] + control_point[0, 0]) ** 2 + (
            -bias[1] + control_point[0, 1]
        ) ** 2

        bounds1 = self.solve_quadratic_equation(a, b, c - h**2)
        bounds2 = self.solve_quadratic_equation(a, b, c - 4 * h**2)
        if (
            bounds1[0] == float("nan")
            or bounds1[1] == float("nan")
            or bounds2[0] == float("nan")
            or bounds2[1] == float("nan")
        ):
            print("bound nan")
        if bounds1[1] > bounds2[1] or bounds1[0] < bounds2[0]:
            print("Something is wrong with the integral bounds!")

        if bounds1[0] < -1:
            bounds1[0] = -1
        if bounds2[0] < -1:
            bounds2[0] = -1
        if bounds1[1] > 1:
            bounds1[1] = 1
        if bounds2[1] > 1:
            bounds2[1] = 1

        eps = 0.01
        integral = (
            self.integrate_cubic_bound2(a, b, c, h, bounds2[1])
            - self.integrate_cubic_bound2(a, b, c, h, bounds1[1])
            + self.integrate_cubic_bound1(a, b, c, h, bounds1[1])
            - self.integrate_cubic_bound1(a, b, c, h, bounds1[0])
            + self.integrate_cubic_bound2(a, b, c, h, bounds1[0])
            - self.integrate_cubic_bound2(a, b, c, h, bounds2[0])
        )

        return integral

    def solve_quadratic_equation(self, a, b, c):
        D = b**2 - 4 * a * c

        if b == 0 and a != 0 and c <= 0:
            return torch.tensor([-math.sqrt(-c / a), math.sqrt(-c / a)])
        if a == 0 and b != 0:
            return torch.tensor([-c / b, -c / b])
        elif D < 0:
            return torch.tensor([float("nan"), float("nan")])
        elif a == 0 and b == 0:
            return torch.tensor([float("nan"), float("nan")])
        root1 = (-b - math.sqrt(D)) / (2 * a)
        root2 = (-b + math.sqrt(D)) / (2 * a)

        return torch.tensor([root1, root2])

    def integrate_cubic_bound1(self, a, b, c, h, t):
        if 4 * a * c - b**2 <= 0:
            out = (
                (3 * (math.sqrt(a) * t + math.sqrt(c)) ** 4)
                / (8 * math.sqrt(a) * h**3)
                - (5 * a * t**3) / (6 * h**2)
                - (5 * math.sqrt(a) * math.sqrt(c) * t**2) / (2 * h**2)
                + (1 - (5 * c) / (2 * h**2)) * t
            )
        elif b == 0 and c == 0:
            out = (
                (3 * a ** (3 / 2) * t**3 * abs(t)) / (8 * h**3)
                - (5 * a * t**3) / (6 * h**2)
                + t
            )
        else:
            out = (
                27
                * math.sqrt(a)
                * (4 * a * c - b**2) ** 2
                * math.asinh((2 * a * t + b) / math.sqrt(4 * a * c - b**2))
                - 640 * a**4 * h * t**3
                - 960 * a**3 * b * h * t**2
                + 18
                * a
                * (2 * a * t + b)
                * math.sqrt(t * (a * t + b) + c)
                * (4 * a * (2 * t * (a * t + b) + 5 * c) - 3 * b**2)
                + 384 * a**3 * h * (2 * h**2 - 5 * c) * t
            ) / (768 * a**3 * h**3)

        return out

    def integrate_cubic_bound2(self, a, b, c, h, t):
        if 4 * a * c - b**2 <= 0:
            out = (
                t
                * (
                    math.sqrt(a) * ((48 * h**2 - 18 * c) * t - 3 * a * t**3)
                    + math.sqrt(c)
                    * (
                        -12 * a * t**2
                        + 60 * math.sqrt(a) * h * t
                        + 96 * h**2
                        - 12 * c
                    )
                    + 20 * a * h * t**2
                    + 48 * h**3
                    + 60 * c * h
                )
            ) / (24 * h**3)
        elif b == 0 and c == 0:
            out = (
                -(math.sqrt(a) * (3 * a * t**3 + 48 * h**2 * t) * abs(t))
                / (24 * h**3)
                + (5 * a * t**3) / (6 * h**2)
                + 2 * t
            )
        else:
            out = (
                -(
                    (4 * a * c - b**2)
                    * (4 * a * (32 * h**2 + 3 * c) - 3 * b**2)
                    * math.asinh((2 * a * t + b) / math.sqrt(4 * a * c - b**2))
                )
                / (256 * a ** (5 / 2) * h**3)
                + (5 * a * t**3) / (6 * h**2)
                + (5 * b * t**2) / (4 * h**2)
                - (
                    (2 * a * t + b)
                    * math.sqrt(t * (a * t + b) + c)
                    * (4 * a * (2 * t * (a * t + b) + 32 * h**2 + 5 * c) - 3 * b**2)
                )
                / (128 * a**2 * h**3)
                + ((4 * h**2 + 5 * c) * t) / (2 * h**2)
            )

        return out


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = "cpu"
print(f"Using {device} device")
TEST_INTEGRATE = False
if TEST_INTEGRATE:
    n = 32
    model = SplineNetwork(n)

    t = torch.linspace(-0.5, -0.25, steps=30)
    theta = torch.arange(0, 5, step=1)

    sinogram = torch.zeros((len(t), len(theta)))
    for i, x in enumerate(t):
        print(i)
        for phi in theta:
            sinogram[i, phi] = model.integrate_line(x, phi)

    plt.imshow(sinogram.cpu().detach().numpy())
    plt.show()
