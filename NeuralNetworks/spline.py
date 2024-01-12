import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from skimage.transform import radon
from torch import nn

from DatasetClasses.funcInterval import FuncInterval
from DatasetClasses.integrals import *


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
        device = z.device
        if theta == 0 or theta == 90 or theta == 180:
            return 0

        # 0. Find line slope and normal of slope
        # 1. Find all control points that are close to the line
        #   - project all control points to normal of slope to find distance
        h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()

        threshold_x = 2 * h_x
        threshold_y = 2 * h_y

        # Transform degree into radians and compute rotation matrix
        phi = torch.tensor(theta * math.pi / 180, device=device)
        s = torch.sin(phi)
        c = torch.cos(phi)

        rot = torch.stack(
            [torch.stack([c, s]), torch.stack([-s, c])]
        ).T  # .T yields counterclockwise rotation

        # Calculate the normal to the line (same direction as projection plane)
        # case z = 0
        normal = torch.tensor([[0.0], [-z]], dtype=torch.float32, device=device)
        normal = torch.matmul(rot, normal)

        control_points = self.control_points.to(device)

        projections = (
            torch.matmul(control_points, normal)
            / (torch.linalg.norm(normal).unsqueeze(-1) ** 2)
        ) * normal.T

        distances = torch.abs(projections - normal.T)
        indices = torch.all(
            distances <= torch.tensor([threshold_x, threshold_y], device=device), dim=1
        ).nonzero()

        # 2. Find extrema of line, x- x+ s.t line is x- + t(x+ - x-) for t in [0,1]
        line_slope = torch.tensor([normal[1], -normal[0]], device=device)
        line_bias = normal.squeeze()

        if theta == 0 or theta == 90 or theta == 180:
            t_min = torch.tensor([-1, z], device=device)
            t_max = torch.tensor([1, z], device=device)
        else:
            t_x_pos = (1 - line_bias[0]) / line_slope[0]
            t_x_neg = (-1 - line_bias[0]) / line_slope[0]
            t_y_pos = (1 - line_bias[1]) / line_slope[1]
            t_y_neg = (-1 - line_bias[1]) / line_slope[1]

            line_value_x_pos = line_slope * t_x_pos + line_bias
            line_value_x_neg = line_slope * t_x_neg + line_bias
            line_value_y_pos = line_slope * t_y_pos + line_bias
            line_value_y_neg = line_slope * t_y_neg + line_bias

            t_min_max = []
            # maybe 1 plus h_x
            if abs(line_value_x_pos[1]) <= 1:
                t_min_max.append(line_value_x_pos)
            if abs(line_value_x_neg[1]) <= 1:
                t_min_max.append(line_value_x_neg)
            if abs(line_value_y_pos[0]) <= 1:
                t_min_max.append(line_value_y_pos)
            if abs(line_value_y_neg[0]) <= 1:
                t_min_max.append(line_value_y_neg)

            # Normalize line slope and bias
        if len(t_min_max) < 2:
            return 0
        t_min = t_min_max[0]
        t_max = t_min_max[1]

        if (t_max - t_min)[0] < 0:
            t_min, t_max = t_max, t_min
        line_slope = t_max - t_min
        line_bias = t_min

        # 3. Integrate all control points close to the line
        weights = self.weights.to(device)
        integral = 0
        for ind in indices:
            if weights[ind] == 0:
                integral = integral + 0
            else:
                integral = integral + (
                    self.integrate_control_point(
                        line_slope, line_bias, control_points[ind]
                    )
                    * weights[ind]
                    * torch.norm(line_slope)
                )
        return integral

    def integrate_control_point(self, slope, bias, control_point):
        h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()

        a = slope[0]
        b = bias[0]
        c = slope[1]
        d = bias[1]

        x = control_point[0][0]
        y = control_point[0][1]

        # Create and find the intervals for t in which the distance is 1 or 2
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

        a, b, c = self.create_intervals_from_bounds("x", x_bounds_1, x_bounds_2)
        d, e, f = self.create_intervals_from_bounds("y", y_bounds_1, y_bounds_2)

        x_list = [a, b, c]
        y_list = [d, e, f]

        x_list = self.split_intervals_at_crossing(x_crossing, x_list)
        y_list = self.split_intervals_at_crossing(y_crossing, y_list)

        x_list = self.assign_interval_signs(slope[0], x_crossing, x_list)
        y_list = self.assign_interval_signs(slope[1], y_crossing, y_list)

        combined_list = self.combine_x_y_intervals(x_list, y_list)

        # self, interval: FuncInterval, slope, bias, control_point
        integral = 0
        for interval in combined_list:
            integral = integral + self.apply_function_for_interval(
                interval, slope, bias, [x, y]
            )

        return integral

    def create_intervals_from_bounds(self, dim, bounds1, bounds2):
        """
        Usually creates 3 intervals. 2 in bounds2 1 inside bounds1
        :param bounds1: t bounds for distance < 1
        :param bounds2: t bounds for distance < 2
        :return: t intvervals for distance 1 and 2
        """
        if dim == "x":
            interval_a = FuncInterval(
                dim, torch.min(bounds2), torch.min(bounds1), x_dist=2
            )
            interval_b = FuncInterval(
                dim, torch.min(bounds1), torch.max(bounds1), x_dist=1
            )
            interval_c = FuncInterval(
                dim, torch.max(bounds1), torch.max(bounds2), x_dist=2
            )
        elif dim == "y":
            interval_a = FuncInterval(
                dim, torch.min(bounds2), torch.min(bounds1), y_dist=2
            )
            interval_b = FuncInterval(
                dim, torch.min(bounds1), torch.max(bounds1), y_dist=1
            )
            interval_c = FuncInterval(
                dim, torch.max(bounds1), torch.max(bounds2), y_dist=2
            )
        else:
            interval_a = FuncInterval(
                dim, torch.min(bounds2), torch.min(bounds1), x_dist=2, y_dist=2
            )
            interval_b = FuncInterval(
                dim, torch.min(bounds1), torch.max(bounds1), x_dist=1, y_dist=1
            )
            interval_c = FuncInterval(
                dim, torch.max(bounds1), torch.max(bounds2), x_dist=2, y_dist=2
            )

        return interval_a, interval_b, interval_c
        pass

    def split_intervals_at_crossing(self, crossing, intervals: list[FuncInterval]):
        new_intervals = []

        for interval in intervals:
            split_int_a, split_int_b = interval.split(crossing)
            if split_int_a == interval and split_int_b == interval:
                new_intervals.append(interval)
            else:
                new_intervals.append(split_int_a)
                new_intervals.append(split_int_b)

        return new_intervals

    def assign_interval_signs(self, slope, crossing, intervals: list[FuncInterval]):
        """
        Assigns "neg" or "pos" to interval. Requires intervals to be split at crossing
        :param slope:
        :param crossing:
        :param intervals:
        :return: intervals with sign
        """

        for interval in intervals:
            if interval.end <= crossing and slope > 0:
                if interval.dim == "x":
                    interval.x_sign = "neg"
                elif interval.dim == "y":
                    interval.y_sign = "neg"
            elif interval.end <= crossing and slope < 0:
                if interval.dim == "x":
                    interval.x_sign = "pos"
                elif interval.dim == "y":
                    interval.y_sign = "pos"
            elif interval.start >= crossing and slope > 0:
                if interval.dim == "x":
                    interval.x_sign = "pos"
                elif interval.dim == "y":
                    interval.y_sign = "pos"
            elif interval.start >= crossing and slope < 0:
                if interval.dim == "x":
                    interval.x_sign = "neg"
                elif interval.dim == "y":
                    interval.y_sign = "neg"

        return intervals

    def combine_x_y_intervals(
        self, x_intervals: list[FuncInterval], y_intervals: list[FuncInterval]
    ):
        stack = sorted(x_intervals + y_intervals, reverse=True)

        combined_list = []
        while len(stack) > 1:
            first = stack.pop()
            second = stack.pop()

            if first.dim == second.dim:
                if first.end <= second.start:
                    stack.append(second)
            elif first == second:
                # Intervals cover the same range and can be combined
                joint_interval = first.join(second)
                combined_list.append(joint_interval)
            elif first.end <= second.start:
                # No overlap, throw away first, put back second
                stack.append(second)
            elif first.start == second.start:
                # Intervals overlap at start, append shared length, put back rest
                if first.end > second.end:
                    # Make first interval always the smallest
                    first, second = second, first

                shortened_interval = second.modify_length(first.start, first.end)
                joint_interval = first.join(shortened_interval)
                combined_list.append(joint_interval)

                rest_interval = second.modify_length(first.end, second.end)
                stack.append(rest_interval)
            elif first.end < second.end:
                # Also first.start < second.start
                # Partial overlap, throw away first part, append shared part, put back rest
                first_shared_interval = first.modify_length(second.start, first.end)
                second_shared_interval = second.modify_length(second.start, first.end)
                joint_interval = first_shared_interval.join(second_shared_interval)
                combined_list.append(joint_interval)

                rest_interval = second.modify_length(first.end, second.end)
                stack.append(rest_interval)
            else:
                # first.end >= second.end
                # Full overlap, throw first non-shared part, append shared part, put back rest

                first_shared_interval = first.modify_length(second.start, second.end)
                joint_interval = first_shared_interval.join(second)
                combined_list.append(joint_interval)

                rest_interval = first.modify_length(second.end, first.end)
                stack.append(rest_interval)
        return combined_list

    def apply_function_for_interval(
        self, interval: FuncInterval, slope, bias, control_point
    ):
        x_sign = interval.x_sign
        y_sign = interval.x_sign
        x_dist = interval.x_dist
        y_dist = interval.y_dist

        a = slope[0]
        b = bias[0]
        x = control_point[0]
        c = slope[1]
        d = bias[1]
        y = control_point[1]

        if x_sign == "neg":
            if x_dist == 2:
                if y_sign == "neg":
                    if y_dist == 2:
                        return int01(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int02(a, b, c, d, x, y, interval.start, interval.end)
                else:
                    if y_sign == "neg":
                        return int03(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int04(a, b, c, d, x, y, interval.start, interval.end)
            else:
                if y_sign == "neg":
                    if y_dist == 2:
                        return int05(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int06(a, b, c, d, x, y, interval.start, interval.end)
                else:
                    if y_sign == "neg":
                        return int07(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int08(a, b, c, d, x, y, interval.start, interval.end)
        else:
            if x_dist == 2:
                if y_sign == "neg":
                    if y_dist == 2:
                        return int09(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int10(a, b, c, d, x, y, interval.start, interval.end)
                else:
                    if y_sign == "neg":
                        return int11(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int12(a, b, c, d, x, y, interval.start, interval.end)
            else:
                if y_sign == "neg":
                    if y_dist == 2:
                        return int13(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int14(a, b, c, d, x, y, interval.start, interval.end)
                else:
                    if y_sign == "neg":
                        return int15(a, b, c, d, x, y, interval.start, interval.end)
                    else:
                        return int16(a, b, c, d, x, y, interval.start, interval.end)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = "cpu"
print(f"Using {device} device")
if __name__ == "__main__":
    N = 32
    model = SplineNetwork(N)
    model.load_state_dict(torch.load("../spline_image.pt", map_location=device))
    model.eval()

    interval = torch.linspace(-1, 1, steps=math.ceil(N * 1.0))
    gridx, gridy = torch.meshgrid(interval, interval, indexing="xy")
    model_input = torch.stack((gridx, gridy), dim=2)
    model_output, _ = model(model_input)
    radon_transform = radon(
        model_output.view(N, N).cpu().detach().numpy(),
        theta=np.linspace(0.0, 180.0, N) + 90,
        circle=False,
    )
    plt.imshow(model_output.view(N, N).cpu().detach().numpy())
    plt.show()
    plt.imshow(radon_transform)
    plt.show()

    t = torch.linspace(
        -math.sqrt(2), math.sqrt(2), steps=math.ceil(N * math.sqrt(2)), device=device
    )
    theta = torch.linspace(0.01, 179.99, steps=N, device=device)

    sinogram = torch.zeros((len(t), len(theta)))
    for i, x in enumerate(t):
        print(i)
        for j, phi in enumerate(theta):
            print(phi)
            sinogram[i, j] = model.integrate_line(x, phi)

    torch.save(sinogram, "integral_image.pt")
    plt.imshow(sinogram.cpu().detach().numpy())
    plt.show()
