import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from skimage.transform import radon
from torch import nn

from DatasetClasses.funcInterval import FuncInterval
from DatasetClasses.explicit_integrals import *


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
        self.weights = 1/9*torch.ones(N, N, requires_grad=True)
        self.weights = self.weights.reshape(-1, 1)
        self.weights = nn.Parameter(self.weights)

        self.control_points = get_mgrid(sidelen=N, dim=2).view(-1, 2)

        if circle:
            self.control_points = self.control_points / math.sqrt(2)
        self.control_points.requires_grad = False


        self.h_x = (self.control_points[0][0] - self.control_points[1][0]).norm()
        self.h_y = (self.control_points[0][1] - self.control_points[self.N][1]).norm()


    def forward(self, x):
        """
        :param x: input with shape (batch, 2)
        :return: cubic interpolation function: sum w_i conv(||x-X||/h), shape (batch)
        """
        K = 16

        device = x.device

        #self.weights = torch.nn.Parameter(self.weights.to(device))
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
        h_x = self.h_x
        h_y = self.h_y
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

        h_x = self.h_x
        h_y = self.h_y

        threshold = math.sqrt((2*h_x)**2+(2*h_y)**2)

        # Transform degree into radians and compute rotation matrix
        phi = torch.tensor(theta * math.pi / 180, device=device)
        s = torch.sin(phi)
        c = torch.cos(phi)

        rot = torch.stack(
            [torch.stack([c, s]), torch.stack([-s, c])]
        ).T  # .T yields counterclockwise rotation

        # Calculate the normal to the line (same direction as projection plane)
        # case z = 0
        normal = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=device)
        normal_orth = torch.tensor([[1.0], [0.0]], dtype=torch.float32, device=device)


        control_points = self.control_points.clone().detach()
        control_points[:,1] = - self.control_points.clone().detach()[:,1]

        control_points_rot = torch.matmul(control_points, rot)

        projections = torch.matmul(control_points_rot, normal_orth)

        distances = torch.abs(projections - z)
        indices = (distances.squeeze(1) < threshold).nonzero()

        line_slope = torch.matmul(rot, normal).squeeze()
        line_bias = torch.tensor([[z], [0]], device=device)
        line_bias = torch.matmul(rot, line_bias).squeeze()


        integral = 0
        for k in indices:
            if self.weights[k] != 0:
                control_point = control_points[k]
                x = control_point[0][0]
                y = control_point[0][1]
                m1 = line_slope[0]
                m2 = line_slope[1]
                b1 = line_bias[0]
                b2 = line_bias[1]

                t = ((y-b2)*m2 + (x-b1)*m1)/(m1**2 + m2**2)
                line_bias = line_slope*t + line_bias
                tmp = self.integrate_control_point(line_slope, line_bias, control_point) * torch.norm(line_slope) * self.weights[k]
                integral = integral + tmp

        return integral

    def find_interval(self,alpha, n, h):
        A_tilde = n ** 2
        B_tilde = 2 * n * alpha
        C_tilde1 = alpha ** 2 - h ** 2
        delta = B_tilde ** 2 - 4 * A_tilde * C_tilde1
        if torch.abs(delta) < 1e-4:
            delta = delta * 0.
        t_p = (-B_tilde + torch.sqrt(delta)) / (2 * A_tilde)
        t_m = (-B_tilde - torch.sqrt(delta)) / (2 * A_tilde)
        return t_m, t_p

    def integrate_control_point(self, slope, bias, control_point):
        h_x = self.h_x
        h_y = self.h_y

        x = control_point[0][0]
        y = control_point[0][1]

        # Create and find the intervals for t in which the distance is 1 or 2
        x_bounds_1, y_bounds_1, x_bounds_2, y_bounds_2 = (
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )

        x_bounds_1[0] = ((-1*h_x + x - bias[0]) / slope[0])
        x_bounds_1[1] = ((1*h_x + x - bias[0]) / slope[0])
        x_bounds_2[0] = ((-2 * h_x + x - bias[0]) / slope[0])
        x_bounds_2[1] = ((2 * h_x + x - bias[0]) / slope[0])

        x_crossing = (x - bias[0]) / slope[0]

        y_bounds_1[0] = ((-1*h_y + y - bias[1]) / slope[1])
        y_bounds_1[1] = ((1*h_y + y - bias[1]) / slope[1])
        y_bounds_2[0] = ((-2 * h_y + y - bias[1]) / slope[1])
        y_bounds_2[1] = ((2 * h_y + y - bias[1]) / slope[1])

        y_crossing = (y - bias[1]) / slope[1]

        # It can happen that the bounds are not in start, end order, therefore sort
        x_bounds_1, _ = torch.sort(x_bounds_1)
        x_bounds_2, _ = torch.sort(x_bounds_2)
        y_bounds_1, _ = torch.sort(y_bounds_1)
        y_bounds_2, _ = torch.sort(y_bounds_2)

        # If statements to handle special case where the line is perpendicular to the x- or y-axis.
        # If the line is perpendicular to the x-axis it will have slope[0] = 0 and therefore no change in x.
        if slope[0] != 0:
            x_list = self.create_intervals_from_bounds("x", x_bounds_1, x_bounds_2)
            x_list = self.split_intervals_at_crossing(x_crossing, x_list)
            x_list = self.assign_interval_signs(slope[0], x_crossing, x_list)
            combined_list = x_list

        # If the line is perpendicular to the y-axis it will have slope[1] = 0 and therefore no change in y.
        if slope[1] != 0:
            y_list = self.create_intervals_from_bounds("y", y_bounds_1, y_bounds_2)
            y_list = self.split_intervals_at_crossing(y_crossing, y_list)
            y_list = self.assign_interval_signs(slope[1], y_crossing, y_list)
            combined_list = y_list

        # If the line is not perpendicular to the x- or y-axis (which will be the case for most lines)
        # we will have change in both x and y and need to merge the two lists of intervals.
        if slope[0] != 0 and slope[1] != 0:
            combined_list = self.combine_x_y_intervals(x_list, y_list)

        # Integrate over all the intervals in combined_list separately.
        integral = 0
        for interval in combined_list:
            integral = integral + (self.apply_function_for_interval(
                interval, slope, bias, [x, y]
            ))

        return integral


    def create_intervals_from_bounds(self, dim, bounds1, bounds2):
        """
        Usually creates 3 intervals. 2 in bounds2 1 inside bounds1
        :param bounds1: t bounds for distance < 1
        :param bounds2: t bounds for distance < 2
        :return: t intervals for distance 1 and 2
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

        return_list = []
        if interval_a.start != interval_a.end:
            return_list.append(interval_a)
        if interval_b.start != interval_b.end:
            return_list.append(interval_b)
        if interval_c.start != interval_c.end:
            return_list.append(interval_c)
        return return_list

    def split_intervals_at_crossing(self, crossing, intervals: list[FuncInterval]):
        """
        When an interval [a,b] in the given list of intervals contains the value 'crossing', it will be split
        and two new intervals [a,crossing], [crossing,b] will be created.
        :param crossing: Value at which intervals will be split
        :param intervals: List of intervals
        :return: List of intervals split at crossing
        """
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
        Assigns "neg" or "pos" to interval. Requires intervals to be split at 0 crossing
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
        """
        Combine two sets of intervals to one. For example:
        A = {[0,2],[2,4]}, B = {[1,3]} will become C = {[0,1],[1,2],[2,3],[3,4].
        The intervals that are used here are intervals of t for which a given line has
        either distance less than 1 or less than 2 to a given control point in a given coordinate (x or y coordinate).
        This information will be kept in the combined interval list.
        :param x_intervals: t intervals for in which certain distances in x are reached
        :param y_intervals: t intervals for in which certain distances in y are reached
        :return
        """
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
        """
        Given a interval select the correct function to integrate it
        :param interval: Interval to integrate over
        :param slope: Slope of the line that is integrated along
        :param bias: Bias term of the line that is integrated along
        :param control_point: Spline control_point that is integrated
        :return: Exact integral value
        """

        x_sign = interval.x_sign
        y_sign = interval.y_sign
        x_dist = interval.x_dist
        y_dist = interval.y_dist

        h = self.h_y
        a = slope[0]
        b = bias[0]
        x = control_point[0]
        c = slope[1]
        d = bias[1]
        y = control_point[1]

        start = interval.start
        end = interval.end

        if interval.dim == "x":
            result = integrate_1d(x_sign, x_dist, a, b, x, h, start, end) * self.cubic_conv((d-y)/self.h_y)
        elif interval.dim == "y":
            result = integrate_1d(y_sign, y_dist, c, d, y, h, start, end) * self.cubic_conv((b-x)/self.h_x)
        else:
            # dim == "xy"
            result = integrate_exact(x_sign, x_dist, y_sign, y_dist, a, b, c, d, x, y, h, start, end)
        return result


