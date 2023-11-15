import torch
from torchvision.transforms import v2
import typing
import skimage
import numpy as np
import math
from torch.nn import ConstantPad2d


def radon_transform(image: torch.Tensor, theta=None):
    """Dimension of torch tensor should be 1 x height x width"""
    if theta is None:
        theta = 180

    # Transform image to Grayscale with single channel
    image = v2.Grayscale(1)(image)

    _, height, width = image.shape
    diagonal = math.sqrt(2) * max([height, width])
    pad = [int(math.ceil(diagonal - s)) for s in [height, width]]
    new_center = [(s + p) // 2 for s, p in zip([height, width], pad)]
    old_center = [s // 2 for s in [height, width]]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    pad = ConstantPad2d(pad_width[0] + pad_width[1], 0)
    image = pad(image)
    sinogram = torch.zeros([1, image.shape[1], theta])

    # Rotate by angle and sum in one dimension
    for i in range(theta):
        rotated_image = v2.RandomRotation((-i, -i), expand=False)(image)
        sum = torch.sum(rotated_image, 1)
        sinogram[0][:, i] = sum

    return sinogram


def batch_radon(t, theta=None, f, L):
    """
    t : List of points on radon projection plane
    theta : List [a, b] a start degree, b end degree
    f : function to sample from (support on [-1, 1]
    L : number of sample points on one Line
    """

    if theta is None:
        theta = np.arange(180)
    else:
        theta = np.arange(theta[0], theta[1])

    # line equation:
    # (x(z),y(z)) = ( z sin(theta) + s cos(theta), -z cos(theta) + s sin(theta)

    sample_grid =

    for i in theta:

        pass



