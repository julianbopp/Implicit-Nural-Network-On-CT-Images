import math
import torch
from torch.nn import ConstantPad2d
from torchvision.transforms import v2


def radon_transform(image: torch.Tensor, theta=None, circle=True, SIREN=False):
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

    if not circle:
        image = pad(image)
    sinogram = torch.zeros([1, image.shape[1], theta])

    # Rotate by angle and sum in one dimension
    for i in range(theta):
        rotated_image = v2.RandomRotation((-i, -i), expand=False)(image)
        sum = torch.sum(rotated_image, 1)
        sinogram[0][:, i] = sum

    return sinogram


def radon_transform_alt(image: torch.Tensor, theta: int):
    # Transform image to Grayscale with single channel
    image = v2.Grayscale(1)(image)

    # Rotate image once by 45 to create padding with 0's
    image = v2.RandomRotation((45, 45), expand=True)(image)
    image = v2.RandomRotation((-45, -45), expand=False)(image)
    print(image.shape)

    _, height, width = image.shape
    sinogram = torch.zeros([1, height, theta])

    # Rotate by angle and sum in one dimension
    for i in range(theta):
        rotated_image = v2.RandomRotation((i, i), expand=False)(image)
        sum = torch.sum(rotated_image, 1)
        sinogram[0][:, i] = sum

    return sinogram


def batch_radon_siren(z, f, L, theta=None, device=None, circle=True, SIREN=True):
    """
    z : tensor of points on radon projection plane
    theta : List [a, b] a start degree, b end degree
    f : function to sample from (support on [-1, 1])
    L : number of sample points on one Line
    """

    if circle:
        t = torch.linspace(-math.sqrt(1), math.sqrt(1), steps=L, device=device)
    else:
        t = torch.linspace(-math.sqrt(2), math.sqrt(2), steps=L, device=device)
    output = torch.zeros(len(z), len(theta), device=device)

    mgrid = torch.stack(torch.meshgrid(t, z, indexing="xy"), dim=2)
    mgrid = mgrid.unsqueeze(1).expand(-1, len(theta), -1, -1)

    phi = torch.tensor((theta) * math.pi / 180, device=device)

    s = torch.sin(phi)
    c = torch.cos(phi)

    rot = torch.stack([torch.stack([c, s]), torch.stack([-s, c])])
    # rot = rot.permute(2, 0, 1)
    grid = mgrid.permute(1, 0, 2, 3)
    rot = rot.permute(2, 0, 1)
    grid = grid.view(len(theta), -1, 2)
    grid = grid.permute(0, 2, 1)

    mgrid_rot = torch.matmul(rot, grid)
    mgrid_rot = mgrid_rot.permute(0, 2, 1)
    mgrid_rot = mgrid_rot.view(len(theta), len(z), L, 2)
    mgrid_rot = mgrid_rot.permute(1, 0, 2, 3)

    mask = (torch.linalg.norm(mgrid_rot, ord=float("inf"), dim=3) <= 1).unsqueeze(3)

    # Reshape mgrid_rot such that it only has one batch dimension
    # i.e. the angle and coord batch dimension will be reduced from 2 to 1
    old_shape = mgrid_rot.shape
    mgrid_rot = mgrid_rot.reshape(
        -1, old_shape[2], 2
    )  # Shape: [len(z) * len(theta), L, 2]

    f_out, _ = f(mgrid_rot)
    f_out = f_out.reshape(old_shape[0], old_shape[1], old_shape[2], 1)
    f_out = mask * f_out  # Shape: [len(z), len(theta), L, 1]

    f_sum = torch.sum(f_out, dim=2)  # Shape: [len(z), len(theta), 1]

    # Normalization constant
    # normalization = torch.sum(mask) / (L**2)
    # zero_mask = normalization != 0
    # f_sum[zero_mask] = f_sum[zero_mask] / normalization[zero_mask]
    # f_sum = f_sum * 511

    output = f_sum[:, :, 0] / L  # Shape: [len(z), len(theta)]

    return output
