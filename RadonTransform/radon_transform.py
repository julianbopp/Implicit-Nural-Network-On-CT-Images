import math
import torch
from torch.nn import ConstantPad2d
from torchvision.transforms import v2


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


def batch_radon(z, f, L, theta=None, CUDA=False):
    """
    z : tensor of points on radon projection plane
    theta : List [a, b] a start degree, b end degree
    f : function to sample from (support on [-1, 1])
    L : number of sample points on one Line
    """
    t = torch.linspace(-math.sqrt(2), math.sqrt(2), steps=L)
    output = torch.zeros(len(z), len(theta))

    if CUDA:
        t = t.cuda()
        z = z.cuda()
        output = output.cuda()

    mgrid = torch.stack(torch.meshgrid(z, t, indexing="xy"))
    mgrid = mgrid.permute(2, 0, 1)

    mgrid = mgrid.unsqueeze(1).expand(-1, len(theta), -1, -1)

    phi = torch.tensor((theta) * math.pi / 180 + math.pi / 2)

    s = torch.sin(phi)
    c = torch.cos(phi)

    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    rot = rot.permute(2, 0, 1)
    if CUDA:
        rot = rot.cuda()

    # mgrid_rot = mgrid @ rot.t()
    mgrid_rot = torch.matmul(rot, mgrid)

    mgrid_rot = mgrid_rot.permute(0, 1, 3, 2)
    mask = (torch.linalg.norm(mgrid_rot, ord=float("inf"), dim=3) <= 1).unsqueeze(3)
    f_out, _ = f(mgrid_rot)
    f_out = mask * f_out

    f_sum = torch.sum(f_out, dim=2)

    output = f_sum[:, :, 0]

    return output

    pass
