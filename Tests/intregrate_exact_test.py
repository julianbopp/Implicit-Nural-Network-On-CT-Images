import math

import numpy as np
from skimage.transform import radon
import torch
from matplotlib import pyplot as plt

from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork

N = 128

c = 1
sx = 0
sy = 0


def Radon2(spline, pos_list, angles_list):
    interval = torch.linspace(-1, 1, steps=len(pos_list))
    gridx, gridy = torch.meshgrid(interval, interval, indexing="ij")
    model_input = torch.stack((gridx, gridy), dim=2)

    transform = torch.zeros((len(t), len(angles_list))).to(device)
    for i, theta in enumerate(angles_list):
        # Transform degree into radians and compute rotation matrix
        theta_rad = (theta * math.pi / 180.0).to(device)
        s = torch.sin(theta_rad)
        c = torch.cos(theta_rad)
        rot = torch.stack([torch.stack([c, s]), torch.stack([-s, c])])
        inp = torch.matmul(model_input, rot.T)
        model_output, _ = spline_representation(inp)
        transform[:, i] = model_output.sum((-2, -1))
    return transform


square_image = torch.zeros([N, N])
square_image[N // 2 + sx, N // 2 + sy] = 1
# square_image[N//2+1,N//2] = 1
lodopabImage = LodopabImage(N, pad=False)
square_image = lodopabImage.image
spline_representation = SplineNetwork(N, circle=True)
spline_representation.weights = torch.nn.Parameter(square_image.view(-1, 1))


interval = torch.linspace(-1, 1, steps=math.ceil(N * 1.0))
gridx, gridy = torch.meshgrid(interval, interval, indexing="xy")
model_input = torch.stack((gridx, gridy), dim=2)
model_output, _ = spline_representation(model_input)
plt.imshow(model_output.view(N, N).detach().numpy())
plt.show()
device = "cpu"
theta = np.linspace(0.0, 180.0, num=N)
t = torch.linspace(-math.sqrt(1), math.sqrt(1), steps=N, device=device)
theta = torch.linspace(0.0, 180.0, steps=180)
#radon = Radon2(spline_representation, t, theta)
#plt.imshow(radon.detach().cpu().numpy())
#plt.colorbar()
#plt.show()


sinogram = np.zeros((len(t), len(theta)))
#sinogram.requires_grad = False
for i, x in enumerate(t):
    print(i)
    for j, phi in enumerate(theta):
        tmp = spline_representation.integrate_line(x,phi)
        if torch.is_tensor(tmp):
            tmp = tmp.detach().numpy()
        sinogram[i, j] = tmp

sinogram = torch.from_numpy(sinogram)
plt.imshow(sinogram.cpu().detach().numpy())
ax = plt.gca()
ax.set_xticklabels(
    np.linspace(theta.min().item(), theta.max().item(), 10).astype(np.int16)
)
plt.colorbar()
plt.show()
print(sinogram)
