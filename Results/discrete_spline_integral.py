import math
import numpy as np
import torch
from matplotlib import pyplot as plt

from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork
from RadonTransform.radon_transform import batch_radon_siren
from skimage.transform import iradon

N = 128


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

lodopabImage = LodopabImage(N, pad=False)
square_image = lodopabImage.image
spline_representation = SplineNetwork(N, circle=True)
spline_representation.weights = torch.nn.Parameter(square_image.view(-1, 1))


interval = torch.linspace(-1, 1, steps=N)
gridx, gridy = torch.meshgrid(interval, interval, indexing="xy")
model_input = torch.stack((gridx, gridy), dim=2)
model_output, _ = spline_representation(model_input)
plt.imshow(model_output.view(N, N).detach().numpy())
plt.show()
device = "cpu"
theta = np.linspace(0.0, 180.0, num=N)
t = torch.linspace(-math.sqrt(1), math.sqrt(1), steps=N, device=device)
theta = torch.linspace(0.0, 180.0, steps=N)
radon = Radon2(spline_representation, t, theta)
plt.imshow(radon.detach().cpu().numpy())
plt.colorbar()
plt.show()


theta2 = torch.linspace(0, 180, steps=180)
sampled_line_integral_sinogram = batch_radon_siren(t, spline_representation, math.ceil(128*math.sqrt(2)), theta2+90, device="cpu", circle=True, SIREN=False)
plt.imshow(sampled_line_integral_sinogram.detach().numpy())
plt.colorbar()
plt.show()
print("calculated sampled line integral sinogram")

exact_integral_sinogram = torch.load("sinogram128.pt")
plt.imshow(exact_integral_sinogram.detach().numpy())
plt.colorbar()
plt.show()
reconstruction_fbp = iradon(sampled_line_integral_sinogram.detach().numpy(), theta=theta2.detach().numpy(), filter_name='ramp')
reconstruction_fbp_exact = iradon(exact_integral_sinogram.detach().numpy(), theta=theta.detach().numpy(), filter_name='ramp')

plt.imshow(reconstruction_fbp_exact)
plt.show()
plt.imshow(reconstruction_fbp)
plt.show()


