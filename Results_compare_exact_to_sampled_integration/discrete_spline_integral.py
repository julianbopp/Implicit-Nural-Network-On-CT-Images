import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork
from RadonTransform.radon_transform import batch_radon_siren
from skimage.transform import iradon
from DatasetClasses.utils import SNR

N = 362

lodopabImage = LodopabImage(N, pad=False)
square_image = lodopabImage.image
spline_representation = SplineNetwork(N, circle=False)
spline_representation.weights = torch.nn.Parameter(square_image.view(-1, 1))


interval = torch.linspace(-1, 1, steps=N)
gridx, gridy = torch.meshgrid(interval, interval, indexing="xy")
model_input = torch.stack((gridx, gridy), dim=2)
model_output, _ = spline_representation(model_input)
plt.imshow(model_output.view(N, N).detach().numpy())
plt.show()
device = "cpu"
theta = np.linspace(0.0, 180.0, num=N)
t = torch.linspace(-math.sqrt(1), math.sqrt(1), steps=N, device="cpu")


theta2 = torch.linspace(0, 180, steps=N, device="cpu")
N_Samples = [32,64,128,256, 300, 362, 512]
for i in N_Samples:
    angleSet = AngleSet(N, rad=False)
    coordSet = CoordSet(N, circle=True)

    angleLoader = DataLoader(angleSet, batch_size=10)
    coordLoader = DataLoader(coordSet, batch_size=40)

    sampled_line_integral_sinogram = torch.zeros(N, N, requires_grad=False)
    for angles, angleIdx in angleLoader:
        print(angles)
        for coords, coordIdx in coordLoader:
            # Reshape coordIdx and angleIdx
            coordIdx_unsq = coordIdx.unsqueeze(1)  # shape [256, 1]
            angleIdx_unsq = angleIdx.unsqueeze(0)  # shape [1, 10]

            # Create a grid of indices
            coordIdx_grid, angleIdx_grid = torch.meshgrid(
                coordIdx_unsq[:, 0], angleIdx_unsq[0, :], indexing="ij"
            )

            sampled_line_integral_sinogram[coordIdx_grid,angleIdx_grid] = batch_radon_siren(
                coords, spline_representation, i, angles+90
            ).detach()

    name = f"sampled_line_integral_sinogram_{i}2.pt"
    torch.save(sampled_line_integral_sinogram, f"Sampled_line_integrals/{name}")

plt.imshow(sampled_line_integral_sinogram.detach().numpy())
plt.colorbar()
plt.show()
print("calculated sampled line integral sinogram")

exact_integral_sinogram = torch.load("sinogram128.pt")
plt.imshow(exact_integral_sinogram.detach().numpy())
plt.colorbar()
plt.show()

snr1 = SNR(sampled_line_integral_sinogram.detach().numpy(), exact_integral_sinogram.detach().numpy())
print(snr1)

