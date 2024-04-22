import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import CoordSet, AngleSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork

N = 8

lodopabImage = LodopabImage(N, pad=True)
square_image = lodopabImage.image
sinogram = lodopabImage.get_radon_transform(noise=False)
sinogram.requires_grad = True

padded_N = lodopabImage.padded_resolution
coord_size = 1
angle_size = 1

spline_network = SplineNetwork(N)

angles = torch.linspace(0, 180, 180)
detection_plane = torch.linspace(-1, 1, steps=padded_N)

optim = torch.optim.Adam(lr=0.1, params=spline_network.parameters())

training_epochs = 100


def mse_loss(ground_truth, x):
    if ground_truth.shape != x.shape:
        print("inputs of loss function have non compatable shapes ")

    return torch.mean((ground_truth - x) ** 2)


for epoch in range(training_epochs):
    exact_integration_sinogram = torch.zeros((len(detection_plane), len(angles)))

    optim.zero_grad()
    for coord_id, coord in enumerate(detection_plane):
        print(coord_id)
        for angle_id, angle in enumerate(angles):

            exact_integration_sinogram[coord_id, angle_id] = spline_network.integrate_line(coord, angle) * N/2

    loss = mse_loss(sinogram, exact_integration_sinogram)
    print(loss)
    loss.backward()
    optim.step()

    if epoch % 10 == 0:
        plt.imshow(spline_network.weights.view(N,N).detach().numpy())
        plt.show()
