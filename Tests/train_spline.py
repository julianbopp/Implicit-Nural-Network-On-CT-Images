import math
import torch
from skimage.transform import radon
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork
from RadonTransform.radon_transform import batch_radon

N = 32
spline_network = SplineNetwork(N)

lodopabSet = LodopabImage(N)
lodopabLoader = DataLoader(lodopabSet, batch_size=lodopabSet.__len__())
_, lodopabImage = next(iter(lodopabLoader))
ground_truth_image = lodopabImage.reshape(N, N).detach().numpy()
ground_truth_radon = radon(ground_truth_image, circle=False)
ground_truth = torch.from_numpy(ground_truth_radon)

angleSet = AngleSet(180, rad=False)
angleLoader = DataLoader(angleSet, batch_size=1, shuffle=True)

coordSet = CoordSet(int(N * math.sqrt(2)), circle=False)
coordLoader = DataLoader(coordSet, batch_size=64)


optim = torch.optim.Adam(lr=1e-4, params=spline_network.parameters())

sample_points = 10
training_steps = 100
loss_total = []

for step in range(training_steps):
    print(f"training step = {step}")
    for angle, angle_idx in angleLoader:
        for coords, coords_idx in coordLoader:
            optim.zero_grad()

            radon_output = batch_radon(
                coords, spline_network, sample_points, theta=angle
            )

            # Reshape coordIdx and angleIdx
            coordIdx_unsq = coords_idx.unsqueeze(1)  # shape [256, 1]
            angleIdx_unsq = angle_idx.unsqueeze(0)  # shape [1, 10]

            # Create a grid of indices
            coordIdx_grid, angleIdx_grid = torch.meshgrid(
                coordIdx_unsq[:, 0], angleIdx_unsq[0, :], indexing="ij"
            )

            loss = (
                (radon_output - ground_truth[coordIdx_grid, angleIdx_grid]) ** 2
            ).mean()
            loss_total.append(loss.item())

            loss.backward()
            optim.step()

    print(torch.tensor(loss_total[-180 * coordSet.__len__() :]).mean().item())
