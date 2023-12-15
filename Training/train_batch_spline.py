import math
import torch
from matplotlib import pyplot as plt
from skimage.transform import radon
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork
from RadonTransform.radon_transform import batch_radon_siren

N = 128
spline_network = SplineNetwork(N)

lodopabSet = LodopabImage(N)
lodopabLoader = DataLoader(lodopabSet, batch_size=lodopabSet.__len__())
model_input, lodopabImage = next(iter(lodopabLoader))
ground_truth_image = lodopabImage.reshape(N, N).detach().numpy()
ground_truth_radon = radon(ground_truth_image, circle=False)
ground_truth = torch.from_numpy(ground_truth_radon)
plt.imshow(ground_truth_radon)
plt.show()


angleSet = AngleSet(180, rad=False)
angleLoader = DataLoader(angleSet, batch_size=10, shuffle=True)

coordSet = CoordSet(int(N * math.sqrt(2)), circle=False)
coordLoader = DataLoader(coordSet, batch_size=int((N * math.sqrt(2)) / 4))


optim = torch.optim.Adam(lr=0.1, params=spline_network.parameters())

sample_points = int(N * math.sqrt(2)) // 2
training_steps = 100
loss_total = []

for step in range(training_steps):
    print(f"training step = {step}")
    for angle, angle_idx in angleLoader:
        for coords, coords_idx in coordLoader:
            optim.zero_grad()

            radon_output = batch_radon_siren(
                coords, spline_network, sample_points, theta=angle
            )

            # Create a grid to sample radon ground truth
            # reshape angles to [-1,1]
            reshaped_angle = (angle / 179) * 2 - 1
            grid = torch.stack(torch.meshgrid(coords, reshaped_angle)).permute(
                1, 2, 0
            )  # Shape: (len(angle), len(coords), 2)
            sampled_radon = lodopabSet.sample_image(
                grid.unsqueeze(0), torch.from_numpy(ground_truth_radon)
            )
            loss = ((radon_output - sampled_radon) ** 2).mean()
            loss_total.append(loss.item())

            loss.backward()
            optim.step()
    model_output, _ = spline_network(model_input)
    if step % 5 == 0:
        plt.imshow(model_output.reshape(N, N).detach().numpy())
        plt.show()
    print(torch.tensor(loss_total[-180 * coordSet.__len__() :]).mean().item())
