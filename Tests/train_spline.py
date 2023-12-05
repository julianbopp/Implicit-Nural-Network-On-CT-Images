import math
import torch
import torch.nn.functional as F
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
print(ground_truth.shape)


def sample_image(coords):
    # Normalize coordinates from [-1, 1] to [0, N-1]
    coords = (coords + 1) * (N - 1) / 2
    image = lodopabImage.clone().reshape(1, 1, N, N)

    # Grid sample expects grid values in the range of [-1, 1], so normalize again
    coords = coords * 2 / (N - 1) - 1

    # Reshape coords to fit grid_sample
    new_coords = coords.clone().permute(2, 0, 1, 3)

    # Use grid_sample for interpolation
    sampled_img = F.grid_sample(
        image,
        new_coords,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return sampled_img.squeeze(), coords


ground_truth = batch_radon_siren(
    torch.linspace(-math.sqrt(2), math.sqrt(2), steps=int(N * math.sqrt(2))),
    sample_image,
    30,
    theta=torch.linspace(0, 180, steps=180),
)
# ground_truth = ground_truth.reshape(N, N)

angleSet = AngleSet(180, rad=False)
angleLoader = DataLoader(angleSet, batch_size=30, shuffle=True)

coordSet = CoordSet(int(N * math.sqrt(2)), circle=False)
coordLoader = DataLoader(coordSet, batch_size=N**2)


optim = torch.optim.Adam(lr=1e-1, params=spline_network.parameters())

sample_points = 300
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
    model_output, _ = spline_network(model_input)
    if step % 5 == 0:
        plt.imshow(model_output.reshape(N, N).detach().numpy())
        plt.show()
    print(torch.tensor(loss_total[-180 * coordSet.__len__() :]).mean().item())
