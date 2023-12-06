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
plt.imshow(ground_truth_radon)
plt.show()


def sample_image(coords):
    # Normalize coordinates from [-1, 1] to [0, N-1]
    coords = (coords + 1) * (N - 1) / 2
    image = lodopabImage.clone().reshape(N, N)
    image = image.unsqueeze(0).unsqueeze(0)

    # Grid sample expects grid values in the range of [-1, 1], so normalize again
    coords = coords * 2 / (N - 1) - 1

    # Reshape coords to fit grid_sample
    new_coords = coords.clone().permute(2, 0, 1, 3)

    # Use grid_sample for interpolation
    sampled_img = torch.zeros(
        [new_coords.shape[0], 1, new_coords.shape[1], new_coords.shape[2]]
    )

    for i in range(new_coords.shape[0]):
        sampled_img[i, :, :, :] = F.grid_sample(
            image,
            new_coords[i, :, :, :].unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).unsqueeze(2)

    return sampled_img.permute(2, 3, 0, 1), coords


ground_truth = batch_radon_siren(
    torch.linspace(-math.sqrt(2), math.sqrt(2), steps=int(N * math.sqrt(2))),
    sample_image,
    int(N * math.sqrt(2)),
    theta=torch.arange(0, 180, 1) - 90,
)
plt.imshow(ground_truth.detach().numpy())
plt.show()
print(ground_truth.shape)

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

            # model_before, _ = spline_network(model_input)
            # radon_output = radon_transform(model_before.view(1, N, N))

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

            # Create a grid to sample radon ground truth
            # reshape angles to [-1,1]
            angle = (angle / 179) * 2 - 1
            grid = torch.stack(torch.meshgrid(coords, angle)).permute(
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
