# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.audio import SignalNoiseRatio

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork
from RadonTransform.radon_transform import batch_radon_siren

N = 128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CIRCLE = True

lodopabSet = LodopabImage(N, pad=CIRCLE)
padded_N = lodopabSet.padded_resolution

image = lodopabSet.image
sinogram = lodopabSet.get_radon_transform()
plt.imshow(image.view(padded_N, padded_N).cpu())
plt.show()

model_input = lodopabSet.coords
model_input = model_input.to(device)

spline_network = SplineNetwork(padded_N, circle=CIRCLE)
spline_network = spline_network.to(device)

angleSet = AngleSet(180, rad=False)
angleSet.angles = angleSet.angles.to(device)
angleLoader = DataLoader(angleSet, batch_size=180, shuffle=True)

coordSet = CoordSet(padded_N, circle=CIRCLE)
coordSet.coords = coordSet.coords.to(device)
coordLoader = DataLoader(coordSet, batch_size=padded_N, shuffle=True)


optim = torch.optim.Adam(lr=0.1, params=spline_network.parameters())

sample_points = int(padded_N)
training_steps = 500
loss_total = []

for step in range(training_steps):
    print(f"training step = {step}")
    if step % 10 == 0:
        model_output, _ = spline_network(model_input)
        plt.imshow(model_output.reshape(padded_N, padded_N).cpu().detach().numpy())
        plt.show()
        plt.imshow(image.reshape(padded_N, padded_N).cpu().detach().numpy())
        plt.show()
        snr = SignalNoiseRatio()
        print(
            snr(
                model_output.view(-1, 1).squeeze().cpu(),
                image.view(-1, 1).squeeze().cpu(),
            )
        )
    for angle, angle_idx in angleLoader:
        for coords, coords_idx in coordLoader:
            optim.zero_grad()

            radon_output = batch_radon_siren(
                coords,
                spline_network,
                sample_points,
                theta=-angle,
                device=device,
                circle=CIRCLE,
            )

            # Create a grid to sample radon ground truth
            # reshape angles to [-1,1]
            reshaped_angle = (angle / 179) * 2 - 1

            grid = torch.stack(
                torch.meshgrid(
                    reshaped_angle,
                    coords,
                    indexing="xy",
                ),
                dim=2,
            )

            sampled_radon, _ = lodopabSet.sample_radon(grid.unsqueeze(0))
            sampled_radon = sampled_radon.squeeze()

            loss = ((radon_output - sampled_radon) ** 2).mean()
            loss_total.append(loss.item())

            loss.backward()
            optim.step()
    print(torch.tensor(loss_total[-180 * coordSet.__len__() :]).mean().item())
