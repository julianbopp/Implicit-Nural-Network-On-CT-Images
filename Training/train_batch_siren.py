import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.siren import Siren
from RadonTransform.radon_transform import batch_radon_siren

CUDA = torch.cuda.is_available()
device = torch.device("cuda") if CUDA else torch.device("cpu")

RESOLUTION = 182
CIRCLE = True

lodopabImage = LodopabImage(RESOLUTION)

RESOLUTION = lodopabImage.get_padded_resolution()
img_siren = Siren(
    in_features=2,
    out_features=1,
    hidden_features=lodopabImage.resolution,
    hidden_layers=3,
    outermost_linear=True,
)
img_siren = img_siren.to(device)

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())


# Get ground truth radon image
ground_truth = lodopabImage.get_radon_transform()
model_input = lodopabImage.coords
model_input = model_input.to(device)
angleSet = AngleSet(180, rad=False)
if CUDA:
    angleSet.angles = angleSet.angles.cuda()
angleLoader = DataLoader(angleSet, batch_size=10, shuffle=True)

coordSet = CoordSet(RESOLUTION, circle=CIRCLE)
if CUDA:
    coordSet.coords = coordSet.coords.cuda()
coordLoader = DataLoader(coordSet, batch_size=RESOLUTION // 10, shuffle=True)
coordIter = iter(coordLoader)

training_steps = 100
sample_points = RESOLUTION

if CUDA:
    ground_truth = ground_truth.cuda()

step = 0
loss_total = []

for step in range(training_steps):
    print(f"training step = {step}")
    for angle, angle_idx in angleLoader:
        for coords, coords_idx in coordLoader:
            # Sort angles and coords
            radon_output = batch_radon_siren(
                coords,
                img_siren,
                sample_points,
                theta=-angle,
                device=device,
                SIREN=True,
            )
            # reshape angles to [-1,1]
            reshaped_angles = (angle / 179) * 2 - 1

            gridx, gridy = torch.meshgrid(reshaped_angles, coords, indexing="xy")
            grid = torch.stack((gridx, gridy), dim=2)
            sampled_radon, _ = lodopabImage.sample_radon(grid.unsqueeze(0).to("cpu"))
            sampled_radon = sampled_radon.squeeze()

            loss = ((radon_output.cpu() - sampled_radon.cpu()) ** 2).mean()
            loss_total.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

    if step % 20 == 0:
        img_siren = img_siren.to("cpu")
        model_input = model_input.to("cpu")
        model_output, _ = img_siren(model_input)
        model_output = model_output.view(RESOLUTION, RESOLUTION).cpu()
        img_siren = img_siren.to(device)
        model_input = model_input.to(device)

        plt.imshow(model_output.cpu().detach().numpy().squeeze())
        plt.show()
        plt.imshow(sampled_radon.cpu().detach().numpy().squeeze())
        plt.show()

    print(torch.tensor(loss_total[-180 * coordSet.__len__() :]).mean().item())

torch.save(img_siren.state_dict(), "../img_batch_siren.pt")
