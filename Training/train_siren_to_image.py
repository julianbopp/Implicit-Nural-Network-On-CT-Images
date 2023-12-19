import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.siren import Siren

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

RESOLUTION = 364
CIRCLE = True

lodopabSet = LodopabImage(RESOLUTION, pad=False)
model_input = lodopabSet.coords
model_input = model_input.to(device)
ground_truth, _ = lodopabSet.sample_image(
    model_input.view(1, lodopabSet.padded_resolution, lodopabSet.padded_resolution, 2)
)
ground_truth = ground_truth.squeeze().view(-1, 1)

ground_truth = lodopabSet.pixels
ground_truth = ground_truth.to(device)
img_siren = Siren(
    in_features=2,
    out_features=1,
    hidden_features=lodopabSet.padded_resolution,
    hidden_layers=3,
    outermost_linear=True,
)
img_siren = img_siren.to(device)

optim = torch.optim.Adam(lr=1e-3, params=img_siren.parameters())

angleSet = AngleSet(180, rad=False)
coordSet = CoordSet(lodopabSet.padded_resolution, circle=CIRCLE)
angleLoader = DataLoader(angleSet, batch_size=180, shuffle=False)
coordLoader = DataLoader(
    coordSet, batch_size=lodopabSet.padded_resolution, shuffle=False
)


x = torch.linspace(-1, 1, steps=100)
y = torch.linspace(-0.5, 0.5, steps=50)
gridx, gridy = torch.meshgrid(x, y, indexing="xy")
grid = torch.stack((gridx, gridy), dim=2)
grid = grid.unsqueeze(0)
grid = grid.to(device)
sampled_image, _ = lodopabSet.sample_image(grid)

plt.imshow(sampled_image.squeeze().cpu().detach().numpy())
plt.show()


training_steps = 1000
sample_points = lodopabSet.padded_resolution

loss_total = []

for step in range(training_steps):
    print(f"training step = {step}")
    model_output, _ = img_siren(model_input)
    loss = ((model_output - ground_truth) ** 2).mean()
    print(loss.item())

    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 50 == 0:
        plt.imshow(
            model_output.view(lodopabSet.padded_resolution, -1).cpu().detach().numpy()
        )
        plt.show()
        plt.imshow(
            ground_truth.view(lodopabSet.padded_resolution, -1).cpu().detach().numpy()
        )
        plt.show()


z = torch.linspace(-1, 1, steps=100, device=device)
f = img_siren
theta = torch.arange(0, 180, step=1, device=device)
L = 80
# batch_radon = batch_radon_siren(z, f, L, -theta, device=device)
# plt.imshow(batch_radon.cpu().detach().numpy())
# plt.show()
