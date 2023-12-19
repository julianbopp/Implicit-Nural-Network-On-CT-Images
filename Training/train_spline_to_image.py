import math
import torch
from matplotlib import pyplot as plt

from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork

N = 256
lodopabSet = LodopabImage(N, pad=True)
padded_N = lodopabSet.padded_resolution
print(padded_N)

model_input = lodopabSet.coords
ground_truth = lodopabSet.pixels

interval = torch.linspace(-1, 1, steps=math.ceil(N * 1.0))
gridx, gridy = torch.meshgrid(interval, interval, indexing="xy")
model_input = torch.stack((gridx, gridy), dim=2)
model_input = model_input.reshape(-1, 2)
spline_network = SplineNetwork(padded_N)
optim = torch.optim.Adam(lr=1e-2, params=spline_network.parameters())

sampled_ground_truth, _ = lodopabSet.sample_image(
    model_input.unsqueeze(0).unsqueeze(0).cuda()
)

sampled_ground_truth = sampled_ground_truth.squeeze().unsqueeze(-1).cpu()
training_steps = 1000
spline_network.train()
for step in range(training_steps):
    print(f"training step: {step}")
    model_output, _ = spline_network(model_input)

    loss = (abs(model_output - sampled_ground_truth) ** 2).mean()
    print(loss.item())
    loss.backward()
    optim.step()
    optim.zero_grad()
    if step % 100 == 0:
        plt.imshow(model_output.view(math.ceil(N * 1.0), -1).cpu().detach().numpy())
        plt.show()
        plt.imshow(ground_truth.view(padded_N, padded_N))
        plt.show()
