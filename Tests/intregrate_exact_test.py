import math

import torch
from matplotlib import pyplot as plt

from NeuralNetworks.spline import SplineNetwork

N = 128
c = 5
square_image = torch.zeros([N, N])
square_image[N // 2 - c : N // 2 + c, N // 2 - c : N // 2 + c] = 1

spline_representation = SplineNetwork(N)
spline_representation.weights = torch.nn.Parameter(square_image.view(-1, 1))


interval = torch.linspace(-1, 1, steps=math.ceil(N * 1.0))
gridx, gridy = torch.meshgrid(interval, interval, indexing="xy")
model_input = torch.stack((gridx, gridy), dim=2)
model_output, _ = spline_representation(model_input)
plt.imshow(model_output.view(N, N).detach().numpy())
plt.show()
