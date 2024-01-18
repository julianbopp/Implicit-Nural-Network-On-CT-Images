import math

import numpy as np
from skimage.transform import radon
import torch
from matplotlib import pyplot as plt

from NeuralNetworks.spline import SplineNetwork

N = 40
c = 1
d = 5
square_image = torch.zeros([N, N])
square_image[N // 2 - c : N // 2 + c, N // 2 - d : N // 2 + d] = 1
square_image[N//2-d:N//2 + d, N//2:N//2 +2*c] = 1
square_image[N//2,N//2] = 1
#square_image[N//2+1,N//2] = 1
spline_representation = SplineNetwork(N)
spline_representation.weights = torch.nn.Parameter(square_image.view(-1, 1))


interval = torch.linspace(-1, 1, steps=math.ceil(N * 1.0))
gridx, gridy = torch.meshgrid(interval, interval, indexing="xy")
model_input = torch.stack((gridx, gridy), dim=2)
model_output, _ = spline_representation(model_input)
plt.imshow(model_output.view(N, N).detach().numpy())
plt.show()

radon = radon(model_output.view(N,N).detach().numpy(), theta=np.linspace(0.01, 179.9, num=N))
plt.imshow(radon)
plt.show()

device = "cpu"
t = torch.linspace(
    -math.sqrt(1), math.sqrt(1), steps=math.ceil(N), device=device
)
theta = torch.linspace(0.01, 179.9, steps=N, device=device)
#theta = torch.tensor([39])
#theta = torch.tensor([45])
#t = torch.tensor([0])
sinogram = torch.zeros((len(t), len(theta)))
for i, x in enumerate(t):
    print(i)
    for j, phi in enumerate(theta):
        print(phi)
        sinogram[i, j] = spline_representation.integrate_line(x, phi)

plt.imshow(sinogram.cpu().detach().numpy())
plt.show()
print(sinogram)
