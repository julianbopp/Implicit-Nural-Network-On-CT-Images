import math

import numpy as np
from skimage.transform import radon
import torch
from matplotlib import pyplot as plt
plt.ion()

import sys
sys.path.insert(0, '..')
from NeuralNetworks.spline import SplineNetwork


# my own Radon 
def Radon2(spline,pos_list,angles_list):
    interval = torch.linspace(-1, 1, steps=len(pos_list))
    gridx, gridy = torch.meshgrid(interval, interval, indexing="ij")
    model_input = torch.stack((gridx, gridy), dim=2)

    transform = torch.zeros((len(t),len(angles_list))).to(device)
    for i, theta in enumerate(angles_list):
        # Transform degree into radians and compute rotation matrix
        theta_rad = (theta * math.pi / 180.).to(device)
        s = torch.sin(theta_rad)
        c = torch.cos(theta_rad)
        rot = torch.stack(
            [torch.stack([c, s]), torch.stack([-s, c])]
        )
        inp = torch.matmul(model_input, rot.T)
        model_output, _ = spline_representation(inp)
        transform[:,i] = model_output.sum((-2,-1))
    return transform

device = "cpu"

N = 20
c = 0
d = 0
sx = 5
sy = 5
t = torch.linspace(-1, 1, steps=N, device=device)
theta = torch.linspace(0.0, 180, steps=180, device=device)


square_image = torch.zeros([N, N])
# square_image[sx + N // 2 - c : sx + N // 2 + c, sy + N // 2 - d : sy + N // 2 + d] = 1
# square_image[N//2-d:N//2 + d, N//2:N//2 +2*c] = 1
square_image[sx + N//2, sy + N//2] = 1
#square_image[N//2+1,N//2] = 1
spline_representation = SplineNetwork(N)
spline_representation.weights = torch.nn.Parameter(square_image.view(-1, 1))


plt.figure(1)
interval = torch.linspace(-1, 1, steps=math.ceil(N * 1.0))
gridx, gridy = torch.meshgrid(interval, interval, indexing="ij")
model_input = torch.stack((gridx, gridy), dim=2)
model_output, _ = spline_representation(model_input)
plt.imshow(model_output.view(N, N).detach().numpy())
# plt.show()


spline = spline_representation
pos_list = t
angles_list = theta
radon_img = Radon2(spline,pos_list,angles_list)
plt.figure(2)
plt.clf()
plt.imshow(radon_img.detach().cpu().numpy())
ax = plt.gca()
ax.set_xticks(np.linspace(0,len(theta),10))
ax.set_xticklabels(np.linspace(theta.min().item(), theta.max().item(),10).astype(np.int16))
plt.colorbar()


# plt.figure(2)
# plt.clf()
# radon_img = radon(model_output.view(N,N).detach().numpy(), theta)
# plt.imshow(radon_img)
# ax = plt.gca()
# ax.set_xticks(np.linspace(0,len(theta),10))
# ax.set_xticklabels(np.linspace(0.01, 90,10).astype(np.int16))
# plt.colorbar()
# # plt.show()


sinogram = torch.zeros((len(t), len(theta)))
for i, x in enumerate(t):
    print(i)
    for j, phi in enumerate(theta):
        # print(phi)
        sinogram[i, j] = spline_representation.integrate_line(x, phi)*N/2

plt.figure(3)
plt.clf()
plt.imshow(sinogram.cpu().detach().numpy())
ax = plt.gca()
ax.set_xticks(np.linspace(0,len(theta),10))
ax.set_xticklabels(np.linspace(theta.min().item(), theta.max().item(),10).astype(np.int16))
plt.colorbar()

plt.figure(4)
model_output, _ = spline_representation(model_input)
plt.imshow(model_output.view(N, N).detach().numpy())
plt.show()


# kk = 15
# print(theta[kk])
# plt.figure(5)
# plt.clf()
# plt.plot(sinogram[:,kk].detach().cpu().numpy(),'o')





# phi = torch.tensor(0)
# sinogram_line = torch.zeros(len(t))
# for i, x in enumerate(t):
#     print(i)
#     sinogram_line[i] = spline_representation.integrate_line(x, phi)*N/2


# plt.figure(2)
# plt.clf()
# plt.plot(radon_img[:,0],'o')

# plt.figure(3)
# plt.clf()
# plt.plot(sinogram_line.detach().cpu().numpy(),'o')

