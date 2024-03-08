import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworks.spline import SplineNetwork
from NeuralNetworks.siren import Siren
from DatasetClasses.lodopabimage import LodopabImage
from DatasetClasses.utils import SNR
from DatasetClasses.utils import addGaussianNoise


RESOLUTION = 36

image_util = LodopabImage(resolution=RESOLUTION, pad=False)
ground_truth_image = image_util.image

PADDED_RESOLUTION = image_util.padded_resolution

input = image_util.coords

def train(ground_truth, model_output, optimizer):

    optimizer.zero_grad()

    loss = ((model_output - ground_truth)**2).mean()
    loss.backward()

    optimizer.step()





training_steps = 1000

stds = [0,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

spline_snr = torch.zeros([len(stds)])
siren_snr = torch.zeros([len(stds)])
noise_snr = torch.zeros([len(stds)])
for j in range(len(stds)):

    spline_network = SplineNetwork(RESOLUTION)

    siren_network = Siren(
        in_features=2,
        out_features=1,
        hidden_features=RESOLUTION,
        hidden_layers=3,
        outermost_linear=True,
    )

    siren_network.cuda()

    optimizer_spline = torch.optim.Adam(spline_network.parameters(), lr=1e-3)
    optimizer_siren = torch.optim.Adam(params=siren_network.parameters(), lr=1e-3)

    std = stds[j]
    ground_truth = torch.from_numpy(addGaussianNoise(ground_truth_image.detach().numpy(), 0, std)).view(-1)

    for i in range(training_steps):
        output_spline, _ = spline_network(input)
        output_siren, _ = siren_network(input.cuda())
        output_siren = output_siren.view(-1)

        train(ground_truth, output_spline, optimizer_spline)
        train(ground_truth.cuda(), output_siren, optimizer_siren)


    #plt.imshow(siren_network(input.cuda())[0].cpu().view(RESOLUTION, RESOLUTION).detach().numpy())
    #plt.show()
    #plt.imshow(ground_truth.cpu().view(RESOLUTION,RESOLUTION).detach().numpy())
    #plt.show()

    spline_snr[j] = SNR(ground_truth_image.view(-1).detach().numpy(), output_spline.cpu().view(-1).detach().numpy())
    siren_snr[j] = SNR(ground_truth_image.view(-1).detach().numpy(), output_siren.cpu().view(-1).detach().numpy())

    noise_snr[j] = SNR(ground_truth_image.view(-1).detach().numpy(), ground_truth.view(-1).detach().numpy())

plt.figure(1)
plt.plot(noise_snr, spline_snr, 'ro-', label="spline snr")
plt.plot(noise_snr, siren_snr, 'b+--', label="siren snr")
plt.xlabel("noise snr")
plt.ylabel("fitting snr")
plt.legend()
plt.title("Fitting SNR over Noise SNR")

plt.show()








