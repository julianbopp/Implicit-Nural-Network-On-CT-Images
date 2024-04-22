import math
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork
from RadonTransform.radon_transform import batch_radon_siren
from skimage.transform import iradon
from DatasetClasses.utils import SNR


PATH = "Sampled_line_integrals/"
print(os.getcwd())
names = [32,64,128,256,300,362,400,500,510,511,512,515,530,800]
ground_truth = torch.load(f"{PATH}sampled_line_integral_sinogram_5122.pt")
ground_truth = ground_truth/ground_truth.max()
exact_integration = torch.load("sinogram362.pt")
exact_integration = exact_integration/exact_integration.max()

snr_data = np.zeros(len(names))

i = 0
for name in names:
    fullpath = f"{PATH}sampled_line_integral_sinogram_{name}2.pt"
    sinogram = torch.load(fullpath)
    sinogram = sinogram/sinogram.max()

    snr_data[i] = SNR(sinogram,ground_truth)
    i = i + 1


exact_snr = SNR(exact_integration, ground_truth)
print(snr_data)
fig, ax = plt.subplots()
ax.plot(names, snr_data, "x--", label="sampled integration")
ax.plot(names, [exact_snr]*len(names), label="exact integration")
ax.set(xlabel="amount of line samples", ylabel="signal to noise ratio", title=f"Comparison of Sampled to Exact Integral")
ax.set_xlim(32,800)
ax.grid(which="minor", alpha=0.3, linestyle="--")
ax.grid(which="major", alpha=0.9)
ax.minorticks_on()
fig.autofmt_xdate()
ax.set_xticks([32,64,128,256,300,362,400,512,800])
filtered_data = [i for i in snr_data if not math.isinf(i)]
filtered_data = filtered_data[0:-5]
ax.legend()
ax.set_yticks(filtered_data + [exact_snr] )
plt.savefig("comparison_exact_and_sampled.svg", format="svg")
plt.show()
plt.imshow(ground_truth)
plt.show()
