import torch
import numpy as np
import matplotlib.pyplot as plt

from DatasetClasses.utils import SNR
from DatasetClasses.lodopabimage import LodopabImage

RES = 362
imageHelper = LodopabImage(RES, pad=False)
ground_truth = imageHelper.image

noise_levels = [0, 1, 2, 3, 4,5]
noise_std = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1]

siren_data = np.zeros([6, 30])
spline_data = np.zeros([6, 30])

i = 0

fig, axs = plt.subplots(3,2, figsize=(10,10))

for noise in noise_levels:
    j = 0
    for step in range(300):
        if step % 10 == 0:
            siren_output = torch.load(f"Output/siren_output_noise{noise}_step{step}.pt")
            spline_output = torch.load(f"Output/spline_output_noise{noise}_step{step}.pt")


            siren_data[i,j] = SNR(ground_truth, siren_output.view(RES,RES).detach().numpy())
            spline_data[i,j] = SNR(ground_truth, spline_output.view(RES,RES).detach().numpy())


            #print(f"siren_snr: {siren_snr}")
            #print(f"spline_snr: {spline_snr}")

            j = j + 1

    ax = axs[int(i/2), i % 2]
    line1 = ax.plot(np.arange(0,30,step=1)*10,siren_data[i,:], "--", label="Siren")
    line2 = ax.plot(np.arange(0,30,step=1)*10,spline_data[i,:], label="Spline")
    ax.set(xlabel="iterations", ylabel="signal to noise ratio", title=f"Noise $\sigma$: {noise_std[i]}")
    ax.legend()
    ax.set_xlim(0, 290)
    ax.grid(which="minor", alpha=0.3, linestyle="--")
    ax.grid(which="major", alpha=0.9)
    ax.minorticks_on()

    print(siren_data)
    print(spline_data)

    i = i + 1
plt.tight_layout()
plt.savefig("compare_spline_fit_to_gt.svg", format="svg")
plt.show()


print("test")


