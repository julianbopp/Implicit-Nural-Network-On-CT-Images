import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import radon, iradon
from torch.utils.data import DataLoader
from torchmetrics.audio import SignalNoiseRatio

from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.siren import Siren

CUDA = False

resolution = 256
img_siren = Siren(
    in_features=2,
    out_features=1,
    hidden_features=resolution,
    hidden_layers=3,
    outermost_linear=True,
)

if CUDA:
    img_siren.load_state_dict(torch.load("../img_siren.pt"))
else:
    img_siren.load_state_dict(
        torch.load("../img_siren.pt", map_location=torch.device("cpu"))
    )
img_siren.eval()

dataset = LodopabImage(resolution)
dataloader = DataLoader(dataset, batch_size=dataset.__len__())

model_input, ground_truth = next(iter(dataloader))

ground_truth = ground_truth.view(1, resolution, -1)
ground_truth_image = ground_truth.reshape(resolution, resolution).detach().numpy()
ground_truth_radon = radon(ground_truth_image, np.arange(180), circle=True)
ground_truth = torch.from_numpy(ground_truth_radon).unsqueeze(0)

model_output, coords = img_siren(model_input)
model_output_orig = model_output.view(resolution, resolution)
model_output = radon(
    model_output.view(resolution, resolution).detach().numpy(), circle=True
)

fig, axes = plt.subplots(2, 2, figsize=(18, 6))

axes[0][0].set_title("SIREN Radon")
axes[0][0].imshow(model_output)

axes[0][1].set_title("SIREN Inv Radon")
axes[0][1].imshow(model_output_orig.detach().numpy())

axes[1][0].set_title("Ground Truth Radon")
axes[1][0].imshow(ground_truth_radon)

axes[1][1].set_title("Ground Truth Inverse Radon")
axes[1][1].imshow(iradon(ground_truth_radon, circle=False))

snr = SignalNoiseRatio()
print(snr(model_output_orig, torch.from_numpy(iradon(ground_truth_radon, circle=True))))

plt.show()
