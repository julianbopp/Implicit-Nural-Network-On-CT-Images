import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from skimage.transform import radon
from torch.utils.data import DataLoader
from DatasetClasses.utils import SNR

from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.spline import SplineNetwork
from RadonTransform.radon_transform import radon_transform

CUDA = torch.cuda.is_available()
resolution = 182
spline = SplineNetwork(resolution)
spline.load_state_dict(torch.load("../spline_network_batch.pt"))
spline.eval()

dataset = LodopabImage(resolution,pad=False)
dataloader = DataLoader(dataset, batch_size=dataset.__len__())

model_input, ground_truth = next(iter(dataloader))

ground_truth = ground_truth.view(1, resolution, -1)
ground_truth_image = ground_truth.reshape(resolution, resolution).detach().numpy()
ground_truth_radon = radon(ground_truth_image, np.arange(180), circle=False)
ground_truth = torch.from_numpy(ground_truth_radon).unsqueeze(0)

model_output, coords = spline(model_input)
model_output_orig = model_output.view(resolution, resolution)
model_output = radon_transform(model_output.view(1, resolution, resolution), 180)

fig, axes = plt.subplots(2, 2, figsize=(18, 6))

axes[0][0].set_title("Spline Radon")
axes[0][0].imshow(model_output.cpu().view(-1, 180).detach().numpy())

axes[0][1].set_title("Spline output")
axes[0][1].imshow(model_output_orig.detach().numpy())

axes[1][0].set_title("Ground Truth Radon")
axes[1][0].imshow(ground_truth_radon)

axes[1][1].set_title("Ground Truth Image")
axes[1][1].imshow(ground_truth_image)


snr = SNR
print(snr((model_output_orig.detach()/model_output_orig.max()).detach().numpy(), (ground_truth_image/ground_truth_image.max())))

plt.show()
