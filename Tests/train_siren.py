import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import radon, iradon
from torch.utils.data import DataLoader
from torchmetrics.audio import SignalNoiseRatio

from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.siren import Siren
from RadonTransform.radon_transform import radon_transform

CUDA = True
resolution = 256
img_siren = Siren(
    in_features=2,
    out_features=1,
    hidden_features=resolution,
    hidden_layers=3,
    outermost_linear=True,
)

if CUDA:
    img_siren.cuda()


total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 50

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

dataset = LodopabImage(resolution)
dataloader = DataLoader(dataset, batch_size=dataset.__len__())

print(dataset.get_2d_np().shape)

model_input, ground_truth = next(iter(dataloader))
ground_truth = ground_truth.view(1, resolution, -1)
ground_truth_image = ground_truth.reshape(resolution, resolution).detach().numpy()
ground_truth_radon = radon(ground_truth_image, np.arange(180), circle=False)
ground_truth = torch.from_numpy(ground_truth_radon).unsqueeze(0)

plt.imshow(iradon(ground_truth_radon, circle=False))
plt.show()

if CUDA:
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()


def train():
    for step in range(total_steps):
        print(f"step: {step}")
        model_output, coords = img_siren(model_input)
        model_output = radon_transform(
            model_output.view(1, resolution, resolution), 180
        )

        if CUDA:
            model_output = model_output.cuda()

        loss = ((model_output - ground_truth) ** 2).mean()

        if not (step + 1) % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(-1, 180).detach().numpy())
            axes[1].imshow(ground_truth.cpu().view(-1, 180).detach().numpy())
            plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()

    snr = SignalNoiseRatio().cuda()
    print(snr(model_output, ground_truth))

    fig, axes = plt.subplots(2, 2, figsize=(18, 6))

    axes[0][0].set_title("SIREN Radon")
    axes[0][0].imshow(model_output.cpu().view(-1, 180).detach().numpy())

    axes[0][1].set_title("SIREN Inv Radon")
    axes[0][1].imshow(
        iradon(model_output.cpu().view(-1, 180).detach().numpy(), circle=False)
    )

    axes[1][0].set_title("Ground Truth Radon")
    axes[1][0].imshow(ground_truth_radon)

    axes[1][1].set_title("Ground Truth Inverse Radon")
    axes[1][1].imshow(iradon(ground_truth_radon, circle=False))
    plt.show()


train()
torch.save(img_siren.state_dict(), "../img_siren.pt")
