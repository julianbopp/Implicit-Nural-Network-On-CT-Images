import torch
from DatasetClasses.lodopabimage import LodopabImage

from NeuralNetworks.siren import Siren
from NeuralNetworks.spline import SplineNetwork

from DatasetClasses.utils import SNR
from DatasetClasses.utils import addGaussianNoise

RES = 362
device = "cpu"

imageHelper = LodopabImage(RES, pad=False)
ground_truth_image = imageHelper.image
ground_truth_pixels = imageHelper.pixels

model_input = imageHelper.coords
model_input_cuda = (imageHelper.coords).to("cuda")


training_steps = 300

noise_level = [0, 0.0001, 0.001, 0.01, 0.05, 0.1]
noise_level = [0.05, 0.1]
j = 4
for noise in noise_level:
    siren_model = Siren(in_features=2, out_features=1, hidden_features=RES, hidden_layers=3, outermost_linear=True)
    siren_model = siren_model.to("cuda")

    spline_model = SplineNetwork(RES)

    optim_siren = torch.optim.Adam(lr=1e-3, params=siren_model.parameters())
    optim_spline = torch.optim.Adam(lr=1e-1, params=spline_model.parameters())
    ground_truth_noise = addGaussianNoise(ground_truth_pixels, 0, noise)
    for step in range(training_steps):

        siren_output, _ = siren_model(model_input_cuda)
        siren_output = siren_output.to(device)
        spline_output, _ = spline_model(model_input)

        loss_siren = ((siren_output - ground_truth_noise) ** 2).mean()
        loss_spline = ((spline_output - ground_truth_noise.view(-1)) ** 2).mean()


        loss_siren.backward()
        loss_spline.backward()

        optim_siren.step()
        optim_spline.step()

        optim_siren.zero_grad()
        optim_spline.zero_grad()

        if step % 10 == 0:
            print(f"training step: {step}")
            snr_siren = SNR(ground_truth_image, siren_output.view(RES,RES).detach().numpy())
            snr_spline = SNR(ground_truth_image, spline_output.view(RES,RES).detach().numpy())
            print(f"snr_siren: {snr_siren}")
            print(f"snr_spline: {snr_spline}")

            torch.save(siren_output, f"Output/siren_output_noise{j}_step{step}.pt")
            torch.save(spline_output, f"Output/spline_output_noise{j}_step{step}.pt")

    j = j+1

print("----")
print("DONE")
print("----")

