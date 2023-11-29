import numpy as np
import torch
from skimage.transform import radon
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics.audio import SignalNoiseRatio

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.siren import Siren
from RadonTransform.radon_transform import batch_radon, radon_transform

CUDA = torch.cuda.is_available()
resolution = 256
img_siren = Siren(in_features=2, out_features=1, hidden_features=resolution,
                  hidden_layers=3, outermost_linear=True)

if CUDA:
    img_siren.load_state_dict(torch.load('../img_batch_siren.pt'))
else:
    img_siren.load_state_dict(torch.load('../img_batch_siren.pt', map_location=torch.device('cpu')))
img_siren.eval()


L = 363
angleSet = AngleSet(180, rad=False)
coordSet = CoordSet(L, circle=False)

angleLoader = DataLoader(angleSet, batch_size=3)
coordLoader = DataLoader(coordSet, batch_size=256)

lodopab = LodopabImage(resolution)
mgrid = lodopab.get_mgrid(resolution)

model_output, _ = img_siren(mgrid)
model_output1 = radon(model_output.reshape(resolution,resolution).detach().numpy(),circle=False)
model_output2 = radon_transform(model_output.reshape(1,resolution,resolution))

batch_radon_output = torch.zeros(L, 180, requires_grad=False)
if CUDA:
    img_siren = img_siren.cuda()
    batch_radon_output = batch_radon_output.cuda()

for angles, angleIdx in angleLoader:
    print(angles)
    for coords, coordIdx in coordLoader:
        # Reshape coordIdx and angleIdx
        coordIdx_unsq = coordIdx.unsqueeze(1)  # shape [256, 1]
        angleIdx_unsq = angleIdx.unsqueeze(0)  # shape [1, 10]

        # Create a grid of indices
        coordIdx_grid, angleIdx_grid = torch.meshgrid(coordIdx_unsq[:, 0], angleIdx_unsq[0, :], indexing='ij')

        batch_radon_output[coordIdx_grid, angleIdx_grid] = batch_radon(coords,img_siren,363,angles, CUDA=CUDA).detach()

#batch_radon_output[1:,:] = batch_radon_output[0:-1,:].clone()
plt.imshow(batch_radon_output.cpu().detach().numpy())
plt.show()
plt.imshow(model_output1)
plt.show()
plt.imshow(model_output2.view(363,180).detach().numpy())
plt.show()
snr = SignalNoiseRatio()
print(snr(batch_radon_output.cpu(), torch.from_numpy(model_output1)))
print(snr(batch_radon_output.cpu(), (model_output2.view(L,180))))
print(snr(torch.from_numpy(model_output1), (model_output2.view(L,180))))
