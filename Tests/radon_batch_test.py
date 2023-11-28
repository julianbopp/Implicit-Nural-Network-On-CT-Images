import torch
from skimage.transform import radon
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import NeuralNetworks.load_siren
from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from RadonTransform.radon_transform import batch_radon

RESOLUTION = 256
CUDA = True

img_siren = NeuralNetworks.load_siren.img_siren
ground_truth = NeuralNetworks.load_siren.ground_truth
radon_output = NeuralNetworks.load_siren.model_output

L = 362
angleSet = AngleSet(180)
coordSet = CoordSet(L, circle=False)

angleLoader = DataLoader(angleSet, batch_size=10)
coordLoader = DataLoader(coordSet, batch_size=10)

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

        batch_radon_output[coordIdx_grid, angleIdx_grid] = batch_radon(coords,img_siren,100,angles, CUDA=CUDA).detach()

plt.imshow(batch_radon_output.cpu().detach().numpy())
plt.show()
