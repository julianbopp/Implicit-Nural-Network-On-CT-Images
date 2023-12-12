import math
import torch
from skimage.transform import radon
from torch.utils.data import DataLoader

from DatasetClasses.ParameterSet import AngleSet, CoordSet
from DatasetClasses.lodopabimage import LodopabImage
from NeuralNetworks.siren import Siren
from RadonTransform.radon_transform import batch_radon_siren

CUDA = torch.cuda.is_available()

RESOLUTION = 256
CIRCLE = False
img_siren = Siren(
    in_features=2,
    out_features=1,
    hidden_features=RESOLUTION,
    hidden_layers=3,
    outermost_linear=True,
)

if CUDA:
    img_siren = img_siren.cuda()
optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

lodopabImage = LodopabImage(RESOLUTION)
lodopabLoader = DataLoader(lodopabImage, batch_size=lodopabImage.__len__())

# Get ground truth radon image
_, ground_truth = next(iter(lodopabLoader))
ground_truth_image = ground_truth.reshape(RESOLUTION, RESOLUTION).detach().numpy()
ground_truth_radon = radon(ground_truth_image, circle=CIRCLE)
ground_truth = torch.from_numpy(ground_truth_radon)

angleSet = AngleSet(180, rad=False)
if CUDA:
    angleSet.angles = angleSet.angles.cuda()
angleLoader = DataLoader(angleSet, batch_size=15, shuffle=True)

coordSet = CoordSet(int(RESOLUTION * math.sqrt(2)), circle=CIRCLE)
if CUDA:
    coordSet.coords = coordSet.coords.cuda()
coordLoader = DataLoader(coordSet, batch_size=60, shuffle=True)
coordIter = iter(coordLoader)

training_steps = 100
sample_points = RESOLUTION
if CIRCLE:
    sample_points = round(RESOLUTION * math.sqrt(2))

if CUDA:
    ground_truth = ground_truth.cuda()

step = 0
loss_total = []

for step in range(training_steps):
    print(f"training step = {step}")
    for angle, angle_idx in angleLoader:
        for coords, coords_idx in coordLoader:
            optim.zero_grad()
            radon_output = batch_radon_siren(
                coords, img_siren, sample_points, theta=angle, CUDA=CUDA
            )
            # Reshape coordIdx and angleIdx
            coordIdx_unsq = coords_idx.unsqueeze(1)  # shape [256, 1]
            angleIdx_unsq = angle_idx.unsqueeze(0)  # shape [1, 10]

            # Create a grid of indices
            coordIdx_grid, angleIdx_grid = torch.meshgrid(
                coordIdx_unsq[:, 0], angleIdx_unsq[0, :], indexing="ij"
            )

            loss = (
                (radon_output - ground_truth[coordIdx_grid, angleIdx_grid]) ** 2
            ).mean()
            loss_total.append(loss.item())

            loss.backward()
            optim.step()

    print(torch.tensor(loss_total[-180 * coordSet.__len__() :]).mean().item())

torch.save(img_siren.state_dict(), "../img_batch_siren.pt")
