import torch
from skimage.transform import radon
from torch.utils.data import DataLoader

from DatasetClasses.lodopabimage import LodopabImage

RESOLUTION = 256
lodopabImage = LodopabImage(RESOLUTION)
lodopabLoader = DataLoader(lodopabImage, batch_size=lodopabImage.__len__())

# Get ground truth radon image
_, ground_truth = next(iter(lodopabLoader))
ground_truth_image = ground_truth.reshape(RESOLUTION,RESOLUTION).detach().numpy()
ground_truth_radon = radon(ground_truth_image, circle=True)
ground_truth = torch.from_numpy(ground_truth_radon)
