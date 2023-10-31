import torch
from torchvision.transforms import v2
import typing
import skimage
import numpy as np
import math

def radon_transform(image: torch.Tensor, theta: int):
    # Transform image to Grayscale with single channel
    image = v2.Grayscale(1)(image)

    # Rotate image once by 45 to create padding with 0's
    image = v2.RandomRotation((45, 45), expand=True)(image)
    image = v2.RandomRotation((-45,-45), expand=False)(image)

    _, height, width = image.shape
    sinogram = np.zeros([height, theta])

    # Rotate by angle and sum in one dimension
    for i in range(theta):
        rotated_image = v2.RandomRotation((i,i), expand=False)(image)
        sum = torch.sum(rotated_image, 1)
        sinogram[:, i] = sum

    return sinogram



