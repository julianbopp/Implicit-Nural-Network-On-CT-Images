import torch
from torchvision.transforms import v2
import typing
import skimage
import numpy as np
import math

def radon_transform(image: torch.Tensor, theta: int):
    # Transform image to Grayscale with single channel
    image = v2.Grayscale(1)(image)
    height, width = image.shape

    # Rotate image once to be able to save maximal image size (due to padding with 0's)
    rotation = v2.Compose([
        v2.RandomRotation((45,45), expand=True)
    ])

    pad_size = rotation(image).shape[1] - height

    image = v2.Pad(pad_size)(image)
    
    sinogram = np.zeros([pad_size + height, theta])
    print(sinogram.shape)
    for i in range(theta):
        rotated_image = v2.RandomRotation((i,i), expand=True)(image)
        sinogram[i,:] = torch.sum(rotated_image)

    return sinogram



