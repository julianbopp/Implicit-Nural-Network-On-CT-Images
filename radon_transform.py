import torch
import math
import numpy as np

def radon_transform(image, theta):

    length, width = image.shape
    diag_length = math.sqrt(length ** 2 + width ** 2)
    lengthPad = math.ceil(diag_length - length)  + 2
    widthPad = math.ceil(diag_length - width) + 2
    image_padded = np.zeros(length + lengthPad, width + widthPad)
    image_padded[math.ceil(lengthPad / 2) : (math.ceil(lengthPad / 2) + length - 1), math.ceil(widthPad / 2) : (math.ceil(widthPad / 2) + width - 1)] = image

    n = theta.size
    transform = np.zeros(image_padded.shape[1], n)

    for i in range(n):
        tmpimg = 



