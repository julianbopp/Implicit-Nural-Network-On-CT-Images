import torch
import matplotlib.pyplot as plt
import numpy as np
from DatasetClasses.lodopabimage import LodopabImage
from skimage.transform import radon, iradon
from DatasetClasses.utils import SNR
RES = 362
imageHelper = LodopabImage(RES, base="../", pad=False)
ground_truth_radon = imageHelper.get_radon_transform()

plt.imshow(imageHelper.image[0,:,:])
plt.show()

inverse_radon = iradon(ground_truth_radon.detach().numpy(), circle=False)
plt.imshow(inverse_radon)
plt.show()

imageHelper = LodopabImage(inverse_radon.shape[0], base="../", pad=False)
ground_truth_image = imageHelper.image
snr = SNR(ground_truth_image, inverse_radon)
print(snr)