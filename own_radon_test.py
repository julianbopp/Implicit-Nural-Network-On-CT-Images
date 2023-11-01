import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import PIL
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms
from torchmetrics.audio import SignalNoiseRatio
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon


from radon_transform import radon_transform



ground_truth = h5py.File('dataset/ground_truth_train/ground_truth_train_000.hdf5','r')
observation_test = h5py.File('dataset/observation_train/observation_train_000.hdf5','r')

# read image from hdf5 file and store in numpy array
with observation_test as f:
    group_key = list(f.keys())[0]
    data = list(f[group_key])
    ds_arr_test = f[group_key][()]
    print(f.keys())
with ground_truth as f:
    group_key = list(f.keys())[0]
    data = list(f[group_key])
    ds_arr = f[group_key][()]
#image = np.load(ds_arr).astype(np.float32)

print("shape ground_truth:")
print(ds_arr.shape)
print("shape observation:")
print(ds_arr_test.shape)
image = rescale(ds_arr[0,:,:], scale=0.4, mode='reflect', channel_axis=None)
observation_image = rescale(ds_arr_test[0,:,:], scale=0.4, mode='reflect', channel_axis=None)

theta = range(180)
ground_truth = radon(image)
print(ground_truth.shape)

#transform = transforms.Compose([transforms.ToImageTensor(), transforms.ConvertImageDtype()])
image = Image.fromarray(image)
image = transforms.PILToTensor()(image)
print(type(image))
print(image.shape)

alpha = 180
sinogram = radon_transform(image, alpha)
print(sinogram.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
ax1.imshow(sinogram.view(-1,alpha))
ax2.imshow(ground_truth)
snr = SignalNoiseRatio()

plt.show()
print(snr(sinogram, ToTensor()(ground_truth)))
