import numpy as np
import matplotlib.pyplot as plt
import h5py

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon

#image = shepp_logan_phantom()

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

print("shape grond_truth:")
print(ds_arr.shape)
print("shape observation:")
print(ds_arr_test.shape)
image = rescale(ds_arr[0,:,:], scale=0.4, mode='reflect', channel_axis=None)
observation_image = rescale(ds_arr_test[0,:,:], scale=0.4, mode='reflect', channel_axis=None)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5))
ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)
print(sinogram.shape)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')

ax3.imshow(observation_image, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')
ax3.set_title("observation")
fig.tight_layout()



reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
error = reconstruction_fbp - image
print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction error\nFiltered back projection")
ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()