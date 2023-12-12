from matplotlib import pyplot as plt
from skimage.transform import radon
from torchmetrics.audio import SignalNoiseRatio

from DatasetClasses.lodopabimage import LodopabImage
from RadonTransform.radon_transform import *

RES = 361
LodopabSet = LodopabImage(RES)
LodopabImage = LodopabSet.image  # Shape: [1, RES, RES]
LodopabImage = LodopabImage.unsqueeze(0)  # Shape: [1, 1, RES, RES]
print(LodopabImage.shape)
scikit_radon = radon(LodopabImage.squeeze(), circle=True)
RES = LodopabSet.padded_resolution
x = torch.linspace(-1, 1, steps=RES)
y = torch.linspace(-1, 1, steps=RES)

xx, yy = torch.meshgrid(x, y, indexing="xy")
grid = torch.stack((xx, yy)).permute(1, 2, 0).unsqueeze(0)

sampled_image, _ = LodopabSet.sample_image(grid)
plt.imshow(sampled_image.squeeze().detach().numpy())
plt.show()
plt.imshow(LodopabImage.squeeze())
plt.show()

sampled_image = sampled_image.squeeze()
LodopabImage = LodopabImage.squeeze()


snr = SignalNoiseRatio()

print(snr(LodopabImage, sampled_image))

plt.imshow(scikit_radon, aspect="auto")
plt.show()
batch_radon = LodopabSet.get_radon_transform()
plt.imshow(batch_radon.squeeze(), aspect="auto")
plt.show()

scikit_radon_torch = torch.from_numpy(scikit_radon)

print(LodopabImage.shape)
radon_transform = radon_transform(LodopabImage.unsqueeze(0))
print(scikit_radon_torch.shape)
print(batch_radon.shape)
print(snr(torch.from_numpy(scikit_radon), batch_radon.squeeze()))
