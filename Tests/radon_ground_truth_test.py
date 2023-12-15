from matplotlib import pyplot as plt
from skimage.transform import radon
from torchmetrics.audio import SignalNoiseRatio
from torchvision.transforms import transforms

from DatasetClasses.lodopabimage import LodopabImage
from RadonTransform.radon_transform import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
RES = 361
LodopabSet = LodopabImage(RES)
LodopabImage = LodopabSet.image  # Shape: [1, RES, RES]
LodopabImage = LodopabImage.unsqueeze(0)  # Shape: [1, 1, RES, RES]
print(LodopabImage.shape)
scikit_radon = radon(LodopabImage.squeeze().cpu(), circle=True)
print("lol")
print(scikit_radon.shape)

RES = LodopabSet.padded_resolution
x = torch.linspace(-1, 1, steps=RES, device=device)
y = torch.linspace(-1, 1, steps=RES, device=device)

xx, yy = torch.meshgrid(x, y, indexing="xy")
grid = torch.stack((xx, yy)).permute(1, 2, 0).unsqueeze(0)

sampled_image, _ = LodopabSet.sample_image(grid)
plt.imshow(sampled_image.cpu().squeeze().detach().numpy())
plt.show()
plt.imshow(LodopabImage.cpu().squeeze().detach().numpy())
plt.show()

sampled_image = sampled_image.squeeze()
LodopabImage = LodopabImage.squeeze()


snr = SignalNoiseRatio()

print(snr(LodopabImage.cpu(), sampled_image.cpu()))

plt.imshow(scikit_radon, aspect="auto")
plt.show()
batch_radon = LodopabSet.get_radon_transform()

# batch_radon = v2.functional.rotate(batch_radon.squeeze().unsqueeze(0), angle=-90)
# batch_radon = v2.functional.vertical_flip(batch_radon.squeeze().unsqueeze(0))


plt.imshow(batch_radon.squeeze().cpu().detach().numpy(), aspect="auto")
plt.show()
normalize = transforms.Normalize(torch.Tensor([0]), torch.Tensor([1]))
scikit_radon_torch = normalize(torch.from_numpy(scikit_radon).unsqueeze(0)).squeeze()

print(LodopabImage.shape)
print("ligma")
print(scikit_radon_torch.shape)
print(batch_radon.shape)
print(snr(scikit_radon_torch.cpu() / 511, batch_radon.squeeze().cpu()))

print(snr)
