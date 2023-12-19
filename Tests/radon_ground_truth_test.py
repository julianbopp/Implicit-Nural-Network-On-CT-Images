from matplotlib import pyplot as plt
from skimage.transform import radon
from torchmetrics.audio import SignalNoiseRatio
from torchvision.transforms import transforms

from DatasetClasses.lodopabimage import LodopabImage
from RadonTransform.radon_transform import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
RES = 364
LodopabSet = LodopabImage(RES)
LodopabImage = LodopabSet.image  # Shape: [1, RES, RES]
LodopabImage = LodopabImage.unsqueeze(0)  # Shape: [1, 1, RES, RES]
print(LodopabImage.shape)
scikit_radon = radon(LodopabImage.squeeze().cpu(), circle=True)
print(scikit_radon.shape)

RES = LodopabSet.padded_resolution


snr = SignalNoiseRatio()

batch_radon = LodopabSet.get_radon_transform()


plt.imshow(batch_radon.squeeze().cpu().detach().numpy(), aspect="auto")
plt.show()
plt.imshow(scikit_radon, aspect="auto")
plt.show()
normalize = transforms.Normalize(torch.Tensor([0]), torch.Tensor([1]))
scikit_radon_torch = normalize(torch.from_numpy(scikit_radon).unsqueeze(0)).squeeze()

print(LodopabImage.shape)
print(scikit_radon_torch.shape)
print(batch_radon.shape)
print(
    snr(
        scikit_radon_torch.cpu() / scikit_radon_torch.max(),
        batch_radon.squeeze().cpu() / batch_radon.cpu().max(),
    )
)

print(snr)
