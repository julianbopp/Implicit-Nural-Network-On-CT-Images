import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.transform import radon

from NeuralNetworks.siren import Siren
from DatasetClasses.lodopabimage import LodopabImage
from RadonTransform.radon_transform import *
resolution = 256
img_siren = Siren(in_features=2, out_features=1, hidden_features=resolution,
                  hidden_layers=3, outermost_linear=True)
dataset = LodopabImage(resolution)
img_siren.load_state_dict(torch.load('../img_siren.pt', map_location=torch.device('cpu')))
img_siren.eval()

grid = dataset.get_mgrid(2)
input = torch.zeros(1,2)
input[0][0] = -2
input[0][1] = -2
print(input.shape)
print(img_siren(input))

#radon_batch_output = batch_radon(torch.linspace(-(0.5+1.414)/2,(0.5+1.414)/2,steps=256),img_siren, 30)
radon_batch_output = batch_radon(torch.linspace(-1,1,steps=256),img_siren, 30)
print(radon_batch_output)
print(radon_batch_output.shape)
plt.imshow(radon_batch_output.detach().numpy().T)
plt.show()

dataset = LodopabImage(resolution)
dataloader = DataLoader(dataset, batch_size=dataset.__len__())
model_input, ground_truth = next(iter(dataloader))

ground_truth = ground_truth.view(1,resolution,-1)
ground_truth_image = ground_truth.reshape(resolution,resolution).detach().numpy()
ground_truth_radon = radon(ground_truth_image, np.arange(180), circle=True)
ground_truth = torch.from_numpy(ground_truth_radon).unsqueeze(0)

plt.imshow(ground_truth_radon)
plt.show()