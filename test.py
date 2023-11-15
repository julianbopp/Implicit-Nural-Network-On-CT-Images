from siren import Siren
import torch
from lodopabimage import LodopabImage
resolution = 256
img_siren = Siren(in_features=2, out_features=1, hidden_features=resolution,
                  hidden_layers=3, outermost_linear=True)
dataset = LodopabImage(resolution)
img_siren.load_state_dict(torch.load('img_siren.pt', map_location=torch.device('cpu')))
img_siren.eval()

grid = dataset.get_mgrid(2)
input = torch.zeros(1,2)
input[0][0] = -2
input[0][1] = -2
print(input)
print(img_siren(input))
