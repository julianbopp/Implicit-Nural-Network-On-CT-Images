import torch
import matplotlib.pyplot as plt

image = torch.load("../spline_image.pt")
torch.save(image, "test.pt")
image = image["weights"]
plt.imshow(image.view(64, 64))
plt.show()
