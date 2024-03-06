import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworks.spline import SplineNetwork
from DatasetClasses.lodopabimage import LodopabImage

RESOLUTION = 32

image_util = LodopabImage(pad=False)
test_image = image_util.image

PADDED_RESOLUTION = image_util.padded_resolution

spline_network = SplineNetwork(RESOLUTION)

def train_one_epoch(epoch_index):
