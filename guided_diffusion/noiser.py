
import torch.nn as nn
import numpy as np
from noise_layers import *


class Random_Noise(nn.Module):

    def __init__(self, layers):
        super(Random_Noise, self).__init__()
        self.noise_layers = [eval(layer) for layer in layers]

    def forward(self, image_cover):
        image = image_cover[0]
        cover_image = image_cover[1]
        forward_image = image.clone().detach()
        forward_cover_image = cover_image.clone().detach()
        forward_image_cover = [forward_image, forward_cover_image]
        noise_layer = np.random.choice(self.noise_layers)
        noised_image = noise_layer(forward_image_cover)
        noised_image_gap = noised_image - forward_image
        return image + noised_image_gap
