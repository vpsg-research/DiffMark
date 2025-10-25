import os
import sys
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from noise_layers.simswap.models.models import create_model
from noise_layers.simswap.test_options import TestOptions
from guided_diffusion import dist_util, logger
import torch.nn as nn


class SimSwap(nn.Module):
    def __init__(self):
        super(SimSwap, self).__init__()

        self.transformer_Arcface = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        opt = TestOptions()
        self.model = create_model(opt)

    def forward(self, image_cover_mask):
      # print("SimSwap")
      with torch.no_grad():
        image = image_cover_mask[0]
        cover_image = image_cover_mask[1]
        img_b = image
        img_a = torch.roll(cover_image, shifts=1, dims=0)
        img_a = (img_a + 1) / 2
        img_id = self.transformer_Arcface(img_a)
        img_att = (img_b + 1) / 2
        img_id_downsample = F.interpolate(img_id, size=(112, 112), mode="bilinear")
        latend_id = self.model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to("cpu")
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to(img_b.device)
        img_fake = self.model(img_id, img_att, latend_id, latend_id, True)
        img_fake = img_fake * 2 - 1
        return img_fake


