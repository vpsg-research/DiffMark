import sys
import os
import torch
from torch import nn
import numpy as np
from noise_layers.uniface.training.model import (
    Generator_globalatt_return_32 as Generator,
)
from noise_layers.uniface.training.model import Encoder_return_32 as Encoder
import torch.nn.functional as F
import random

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

cmap = np.array(
    [
        (0, 0, 0),
        (255, 0, 0),
        (76, 153, 0),
        (204, 204, 0),
        (51, 51, 255),
        (204, 0, 204),
        (0, 255, 255),
        (51, 255, 255),
        (102, 51, 0),
        (255, 0, 0),
        (102, 204, 0),
        (255, 255, 0),
        (0, 0, 153),
        (0, 0, 204),
        (255, 51, 153),
        (0, 204, 204),
        (0, 51, 0),
        (255, 153, 51),
        (0, 204, 0),
    ],
    dtype=np.uint8,
)


def normalize_to_minus_one_one(tensor):
    min_val = tensor.min()
    max_val = tensor.max()

    normalized_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1

    return normalized_tensor


def normalize_to_zero_one(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = cmap
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = label == gray_image[0]
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class UniFaceSwap(nn.Module):
    def __init__(
        self,
        ckpt_path="noise_layers/uniface/checkpoints/swap/500000.pt",
        save_image_dir="noise_layers/uniface/expr/swap",
        device="cuda",
        size=256,
        latent_channel_size=512,
        latent_spatial_size=8,
        lr_mul=1.0,
        channel_multiplier=2,
        normalize_mode="LayerNorm",
        small_generator=False,
    ):
        super(UniFaceSwap, self).__init__()

        self.device = device
        self.size = size
        self.latent_channel_size = latent_channel_size
        self.latent_spatial_size = latent_spatial_size
        self.lr_mul = lr_mul
        self.channel_multiplier = channel_multiplier
        self.normalize_mode = normalize_mode
        self.small_generator = small_generator

        self.g_ema = Generator(
            self.size,
            self.latent_channel_size,
            self.latent_spatial_size,
            lr_mul=self.lr_mul,
            channel_multiplier=self.channel_multiplier,
            normalize_mode=self.normalize_mode,
            small_generator=self.small_generator,
        ).to(self.device)

        self.e_ema = Encoder(
            self.size,
            self.latent_channel_size,
            self.latent_spatial_size,
            channel_multiplier=self.channel_multiplier,
        ).to(self.device)

        self.load_checkpoint(ckpt_path)

        self.g_ema.eval()
        self.e_ema.eval()

        self.save_image_fake_dir = os.path.join(save_image_dir, "fake")
        os.makedirs(self.save_image_fake_dir, exist_ok=True)
        self.save_image_pair_dir = os.path.join(save_image_dir, "pair")
        os.makedirs(self.save_image_pair_dir, exist_ok=True)

    def load_checkpoint(self, ckpt_path):

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.g_ema.load_state_dict(ckpt["g_ema"])
        self.e_ema.load_state_dict(ckpt["e_ema"])
        print(f"Loaded checkpoint from {ckpt_path}")

    def tensor2label(self, label_tensor, n_label=19):
  
        label_tensor = label_tensor.float()
        if label_tensor.size()[0] > 1:
            label_tensor = label_tensor.max(0, keepdim=True)[1]
        label_tensor = Colorize(n_label)(label_tensor)
        label_numpy = label_tensor.numpy()

        return label_numpy

    def forward(self, image_cover_mask):

        image = image_cover_mask[0]
        cover_image = image_cover_mask[1]
        trg_imgs = image
        src_imgs = torch.roll(cover_image, shifts=1, dims=0)

        with torch.no_grad():
  
            if trg_imgs.shape[-2:] != (256, 256): 
                trg_imgs = F.interpolate(
                    trg_imgs, size=(256, 256), mode="bilinear", align_corners=False
                )
            if src_imgs.shape[-2:] != (256, 256):
                src_imgs = F.interpolate(
                    src_imgs, size=(256, 256), mode="bilinear", align_corners=False
                )
            trg_imgs = trg_imgs.to(self.device)
            src_imgs = src_imgs.to(self.device)

            trg, src, fake_imgs = self._forward([trg_imgs, src_imgs])
        if (
            fake_imgs.shape[-2:] != image.shape[-2:]
        ):
            fake_imgs = F.interpolate(
                fake_imgs, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return fake_imgs

    def _forward(self, input):

        trg = input[0]
        src = input[1]
        with torch.no_grad():
            trg_src = torch.cat([trg, src], dim=0)
            w, w_feat = self.e_ema(trg_src)
            w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]

            trg_w, src_w = torch.chunk(w, 2, dim=0)

            fake_imgs = self.g_ema([trg_w, src_w, w_feat_tgt])

        return trg, src, fake_imgs


