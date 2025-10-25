import os
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
import random, string
from guided_diffusion import dist_util

# import ID Embedder
from noise_layers.cscs.model.arcface.iresnet import iresnet100

# import ID Embedder Adapter
from noise_layers.cscs.model.arcface.iresnet_adapter import iresnet100_adapter

# import Generator
from noise_layers.cscs.model.faceshifter.layers.faceshifter.layers_arcface import (
    AEI_Net,
)


class CSCS(nn.Module):
    def __init__(self, temp="temp/"):
        super(CSCS, self).__init__()
        self.temp = temp
        if not os.path.exists(temp):
            os.mkdir(temp)
        device = dist_util.dev()
        model_weight = torch.load(
            "noise_layers/cscs/model_34_loss_-0.1688.pth.tar", map_location="cpu"
        )
        self.adapter_type = "add"
        # load ID Embedder
        self.ID_emb = iresnet100()
        self.ID_emb.load_state_dict(
            torch.load(
                "noise_layers/cscs/model/arcface/checkpoint/backbone.pth",
                map_location="cpu",
            )
        )

        # load ID adapter
        self.ID_adapter = iresnet100_adapter(type=self.adapter_type)
        self.ID_adapter.load_state_dict(model_weight["adapter"])

        # build Generator
        self.G = AEI_Net(512)
        self.G.load_state_dict(model_weight["G"])

        self.ID_emb = self.ID_emb.to(device)
        self.G = self.G.to(device)
        self.ID_adapter = self.ID_adapter.to(device)

        self.ID_emb.eval()
        self.G.eval()
        self.ID_adapter.eval()

    def get_temp(self):
        return (
            self.temp
            + "".join(random.sample(string.ascii_letters + string.digits, 16))
            + ".png"
        )

    def forward(self, image_cover_mask):
      with torch.no_grad():
        image = image_cover_mask[0]
        cover_image = image_cover_mask[1]

        img_b = image
        img_a = torch.roll(cover_image, shifts=1, dims=0)

        src_id = F.normalize(
            self.ID_emb(F.interpolate(img_a, size=112, mode="bilinear")),
            dim=-1,
            p=2,
        )
        src_id_adapt = F.normalize(
            self.ID_adapter(F.interpolate(img_a, size=112, mode="bilinear")),
            dim=-1,
            p=2,
        )
        if img_b.shape[-2:] != (256, 256):
            img_b = F.interpolate(
                img_b, size=(256, 256), mode="bilinear", align_corners=False
            )

        if self.adapter_type == "concat":
            src_id = torch.cat([src_id, src_id_adapt], dim=1)
        elif self.adapter_type == "add":
            src_id = src_id + src_id_adapt
        elif self.adapter_type == "replace":
            src_ID_emb_input = src_id_adapt
        swapped, attr, m = self.G(img_b, src_id)
        if (
            swapped.shape[-2:] != image.shape[-2:]
        ):
            swapped = F.interpolate(
                swapped, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return swapped


