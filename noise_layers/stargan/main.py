import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from noise_layers.stargan.model import Generator
from torchvision.utils import save_image
from guided_diffusion.image_datasets_ori import load_test_data


class StarGAN(nn.Module):

    def __init__(
        self,
        c_trg=3,
        image_size=128,
    ):
        super(StarGAN, self).__init__()
        self.c_trg = c_trg
        self.image_size = image_size
        self.attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
        self.G = Generator(64, 5, 6)
        G_path = os.path.join(
            "noise_layers/stargan", str(self.image_size), "200000-G.ckpt"
        )
        self.G.load_state_dict(torch.load(G_path))
        self.G = self.G.cuda()

    def forward(self, image_cover_mask):
      with torch.no_grad():
        image = image_cover_mask[0]
        mask = torch.eye(5, device=image.device)[torch.randint(0, 5, (image.shape[0],))]
        noised_image = self.G(image, mask)
        return noised_image

