import os
import torch
import torch.nn as nn
import random
import numpy as np
import kornia
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random, string


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, image_cover):
        print("Identity")
        image = image_cover[0]
        return image


class Resize(nn.Module):

    def __init__(self, down_scale=0.8):
        super(Resize, self).__init__()
        self.down_scale = down_scale

    def forward(self, image_cover):
        print("Resize")
        image = image_cover[0]
        noised_down = F.interpolate(
            image,
            size=(
                int(self.down_scale * image.shape[2]),
                int(self.down_scale * image.shape[3]),
            ),
            mode="nearest",
        )
        noised_up = F.interpolate(
            noised_down, size=(image.shape[2], image.shape[3]), mode="nearest"
        )
        return noised_up


class Dropout(nn.Module):
    def __init__(self, prob=0.6):
        super(Dropout, self).__init__()
        self.prob = prob

    def forward(self, image_cover):
        print("Dropout")
        image, cover_image = (
            image_cover[0],
            image_cover[1],
        )
        dropout_mask = torch.bernoulli(
            torch.full(
                image.shape[2:], 1 - self.prob, dtype=torch.float32, device=image.device
            )
        )
        dropout_mask = dropout_mask.expand_as(image)

        output = image * dropout_mask + cover_image * (1 - dropout_mask)
        return output


class GaussianNoise(nn.Module):

    def __init__(self, mean=0, std=0.1, p=1):
        super(GaussianNoise, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianNoise(
            mean=mean, std=std, p=p
        )

    def forward(self, image_cover):
        print("GaussianNoise")
        image = image_cover[0]
        return self.transform(image)


class SaltPepper(nn.Module):

    def __init__(self, prob=0.1):
        super(SaltPepper, self).__init__()
        self.prob = prob

    def sp_noise(self, image, prob):
        mask = torch.Tensor(
            np.random.choice(
                (0, 1, 2), image.shape[2:], p=[1 - prob, prob / 2.0, prob / 2.0]
            )
        ).to(image.device)
        mask = mask.expand_as(image)

        image[mask == 1] = 1  # salt
        image[mask == 2] = -1  # pepper

        return image

    def forward(self, image_cover):
        print("SaltPepper")
        image = image_cover[0]
        output = image.clone().detach()
        return self.sp_noise(
            output, self.prob
        )  # image * mask + self.sp_noise(image, self.prob) * (1 - mask)


class GaussianBlur(nn.Module):

    def __init__(self, kernel_size=(5, 5), sigma=(5, 5), p=1):
        super(GaussianBlur, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianBlur(
            kernel_size=kernel_size, sigma=sigma, p=p
        )

    def forward(self, image_cover):
        print("GaussianBlur")
        image = image_cover[0]
        return self.transform(image)


class MedianBlur(nn.Module):

    def __init__(self, kernel_size=(5, 5)):
        super(MedianBlur, self).__init__()
        self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

    def forward(self, image_cover):
        print("MedianBlur")
        image = image_cover[0]
        return self.transform(image)


class Brightness(nn.Module):

    def __init__(self, brightness=0.5, p=1):
        super(Brightness, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(brightness=brightness, p=p)

    def forward(self, image_cover):
        print("Brightness")
        image = image_cover[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Contrast(nn.Module):

    def __init__(self, contrast=0.5, p=1):
        super(Contrast, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(contrast=contrast, p=p)

    def forward(self, image_cover):
        print("Contrast")
        image = image_cover[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Saturation(nn.Module):

    def __init__(self, saturation=0.5, p=1):
        super(Saturation, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(saturation=saturation, p=p)

    def forward(self, image_cover):
        print("Saturation")
        image = image_cover[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class Hue(nn.Module):

    def __init__(self, hue=0.1, p=1):
        super(Hue, self).__init__()
        self.transform = kornia.augmentation.ColorJitter(hue=hue, p=p)

    def forward(self, image_cover):
        print("Hue")
        image = image_cover[0]
        out = (image + 1) / 2
        colorjitter = self.transform(out)
        colorjitter = (colorjitter * 2) - 1
        return colorjitter


class JpegTest(nn.Module):
    def __init__(self, Q=50, subsample=0, path="temp/"):
        super(JpegTest, self).__init__()
        self.Q = Q
        self.subsample = subsample
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def get_path(self):
        return (
            self.path
            + "".join(random.sample(string.ascii_letters + string.digits, 16))
            + ".jpg"
        )

    def forward(self, image_cover):
        print("JpegTest")
        image = image_cover[0]

        noised_image = torch.zeros_like(image)

        for i in range(image.shape[0]):
            single_image = (
                ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255)
                .add(0.5)
                .clamp(0, 255)
                .to("cpu", torch.uint8)
                .numpy()
            )
            im = Image.fromarray(single_image)

            file = self.get_path()
            while os.path.exists(file):
                file = self.get_path()
            im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)
            jpeg = np.array(Image.open(file), dtype=np.uint8)
            os.remove(file)

            noised_image[i] = self.transform(jpeg).unsqueeze(0).to(image.device)

        return noised_image
