import os
import time
import random
import torch as th
from PIL import Image
from torchvision import transforms
from glob import glob
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import torchvision.io as io
from torchvision.utils import save_image

def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def create_diff_img(img1, img2):
    diff = img1 - img2
    diff = (diff - diff.min()) / ((diff.max() - diff.min()) + 1e-6)
    return th.abs(diff - 0.5)


def save_vis(original_images, watermarked_image, distorted_images, vis_dir, rand=True):
    
    os.makedirs(vis_dir, exist_ok=True)
    B = original_images.size(0)
    indices = [random.randint(0, B - 1)] if rand else range(B)
    for i in indices:
        uid = f"{int(time.time())}"
        # origin
        images_01 = (original_images[i] + 1) / 2
        save_image(images_01, os.path.join(vis_dir, f"origin_{uid}.png"))

        # watermarked
        wm_01 = (watermarked_image[i] + 1) / 2
        res_co_wm = create_diff_img(watermarked_image[i], original_images[i])
        save_image(wm_01, os.path.join(vis_dir, f"watermarked_{uid}.png"))
        save_image(res_co_wm, os.path.join(vis_dir, f"res_co_wm_{uid}.png"))

        # distorted
        dt_01 = (distorted_images[i] + 1) / 2
        res_wm_dt = create_diff_img(distorted_images[i], watermarked_image[i])
        save_image(dt_01, os.path.join(vis_dir, f"distorted_{uid}.png"))
        save_image(res_wm_dt, os.path.join(vis_dir, f"res_wm_dt_{uid}.png"))



@lru_cache(maxsize=1)
def _list_image_files(data_dir):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(data_dir, ext)))
    if not files:
        raise ValueError(f"No images found in {data_dir}")
    return files


def get_cover_image(
    data_dir: str,
    n: int = 16,
    image_size: int = 128,
    device: th.device = th.device('cuda' if th.cuda.is_available() else 'cpu'),
    normalize: bool = True,
    num_workers: int = 8,
) -> th.Tensor:
    all_files = _list_image_files(os.path.join(data_dir, "train_" + str(image_size)))

    if len(all_files) < n:
        raise ValueError(f"Only {len(all_files)} images found in {data_dir}, cannot select {n} images")


    selected = random.sample(all_files, n)

    def load_tensor(path):
        img = io.read_image(path)
        return img.float().div(255.0)

    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        imgs = list(exe.map(load_tensor, selected))

    batch = th.stack(imgs, dim=0)
    if normalize:
        batch = (batch - 0.5) / 0.5            # [-1,1]

    return batch.to(device, non_blocking=True)