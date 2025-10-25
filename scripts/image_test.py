import os
import sys

# sys.path.append("/home/xx/DiffMark-main")
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    create_endecoder_and_diffusion,
    endecoder_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
import kornia.losses
import lpips as clpips
import yaml
from easydict import EasyDict
from guided_diffusion.noiser import Random_Noise
from utils.util import get_cover_image
from utils.util import save_vis

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    logger.log("args...")
    logger.log(args)
    logger.log("creating criterion...")
    device = dist_util.dev()
    criterion_LPIPS = clpips.LPIPS(net="vgg").to(device).eval()
    logger.log("creating model and diffusion...")
    model, diffusion = create_endecoder_and_diffusion(
        **args_to_dict(args, endecoder_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    noise_layer = Random_Noise(args.noise_layer) if args.noise_layer else None
    guide_layer = Random_Noise(args.guide_layer) if args.guide_layer else None
    def cond_fn(x, n_t, t, message):
        with th.enable_grad():
            x_t = x.detach().requires_grad_(True)
            cover_image = get_cover_image(args.cover_dir, x_t.shape[0], x_t.shape[2])
            pred_xstart, extract_message = model(
                x_t, n_t, t, message, noise_layer=guide_layer, cover_image=cover_image, **model_kwargs
            )
            selected = diffusion.message_guide(extract_message, message)
            score = th.autograd.grad(selected, x_t)[0] * args.guide_scale
        return score, pred_xstart

    logger.log("creating data loader...")
    data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
    )
    logger.log("testing...")
    test_result = {
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "niqe": 0.0,
        "piqe": 0.0,
        "ber": 0.0,
    }
    num = len(data_loader)
    for i, images in enumerate(data_loader):
        with th.no_grad():
            images = images.to(device)
            message = th.Tensor(
                np.random.choice([0, 1], (images.shape[0], args.message_length))
            ).to(device)
            model_kwargs = {
                # "noise_layer": noise_layer,
                # "cover_image": images,
            }
            sample_fn = (
                diffusion.p_sample_loop
                if not args.use_ddim
                else diffusion.ddim_sample_loop
            )
            if args.use_guidance:
                sample = sample_fn(
                    model,
                    images,
                    message,
                    clip_denoised=args.clip_denoised,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            else:
                sample = sample_fn(
                    model,
                    images,
                    message,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            encoded_images = sample.clone().detach()
            noised_images = noise_layer([sample, images])
            extract_message = model.decoder(noised_images)

            psnr = -kornia.losses.psnr_loss(encoded_images, images, 2)
            ssim = 1 - 2 * kornia.losses.ssim_loss(
                encoded_images, images, window_size=11, reduction="mean"
            )
            lpips = th.mean(criterion_LPIPS(encoded_images, images)).item()
            ber = diffusion.message_ber(extract_message, message)
            result = {
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "ber": ber,
            }
            logger.log(f"result : {result}")
            for key in result:
                test_result[key] += float(result[key])
            logger.log(f"tested {i+1}/{num} steps")

            # Visualization
            # vis_dir = os.path.join("visualization", *args.noise_layer)
            # save_vis(images, encoded_images, noised_images, vis_dir, rand=True)


    for key in test_result:
        test_result[key] = test_result[key] / num
    logger.log(f"average: {test_result}")
    dist.barrier()
    logger.log("testing complete")


def create_argparser():
    with open("configs/test.yaml", "r") as f:
        test_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    noise_layer = test_args.noise_layer
    guide_layer = test_args.guide_layer
    defaults = endecoder_and_diffusion_defaults()
    test_defaults = dict(
        use_ddim=True,
        use_guidance=False,
        clip_denoised=False,
        batch_size=16,
        guide_scale=1000,
        model_path="ema_0.9999_151200.pt",
        # model_path="ema_0.9999_151200.pt",
        data_dir="CelebA-HQ/test_128",
        # data_dir="CelebA-HQ/test_256",
        noise_layer=noise_layer,
        guide_layer=guide_layer,
        cover_dir="CelebA-HQ",
    )
    defaults.update(test_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
