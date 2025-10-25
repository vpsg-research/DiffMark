import os
import sys
# sys.path.append("/home/xx/DiffMark-main")

import argparse
import yaml
from easydict import EasyDict
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_inf
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.script_util import (
    create_endecoder_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    endecoder_and_diffusion_defaults,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log("args...")
    logger.log(args)
    logger.log("creating model and diffusion ...")
    model, diffusion = create_endecoder_and_diffusion(
        **args_to_dict(args, endecoder_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data_inf(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        message_length=args.message_length,
        noise_layers=args.noise_layers,
        threshold=args.threshold,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    with open("configs/train.yaml", "r") as f:
        train_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    noise_layers = train_args.noise_layers
    defaults = endecoder_and_diffusion_defaults()
    train_defaults = dict(
        data_dir="CelebA-HQ/train_128",
        schedule_sampler="uniform",
        threshold=10000,
        lr=1e-4,
        weight_decay=1e-5,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        local_rank=0,
        noise_layers=noise_layers,
    )
    defaults.update(train_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
