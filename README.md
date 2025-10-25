# DiffMark: Diffusion-based Robust Watermark Against Deepfakes

<br>
This is the implementation of the paper "DiffMark: Diffusion-based Robust Watermark Against Deepfakes".

It must be acknowledged that this work still has several limitations. Nevertheless, I sincerely hope it may be helpful to your research. Best wishes!

## TODO

- [x] Project page released
- [x] Dataset preparation instructions released
- [x] Model code released
- [x] Training and evaluation scripts released

## Environment

Please refer to environment.txt.

## Datasets

This model is trained on the CelebA-HQ dataset and evaluated on both CelebA-HQ and LFW datasets at resolutions of 128×128 and 256×256. We do not own these datasets; they can be downloaded from their respective official websites.

- [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [LFW](https://vis-www.cs.umass.edu/lfw/)

## Noise Layers

Please download the [autoencoder](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip) from Stable-Diffusion and place it at ./noise_layers/ldm/models/checkpoints/vq-f4 for training.

The following Deepfake models are used for testing in our experiments:

- [SimSwap](https://github.com/neuralchen/SimSwap)
- [UniFace](https://github.com/xc-csc101/UniFace)
- [CSCS](https://github.com/ICTMCG/CSCS)
- [StarGAN](https://github.com/yunjey/stargan)
- [FSRT](https://github.com/andrerochow/fsrt)

Since we don't own the source code, we recommend downloading and placing the model source code and weights by yourself. We provide the corresponding python scripts for processing.

## Train

- 128x128 model:

```
MODEL_FLAGS="--attention_resolutions 16,8 --image_size 128 --message_length 30 --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1"
TRAIN_FLAGS="--data_dir CelebA-HQ/train_128 --batch_size 16 --lr 1e-4 --lr_anneal_steps 151200 --weight_decay 1e-5 --threshold 5000"
```

- 256x256 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --image_size 256 --message_length 128 --embedding_dim 1024 --num_channels 64 --num_heads 4 --num_res_blocks 1"
TRAIN_FLAGS="--data_dir CelebA-HQ/train_256 --batch_size 16 --lr 1e-4 --lr_anneal_steps 151200 --weight_decay 1e-5 --threshold 10000"
```

- Train Command:

```python
python scripts/image_train.py $MODEL_FLAGS $TRAIN_FLAGS
```

## Test

- 128x128 model:

```
MODEL_FLAGS="--attention_resolutions 16,8 --image_size 128 --message_length 30 --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1"
TEST_FLAGS="--model_path ema_0.9999_151200.pt --data_dir CelebA-HQ/test_128 --batch_size 16 --rescale_timesteps True --timestep_respacing ddim10 --cover_dir CelebA-HQ --use_guidance False"
```

- 256x256 model:

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --image_size 256 --message_length 128 --embedding_dim 1024 --num_channels 64 --num_heads 4 --num_res_blocks 1"
TEST_FLAGS="--model_path ema_0.9999_151200.pt --data_dir CelebA-HQ/test_256 --batch_size 16 --rescale_timesteps True --timestep_respacing ddim10 --cover_dir CelebA-HQ --use_guidance False"
```

- Test Command:

```python
python scripts/image_test.py $MODEL_FLAGS $TEST_FLAGS
```

- To enable deepfake-resistant guidance, set use_guidance to True.

## Acknowledgements

This work is inspired by remarkable studies such as [Guided-Diffusion](https://github.com/openai/guided-diffusion), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion), [SepMark](https://github.com/sh1newu/SepMark) and [LampMark](https://github.com/wangty1/LampMark), with [Guided-Diffusion](https://github.com/openai/guided-diffusion) being a foundational inspiration. We extend our sincere thanks to all the contributors of these works.

## Citation

```bibtex
@article{SUN2025103801,
  title = {DiffMark: Diffusion-based Robust Watermark Against Deepfakes},
  author = {Chen Sun and Haiyang Sun and Zhiqing Guo and Yunfeng Diao and Liejun Wang and Dan Ma and Gaobo Yang and Keqin Li},
  journal = {Information Fusion},
  year = {2025},
}
```
