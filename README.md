<div align="center">

<h1>【INFFUS 2025】DiffMark: Diffusion-based Robust Watermark Against Deepfakes</h1>

</div>

## ⭐ Abstract

Deepfakes pose significant security and privacy threats through malicious facial manipulations. While robust watermarking can aid in authenticity verification and source tracking, existing methods often lack the sufficient robustness against Deepfake manipulations. Diffusion models have demonstrated remarkable performance in image generation, enabling the seamless fusion of watermark with image during generation. In this study, we propose a novel robust watermarking framework based on diffusion model, called DiffMark. By modifying the training and sampling scheme, we take the facial image and watermark as conditions to guide the diffusion model to progressively denoise and generate corresponding watermarked image. In the construction of facial condition, we weight the facial image by a timestep-dependent factor that gradually reduces the guidance intensity with the decrease of noise, thus better adapting to the sampling process of diffusion model. To achieve the fusion of watermark condition, we introduce a cross information fusion (CIF) module that leverages a learnable embedding table to adaptively extract watermark features and integrates them with image features via cross-attention. To enhance the robustness of the watermark against Deepfake manipulations, we integrate a frozen autoencoder during training phase to simulate Deepfake manipulations. Additionally, we introduce Deepfake-resistant guidance that employs specific Deepfake model to adversarially guide the diffusion sampling process to generate more robust watermarked images. Experimental results demonstrate the effectiveness of the proposed DiffMark on typical Deepfakes.

## 🚀 Introduction

<div align="center">
    <img width="1000" alt="image" src="figs\first.png">
</div>

<div align="center">
The difference between our method and the existing methods.
</div>

## 📻 Overview

<div align="center">
    <img width="1000" alt="image" src="figs\framework.png">
</div>

<div align="center">
Illustration of the overall architecture of DiffMark.
</div>

## 📑 TODO

- [x] Project page released
- [x] Dataset preparation instructions released
- [ ] Release of core implementation
- [ ] Release of training and evaluation scripts
- [ ] Pretrained model and demo

## 🖥️ Environment Setup

```python
conda env create -f environment.yml
```

## 📁 Datasets

DiffMark is trained on the CelebA-HQ dataset and evaluated on both CelebA-HQ and LFW datasets at resolutions of 128×128 and 256×256. We do not own these datasets; they can be downloaded from their respective official websites.

- [Download CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Download LFW](https://vis-www.cs.umass.edu/lfw/)

## 🔧 Train

```python
python scripts/image_train.py
```

## 🧪 Test

```python
python scripts/image_test.py
```

## 🖼️ Visualization

<div align="center">
    <img width="1000" alt="image" src="figs\vision.png">
</div>

<div align="center">
Visualization in DiffMark.



## Citation
@article{sun2025diffmark,
  title={DiffMark: Diffusion-based Robust Watermark Against Deepfakes},
  author={Chen Sun, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Liejun Wang, Dan Ma, and Gaobo Yang},
  journal={Information Fusion},
  year={2025}
}
