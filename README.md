
<p align="center">
  <!-- <img src="docs/figs/logo.png" align="center" width="50%"> -->
  
  <h3 align="center"><strong>[ICCV 2025] Stable Score Distillation</strong></h3>

<div align="center">

<a href='https://arxiv.org/abs/2311.14521'><img src='https://img.shields.io/badge/arXiv-2311.14521-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</div>


<!-- ## Demo Videos
<details open>
  <summary>Swift and controllable 3D editing with only 2-7 minutes.</summary>

https://github.com/buaacyw/GaussianEditor/assets/52091468/10740174-3208-4408-b519-23f58604339e

https://github.com/buaacyw/GaussianEditor/assets/52091468/44797174-0242-4c82-a383-2d7b3d4fd693


https://github.com/buaacyw/GaussianEditor/assets/52091468/18dd3ef2-4066-428a-918d-c4fe673d0af8
</details> -->

<!-- ## Release
- [12/5] Docker support. Great thanks to [jhuangBU](https://github.com/jhuangBU). For windows, you can try [this guide](https://github.com/buaacyw/GaussianEditor/issues/9) and [this guide](https://github.com/buaacyw/GaussianEditor/issues/14).
- [11/29] Release segmentation confidence score scaler. You can now scale the threshold of semantic tracing masks. 
- [11/27] 🔥 We released **GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting** and beta version of GaussianEditing WebUI. -->

## Contents
<!-- - [Demo Videos](#demo-videos)
- [Release](#release) -->
- [Contents](#contents)
- [Installation](#installation)
- [Tips](#tips)
- [Command Line](#command-line)
- [Acknowledgement](#acknowledgement)

## Installation
Our environment was tested on Ubuntu 22, CUDA 11.7 with 3090.
```
conda create -n ssd python=3.8 -y 
conda activate ssd

conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc
conda install cuda-toolkit==11.7
pip install ninja
pip install cmake
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Gaussian Splatting
cd gaussiansplatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# Required packages
cd ..
pip install tqdm
pip install plyfile
pip install mediapipe
pip install diffusers==0.27.2
pip install -r requirements_all.txt
```
We provide an [environment.yaml](https://github.com/Alex-Zhu1/SSD/environment.yaml) file to help you verify.


## Tips

1. If the default resolution of **512×512** is not suitable, you can modify the [line here](https://github.com/Alex-Zhu1/SSD/blob/3e1e01d773664c646e7194d3935b56fab3407049/threestudio/data/gs_load.py#L367) by setting `self.use_original_resolution` to `True`, and adjust the resolution accordingly. For example, if the original `(height, width)` is `(729, 985)`, you may change it to something like `(512, 692)`.

2. Some prompts may not work well with **SD2.1**. In such cases, you can try using **IP2P** instead.

3. We provide the evaluation metric code.

## Command Line



## Acknowledgement

Most of our code is adapted from the excellent works of [GaussianEditor](https://github.com/buaacyw/GaussianEditor) and [Threestudio](https://github.com/threestudio-project/threestudio). We sincerely thank the authors for their great contributions.

We also refer to the following projects:

- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)  
- [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
