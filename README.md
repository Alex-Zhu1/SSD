
<p align="center">
  <!-- <img src="docs/figs/logo.png" align="center" width="50%"> -->
  
  <h3 align="center"><strong>[ICCV 2025] Stable Score Distillation</strong></h3>

<div align="center">

<a href='http://arxiv.org/abs/2507.09168'><img src='https://img.shields.io/badge/arXiv-2311.14521-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</div>

## Contents
<!-- - [Demo Videos](#demo-videos)
- [Release](#release) -->
- [Contents](#contents)
- [Installation](#installation)
- [Tips](#tips)
- [Command Line](#command-line)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

Our environment was tested on Ubuntu 22, CUDA 11.7 with 3090.

```bash

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

3. Most configurations are adapted from [GaussianEditor](https://github.com/buaacyw/GaussianEditor).  
Our pipeline relies on [three](https://github.com/Alex-Zhu1/SSD/blob/25316a047a638291cde1afbf427e750a8e23651d/configs/edit-sd-ours.yaml#L56) key configuration: `cross-prompt`, `cross-trajectory`, and `prompt-enhancement`.  
If the default values — **cross-trajectory: 2.0** and **enhance_scale: 5.5** — lead to suboptimal results, users can try adjusting the weights.

4. We provide test [data](https://drive.google.com/file/d/1q5ReFKafdojNrRKHGroeYT_qO7LREoLn/view?usp=drive_link) and the evaluation metric code.

## Command Line

Please try our demo by running [script/face.sh](https://github.com/Alex-Zhu1/SSD/blob/main/script/face.sh).

## Citation

If you find our work helpful in your project, please cite:

```BiBTeX
@misc{zhu2025stablescoredistillation,
      title={Stable Score Distillation}, 
      author={Haiming Zhu and Yangyang Xu and Chenshu Xu and Tingrui Shen and Wenxi Liu and Yong Du and Jun Yu and Shengfeng He},
      year={2025},
      eprint={2507.09168},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.09168}, 
}
```

## Acknowledgement

Most of our code is adapted from the excellent works of [GaussianEditor](https://github.com/buaacyw/GaussianEditor) and [Threestudio](https://github.com/threestudio-project/threestudio). We sincerely thank the authors for their great contributions.

We also refer to the following projects:

- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)  
- [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
- [DreamCatalyst](https://github.com/kaist-cvml/DreamCatalyst)