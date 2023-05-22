<h1 align="center"> Unleashing Mask: Explore the Intrinsic Out-of-Distribution Detection Capability</h1>


This repo contains the sample code of our proposed ```Unleashing Mask (UM)``` and its variant ```Unleashing Mask Adopt Pruning (UMAP)``` to adjust the given well-trained model for OOD detection in our paper: [Unleashing Mask: Explore the Intrinsic Out-of-Distribution Detection Capability](https://https://github.com/ZFancy/Unleashing-Mask) (ICML 2023).
<p align="center"><img src="./figures/framework_overview.jpg" width=90% height=50%></p>
<p align="center"><em>Figure.</em> Framework overview of UM.</p>

## TL;DR
Our work reveal an intermediate training stage with better out-of-distribution (OOD) discriminative capability of the well pre-trained model (on classifying the in-distribution (ID) data). We propose Unleashing Mask to restore it of the given well-trained model for OOD detection, by fine-tuning with the estimated loss contraint to forget those relatively atypical ID samples.

## Setup

1. Set up a virtualenv with python 3.7.4. You can use pyvenv or conda for this.
2. Run ```pip install -r requirements.txt``` to get requirements
3. Create a data directory as a base for all datasets. For example, if your base directory is ```/mnt/datasets``` then imagenet would be located at ```/mnt/datasets/ImageNet``` and CIFAR-10 would be located at ```/mnt/datasets/cifar10```

## Quick Usage

UM is quite a easy-to-adopt method to use in your own pipeline for enhancing the OOD discriminative capability. The key point is to add the loss constraint to the CrossEntropyLoss:
```python
class CustomLoss(nn.Module):
    def __init__(self, criterion=None):
        super(CustomLoss, self).__init__()
        self.criterion = criterion
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        loss = args.beta * self.CrossEntropyLoss(outputs, labels)
        if self.criterion != "CrossEntropyLoss":
            # args.UM is the estimated loss constriant
            loss = (loss - args.UM).abs() + args.UM 
        return loss
```

To use UM/UMAP in your pipeline:
```python
...
from utils.custom_loss import CustomLoss

...
criterion = CustomLoss("UM")

...
for input, target in data:
    ...
    loss = criterion(model(input), target) # use this to calculate UM loss
    loss.backward()

...
```

Or apply UM in a more straightforward way:
```python
...
import torch.nn as nn

...
criterion = nn.CrossEntropyLoss()

...
for input, target in data:
    ...
    loss = criterion(model(input), target)
    loss = (loss - args.UM).abs() + args.UM # args.UM is the estimated loss constriant
    loss.backward()

...
```

## Starting an Experiment 

We use config files located in the ```configs/``` folder to organize our experiments. The basic setup for any experiment is:

```bash
python main.py --config <path/to/config> <override-args>
```

Common example ```override-args``` include ```--multigpu=<gpu-ids seperated by commas, no spaces>``` to run on GPUs. Run ```python main --help``` for more details.

We provide pretrained DenseNet-101 models in ```runs/pretrained_models```, which are trained on CIFAR-10/CIFAR-100 respectively. Note that for each setting, we provide the best (middle-stage) model and the last (final-stage) model. The pretrained model are named as ```densenet_<cifar10/cifar100>_<best/last>.pth```.


### Example Run

Before you begin experiment, please arrange your dataset directory as follows:
```
<path/to/data-dir>:
    |-cifar10
        |-cifar-10-batches-py
    |-cifar100
        |-cifar-100-python
    |-dtd
    |-ImageNet
    |-iNaturalist
    |-iSUN
    |-LSUN
    |-LSUN-resize
    |-MNIST
    |-Places
    |-SUN
    |-svhn
    |-tinyimagenet
```

To estimate the loss constraint for UM/UMAP. Note that you can either estimate the loss constraint by ```estimate_loss.py``` or manual tuning.
```bash
python estimate_loss.py --config configs/estimate_loss/estimate_loss_cifar10.yaml \
                --data <path/to/data-dir>
```

To experiment with the post-hoc OOD detection methods. Use flag```--msp```, ```--odin```, ```--energy```, ```--mahalanobis``` to control the scoring functions respectively. If more than one function is chosen, OOD performance will be measured under these functions respectively. To evaluate OOD performance, you can either use flag ```--final``` to evaluate right after training, or use flag ```--evaluate``` to evaluate a loaded trained model.

```bash
python main.py --config configs/smallscale/densenet_cifar10.yaml \
               --multigpu 0 \
               --name cifar10_UM_post_hoc \
               --data <path/to/data-dir> \
               --UM <estimated_loss>\
               --energy
```
python main.py --config configs/smallscale/densenet_cifar10.yaml

To experiment with the Outlier Exposure (OE) OOD detection methods. Use flag ```--oe-ood-method``` to control the OE-based methods.

```bash
python main_OE.py --config configs/OE/oe-baseline.yaml \
               --multigpu 0 \
               --name cifar10_UM_oe \
               --data <path/to/data-dir> \
               --oe-ood-method <choose from [oe, poem, enregy, doe]> \
               --UM <estimated_loss>
               --energy
```

### Tracking

When your experiment is done, your experiment base directory will automatically be written to ```runs/<config-name>/prune-rate=<prune-rate>/<experiment-name>``` with ```settings.txt```, ```<ID-Dataset>_log.txt```, ```checkpoints/``` and ```logs/``` subdirectories. If your experiment happens to match a previously created experiment base directory then an integer increment will be added to the filepath (eg. ```/0```, ```/1```, etc.). Checkpoints by default will have the first, best, and last models. To change this behavior, use the ```--save-every``` flag. 

## Sample Results

|  $\mathcal{D}_\text{in}$ | Method | AUROC $\uparrow$ | AUPR $\uparrow$             | FPR95 $\downarrow$     | ID-ACC $\uparrow$     | w./w.o. $\mathcal{D}_\text{aux}$   | 
|:----------:|----------|---------|------------------|------------------|------------------|:----:|
| CIFAR-10   | Energy    | 92.07 (0.22) | 92.72 (0.39) | 42.69 (1.31) | **94.01** (0.08) | |
| CIFAR-10   | Energy+**UM**     | **93.73** (0.36) | **94.27** (0.60) | **33.29** (1.70) | 92.80 (0.47) | |
| CIFAR-10   | OE    | 97.07 (0.01) | 97.31 (0.05) | 13.80 (0.28) | 92.59 (0.32) |$\checkmark$ |
| CIFAR-10   | OE+**UM**    | **97.60** (0.03) | **97.87** (0.02) | **11.22** (0.16) | **93.66** (0.12) |$\checkmark$ |

## Requirements

Python 3.7.4, CUDA Version 10.1 (also works with 9.2 and 10.0):

```
absl-py==0.8.1
grpcio==1.24.3
Markdown==3.1.1
numpy==1.17.3
Pillow==6.2.1
protobuf==3.10.0
PyYAML==5.1.2
six==1.12.0
tensorboard==2.0.0
torch==1.3.0
torchvision==0.4.1
tqdm==4.36.1
Werkzeug==0.16.0
```


## Reference Code
- hidden-networks: https://github.com/allenai/hidden-networks

- POEM: https://github.com/deeplearning-wisc/poem

- Energy: https://github.com/wetliu/energy_ood

- ODIN: https://github.com/JoonHyung-Park/ODIN

---
If you find our paper and repo useful, please cite our paper:
```bibtex
@inproceedings{zhu2023unleashing,
title       ={Unleashing Mask: Explore the Intrinsic Out-of-distribution Detection Capability},
author      ={Jianing Zhu and Hengzhuang Li and Jiangchao Yao and Tongliang Liu and Jianliang Xu and Bo Han},
booktitle   ={International Conference on Machine Learning},
year        ={2023}
}
```