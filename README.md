# Unleashing Mask: Explore the Potential Out-Of-Distribution Detection Capability

## Setup

1. Set up a virtualenv with python 3.7.4. You can use pyvenv or conda for this.
2. Run ```pip install -r requirements.txt``` to get requirements
3. Create a data directory as a base for all datasets. For example, if your base directory is ```/mnt/datasets``` then imagenet would be located at ```/mnt/datasets/ImageNet``` and CIFAR-10 would be located at ```/mnt/datasets/cifar10```


## Starting an Experiment 

We use config files located in the ```configs/``` folder to organize our experiments. The basic setup for any experiment is:

```bash
python main.py --config <path/to/config> <override-args>
```

Common example ```override-args``` include ```--multigpu=<gpu-ids seperated by commas, no spaces>``` to run on GPUs. Run ```python main --help``` for more details.

We provide a pretrained DenseNet-101 model in ```pretrained```.

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

To estimate the loss constraint for UM/UMAP
```bash
python estimate_loss.py --config configs/estimate_loss/estimate_loss_cifar10.yaml \
                --data <path/to/data-dir>
```

To experiment with the post-hoc OOD detection methods

```bash
python main.py --config configs/smallscale/example.yaml \
               --multigpu 0 \
               --name cifar10_UM_post_hoc \
               --data <path/to/data-dir> \
               --UM <estimated_loss>
```

To experiment with the Outlier Exposure (OE) OOD detection methods. Use flag ```--sample``` to control the sample method.

```bash
python main_OE.py --config configs/smallscale/example.yaml \
               --multigpu 0 \
               --name cifar10_UM_oe \
               --data <path/to/data-dir> \
               --sample random \
               --UM <estimated_loss>
```


### Tracking

When your experiment is done, your experiment base directory will automatically be written to ```runs/<config-name>/prune-rate=<prune-rate>/<experiment-name>``` with ```settings.txt```, ```<ID-Dataset>_log.txt```, ```checkpoints/``` and ```logs/``` subdirectories. If your experiment happens to match a previously created experiment base directory then an integer increment will be added to the filepath (eg. ```/0```, ```/1```, etc.). Checkpoints by default will have the first, best, and last models. To change this behavior, use the ```--save-every``` flag. 


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

