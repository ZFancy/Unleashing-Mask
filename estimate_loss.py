import os
import sys
import pathlib
import random
import time
import numpy as np
import sklearn.metrics as sk

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    get_trainer,
    get_dataset,
    get_criterion,
    get_model,
    get_optimizer,
    get_directories,
    _run_dir_exists,
    write_result_to_csv,
    set_gpu,
    resume,
    pretrained,
    set_model_prune_rate
)
from utils.schedulers import get_policy
from utils.get_scores import measures, ood_measure

from args import args
import importlib

import data
import models
from utils.custom_loss import CustomLoss

def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # Set up directories
    if args.set == "CIFAR10" or args.set == "CIFAR100":
        args.normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    else:
        args.normalizer = None

    print("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    print(args)

    args.gpu = None
    train, validate, modifier = get_trainer(args)

    if args.multigpu is not None:
        print("Use GPU: {} for training".format(args.multigpu))

    data = get_dataset(args.set)

    # create model and optimizer
    assert(args.conv_type is not "DenseConv" or args.linear_type is not "DenseLinear")
    model = get_model(args)

    assert(args.pretrained)
    if args.pretrained:
        pretrained(args, model)

    model = set_gpu(args, model)

    criterion = get_criterion(args)

    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0

    start_validation = time.time()
    estimated_loss = modifier(data.train_loader, model, criterion, args, None, -1)
    validation_time.update((time.time() - start_validation) / 60)

    print(f'The estimated loss constraint is {estimated_loss}')
    return


if __name__ == "__main__":
    main()
