import os
import sys
import pathlib
import random
import time
import numpy as np
import sklearn.metrics as sk

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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
    select_ood_opt,
    select_ood
)
from utils.schedulers import get_policy
from utils.get_scores import measures, ood_measure
from utils.neural_linear_opt import NeuralLinear, SimpleDataset

import importlib

import data
import models
from utils.custom_loss import CustomLoss

from args import args

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
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
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
    ood_loaders = [get_dataset(dataset).val_loader for dataset in args.ood_set]
    measure = ood_measure(data.val_loader, ood_loaders, msp=args.msp, energy=args.energy, odin=args.odin, mahalanobis=args.mahalanobis)

    ood_dataset_size = int(len(data.train_loader.dataset) * args.ood_factor)
    print('OOD Dataset Size: ', ood_dataset_size)

    # create model and optimizer
    model = get_model(args)
    full_model = get_model(args, full=True)

    if args.pretrained:
        pretrained(args, model)
        pretrained(args, full_model)

    repr_dim = model.repr_dim

    model = set_gpu(args, model)
    full_model = set_gpu(args, full_model)
    criterion = get_criterion(args)
    
    bayes_nn = NeuralLinear(args, model, repr_dim, output_dim = 1)
    cudnn.benchmark = True

    optimizer = get_optimizer(args, model)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)
    
    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(data.val_loader, full_model, criterion, args, writer=None, epoch=args.start_epoch)
        return
    
    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    print('---------------------------------before training---------------------------------')
    bayes_nn.sample_BDQN()
    start_validation = time.time()
    acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, -1)
    validation_time.update((time.time() - start_validation) / 60)
    if (0) % args.save_every == 0:
        print("checking the OOD performance of the initial model ...")
        measure.ood_metrics(model, 0, data.train_loader)

    # Start training
    ood_loader = get_dataset(args.auxiliary_dataset).train_loader
    for epoch in range(args.start_epoch, args.epochs):
        # print(f'Epoch: [{epoch}]')
        lr_policy(epoch, iteration=None)

        cur_lr = get_lr(optimizer)
        if args.sample == "thompson":
            selected_ood_loader = select_ood_opt(ood_loader, bayes_nn, args.batch_size * args.ood_factor, args.num_classes, args.pool_size, ood_dataset_size)
        elif args.sample == "random" or args.oe:
            selected_ood_loader = select_ood(ood_loader, args.batch_size * args.ood_factor, args.num_classes, args.pool_size, ood_dataset_size)

        # train for one epoch
        start_train = time.time()
        if args.sample == "thompson":
            bayes_nn.train_blr(data.train_loader, selected_ood_loader, criterion, optimizer, epoch)
        elif args.sample == "random" or args.oe:
            bayes_nn.train_oe(data.train_loader, selected_ood_loader, criterion, optimizer, epoch)

        if args.sample == "thompson":
            bayes_nn.update_representation()
            bayes_nn.update_bays_reg_BDQN()
            bayes_nn.sample_BDQN()

        train_time.update((time.time() - start_train) / 60)

        if (epoch + 1) % (args.save_every * 5) == 0:
            measure.ood_metrics(model, epoch+1, data.train_loader)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        # acc1 =  bayes_nn.validate(val_loader, model, criterion, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    # if args.final:
    #     measure.ood_metrics(model, args.epochs, data.train_loader)


if __name__ == "__main__":
    main()
