from functools import partial
import os
import sys
import time
import pathlib
import shutil
import math
import importlib
import data
import models
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from args import args
from utils.logging import Logger
from utils.custom_loss import CustomLoss
from utils.schedulers import get_policy
from utils.neural_linear_opt import NeuralLinear, SimpleDataset

def set_retraining(model):
    if hasattr(model, "retraining"):
        model.retraining = True

def set_threshold(model):
    if hasattr(model, "set_threshold"):
        model.set_threshold()

def get_threshold(model, prune_rate, set_mask=False):
    values = torch.tensor([]).cuda()
    for n, m in model.named_modules():
        if hasattr(m, "clamped_scores") and hasattr(m, "mask") and m.mask is not None:
            values = torch.cat((values, m.clamped_scores[torch.nonzero(m.mask > 0, as_tuple=True)].flatten()), 0)

    threshold = float(torch.quantile(values, 1 - prune_rate, dim=0, keepdim=False))

    for n, m in model.named_modules():
        if hasattr(m, "threshold") and m.threshold is not None:
            m.set_threshold(threshold, set_mask)
            

def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")
    if hasattr(model, "set_prune_rate"):
        model.set_prune_rate(prune_rate)
        print(f"==> Setting prune rate of model to {prune_rate}")
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("scores"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    if args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )

        if "state_dict" in pretrained:
            pretrained = pretrained["state_dict"]

        model_state_dict = model.state_dict()
        pretrained_state_dict = {}
        for k, v in pretrained.items():
            k = k.replace("module.", "") if "module." in k else k
            pretrained_state_dict[k] = v

        # for k1, k2 in zip(pretrained_state_dict.keys(), model_state_dict.keys()):
        #     print(k1 + "\t" + k2)

        for k, v in pretrained_state_dict.items():
            if k not in model_state_dict or v.shape != model_state_dict[k].shape:
                print("IGNORE:", k)

        pretrained = {
            k: v
            for k, v in pretrained_state_dict.items()
            if (k in model_state_dict and v.shape == model_state_dict[k].shape)
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(dataset):
    print(f"=> Getting {dataset} dataset")
    dataset = getattr(data, dataset)(args)
    return dataset

def get_criterion(args):
    print(f"=> Getting {args.criterion} criterion")
    if args.label_smoothing is None:
        if args.criterion != "CrossEntropyLoss":
            criterion = CustomLoss(args.criterion).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing).cuda()
    return criterion

def get_model(args, full=False):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
        or args.linear_type != "DenseLinear"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")
        if not full:
            set_model_prune_rate(model, prune_rate=args.prune_rate)
        else :
            set_model_prune_rate(model, prune_rate=1.0)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights and not full:
        freeze_model_weights(model)
    if args.freeze_subnet:
        freeze_model_subnet(model)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)
    if not log_base_dir.exists():
        os.makedirs(log_base_dir)
    if not ckpt_base_dir.exists():
        os.makedirs(ckpt_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))
    sys.stdout = Logger(os.path.join(run_base_dir, args.set + '_log.txt'), mode='a')

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


def select_ood_opt(ood_loader, ood_branch, batch_size, num_classes, pool_size, ood_dataset_size):
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset>=0 and offset<10000:
        offset = np.random.randint(len(ood_loader.dataset))
    ood_loader.dataset.offset = offset
    out_iter = iter(ood_loader)
    print('Start selecting OOD samples...')
    # select ood samples
    with torch.no_grad():
        all_ood_input = torch.empty(0,3,32,32)
        all_abs_val = torch.empty(0)
        duration = 0
        init_start = time.time()
        for k in range(pool_size): 
            start = time.time()
            try:
                out_set = next(out_iter)
            except StopIteration:
                offset = np.random.randint(len(ood_loader.dataset))
                while offset>=0 and offset<10000:
                    offset = np.random.randint(len(ood_loader.dataset))
                ood_loader.dataset.offset = offset
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0] 
            output = ood_branch.predict(input.cuda())
            abs_val = torch.abs(output).squeeze() 
            duration += time.time() - start
            all_ood_input = torch.cat((all_ood_input, input), dim = 0)
            all_abs_val = torch.cat((all_abs_val, abs_val.detach().cpu()), dim = 0)
    print('Scanning Time: ',  duration)
    _, selected_indices = torch.topk(all_abs_val, ood_dataset_size, largest=False)
    print('Total OOD samples: ', len(selected_indices))
    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()

    ood_train_loader = torch.utils.data.DataLoader(
        SimpleDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True, num_workers = 0)

    print('Time: ', time.time()-init_start)
    return ood_train_loader


def select_ood(ood_loader, batch_size, num_classes, pool_size, ood_dataset_size):
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset>=0 and offset<10000:
        offset = np.random.randint(len(ood_loader.dataset))
    ood_loader.dataset.offset = offset
    out_iter = iter(ood_loader)
    print('Start selecting OOD samples...')
    # select ood samples
    with torch.no_grad():
        all_ood_input = torch.empty(0,3,32,32)
        all_abs_val = torch.empty(0)
        duration = 0
        init_start = time.time()
        for k in range(pool_size): 
            start = time.time()
            try:
                out_set = next(out_iter)
            except StopIteration:
                offset = np.random.randint(len(ood_loader.dataset))
                while offset>=0 and offset<10000:
                    offset = np.random.randint(len(ood_loader.dataset))
                ood_loader.dataset.offset = offset
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0] 
            duration += time.time() - start
            all_ood_input = torch.cat((all_ood_input, input), dim = 0)
    print('Scanning Time: ',  duration)
    selected_indices = random.sample(range(len(all_ood_input)), ood_dataset_size)
    selected_indices = torch.tensor(selected_indices)
    print('Total OOD samples: ', len(selected_indices))
    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()

    ood_train_loader = torch.utils.data.DataLoader(
        SimpleDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True, num_workers = 0)

    print('Time: ', time.time()-init_start)
    return ood_train_loader

