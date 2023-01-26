import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy", "superconverge_lr", "piecewise_lr", "linear_lr", "onedrop_lr", "multipledecay_lr"]




def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "superconverge_lr": superconverge_lr,
        "piecewise_lr": piecewise_lr,
        "linear_lr": linear_lr,
        "onedrop_lr": onedrop_lr,
        "multipledecay_lr": multipledecay_lr

    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.multistep_lr_gamma ** (epoch // args.multistep_lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def superconverge_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        lr = np.interp([epoch], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr, 0])[0]

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def piecewise_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch / args.epochs < 0.5:
            lr =  args.lr
        elif epoch / args.epochs < 0.75:
            lr =  args.lr / 10.
        else:
            lr =  args.lr / 100.

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def linear_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        lr = np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr, args.lr, args.lr / 10, args.lr / 100])[0]

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def onedrop_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.lr_drop_epoch * args.epochs:
            lr = args.lr
        else:
            lr = args.lr_one_drop * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def multipledecay_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        lr = args.lr - (epoch//(args.epochs//10))*(args.lr/10)

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
