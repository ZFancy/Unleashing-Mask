"""
Replications of models from Frankle et al. Lottery Ticket Hypothesis
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from utils.builder import get_builder
import math
import numpy as np
from torch.autograd import Variable

from args import args

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        nums = []
        out = torch.tensor([]).cuda()
        for i, score in enumerate(scores):
            s = score.clone().flatten()
            out = torch.cat((out, s), 0)
            nums.append(s.numel())
        _, idx = out.sort()
        j = int((1 - k) * out.numel())
        out[idx[:j]] = 0
        out[idx[j:]] = 1
        result = []
        for score in scores:
            m = Variable(out[:score.numel()].view(list(score.shape)).data, requires_grad=True)
            result.append(m)
            out = torch.cat((torch.tensor([]).cuda(), out[score.numel():]), 0)
        return tuple(result)

    @staticmethod
    def backward(ctx, g):
        return g, None
        

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        builder = get_builder()
        self.conv1 = builder.conv3x3(3, 64, first_layer=True)
        self.conv2 = builder.conv3x3(64, 64)
        self.maxpool = nn.MaxPool2d((2, 2))

        self.fc1 = builder.linear(64 * 16 * 16, 256)
        self.fc2 = builder.linear(256, 256)
        self.fc3 = builder.linear(256, args.num_classes)

    def forward(self, x):
        w1 = self.conv1.weight
        w2 = self.conv2.weight
        f1 = self.fc1.weight
        f2 = self.fc2.weight
        f3 = self.fc3.weight
        out = F.relu(F.conv2d(x, w1, self.conv1.bias, self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups))
        out = F.relu(F.conv2d(out, w2, self.conv2.bias, self.conv2.stride, self.conv2.padding, self.conv2.dilation, self.conv2.groups))
        out = self.maxpool(out)
        out = out.view(out.size(0), 64 * 16 * 16)
        out = F.relu(F.linear(out, f1, self.fc1.bias))
        out = F.relu(F.linear(out, f2, self.fc2.bias))
        out = F.linear(out, f3, self.fc3.bias)
        return out.squeeze()

class SubConv2(nn.Module):
    def __init__(self):
        super(SubConv2, self).__init__()
        builder = get_builder()
        self.conv1 = builder.conv3x3(3, 64, first_layer=True)
        self.conv2 = builder.conv3x3(64, 64)
        self.maxpool = nn.MaxPool2d((2, 2))

        self.fc1 = builder.linear(64 * 16 * 16, 256)
        self.fc2 = builder.linear(256, 256)
        self.fc3 = builder.linear(256, args.num_classes)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        score_list = [self.conv1.scores.abs(), self.conv2.scores.abs(), self.fc1.scores.abs(), self.fc2.scores.abs(), self.fc3.scores.abs()]
        return score_list

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        subnet[0].requires_grad = True
        subnet[1].requires_grad = True
        subnet[2].requires_grad = True
        subnet[3].requires_grad = True
        subnet[4].requires_grad = True
        w1 = self.conv1.weight * subnet[0]
        w2 = self.conv2.weight * subnet[1]
        f1 = self.fc1.weight * subnet[2]
        f2 = self.fc2.weight * subnet[3]
        f3 = self.fc3.weight * subnet[4]
        out = F.relu(F.conv2d(x, w1, self.conv1.bias, self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups))
        out = F.relu(F.conv2d(out, w2, self.conv2.bias, self.conv2.stride, self.conv2.padding, self.conv2.dilation, self.conv2.groups))
        out = self.maxpool(out)
        out = out.view(out.size(0), 64 * 16 * 16)
        out = F.relu(F.linear(out, f1, self.fc1.bias))
        out = F.relu(F.linear(out, f2, self.fc2.bias))
        out = F.linear(out, f3, self.fc3.bias)
        return out.squeeze()


class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv1 = builder.conv3x3(3, 64, first_layer=True)
        self.conv2 = builder.conv3x3(64, 64)
        self.conv3 = builder.conv3x3(64, 128)
        self.conv4 = builder.conv3x3(128, 128)
        self.maxpool = nn.MaxPool2d((2, 2))

        self.fc1 = builder.linear(32 * 32 * 8, 256)
        self.fc2 = builder.linear(256, 256)
        self.fc3 = builder.linear(256, args.num_classes)

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * 8, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()

class Conv4_(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv1 = builder.conv3x3(3, 64, first_layer=True)

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * 8, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    @property
    def clamped_scores(self):
        return

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(256 * 4 * 4, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()

class Conv8(nn.Module):
    def __init__(self):
        super(Conv8, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(256, 512),
            nn.ReLU(),
            builder.conv3x3(512, 512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        builder = get_builder()
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, 300, first_layer=True),
            nn.ReLU(),
            builder.conv1x1(300, 100),
            nn.ReLU(),
            builder.conv1x1(100, 10),
        )

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()

def scale(n):
    return int(n * args.width_mult)


class Conv4Wide(nn.Module):
    def __init__(self):
        super(Conv4Wide, self).__init__()
        builder = get_builder()

        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(128)*8*8, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(128)*8*8, 1, 1)
        out = self.linear(out)
        return out.squeeze()



class Conv6Wide(nn.Module):
    def __init__(self):
        super(Conv6Wide, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(128), scale(256)),
            nn.ReLU(),
            builder.conv3x3(scale(256), scale(256)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(256) * 4 * 4, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(256) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()