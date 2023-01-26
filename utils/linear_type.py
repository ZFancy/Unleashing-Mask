import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args
from utils.conv_type import GetSubnet

DenseLinear = nn.Linear

class SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))


    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        m = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * m
        return F.linear(x, w, self.bias)

class _SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.mask = nn.Parameter(torch.ones_like(self.scores))
        self.mask.requires_grad = False
        self.temp_mask = nn.Parameter(torch.ones(self.weight.size()))
        self.temp_mask.requires_grad = False
        # self.mask = torch.ones(self.weight.size())
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.threshold.requires_grad = False

        # # NOTE: initialize the weights like this.
        # nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def set_threshold(self, threshold=None, set_mask=False):
        if threshold is not None:
            self.threshold.data = torch.tensor(threshold).to(self.threshold.device)
            if set_mask:
                condition1 = self.clamped_scores > self.threshold
                condition2 = self.mask > 0
                self.mask.data = torch.where(condition1 & condition2, 1, 0)
        else:
            self.mask.data = self.temp_mask.data

    @property
    def clamped_scores(self):
        if parser_args.prune_subject == 'scores':
            return self.scores.abs()
        elif parser_args.prune_subject == 'weight':
            return self.weight.abs()

    def forward(self, x):
        # if parser_args.prune_type == "layer_wise":
        m = GetSubnet.apply(self.clamped_scores, self.prune_rate, self.mask, self.threshold)
        # elif parser_args.prune_type == "model_wise":
        #     m = self.mask.clone()
        self.temp_mask.data = m
        w = self.weight * m
        return F.linear(x, w, self.bias)

class MagnitudeLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.weight.abs()

    def get_subnet(self):
        scores = self.clamped_scores
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - self.prune_rate) * scores.numel())
        
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out


    def forward(self, x):
        subnet = self.get_subnet()
        w = self.weight * subnet
        return F.linear(x, w, self.bias)