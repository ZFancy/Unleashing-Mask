import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args


DenseConv = nn.Conv2d

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class _GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, mask, threshold):
        # Get the subnetwork by sorting the scores and using the top k%
        # if parser_args.prune_type == "layer_wise":
        out = scores.clone()
        if parser_args.prune_type == "layer_wise":
            thresh = torch.quantile(scores[torch.nonzero(mask == 1, as_tuple=True)].flatten(), 1 - k, dim=0, keepdim=False)
        elif parser_args.prune_type == "model_wise":
            thresh = threshold
        condition1 = (mask == 1).to(scores.device)
        condition2 = out >= thresh
        out = torch.zeros(scores.size()).to(scores.device)
        out[torch.nonzero(condition1 & condition2, as_tuple=True)] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None, None


# Not learning weights, finding subnet
class _SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.mask = nn.Parameter(torch.ones(self.weight.size()))
        self.mask.requires_grad = False
        self.temp_mask = nn.Parameter(torch.ones(self.weight.size()))
        self.temp_mask.requires_grad = False
        # self.mask = torch.ones(self.weight.size())
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.threshold.requires_grad = False
        self.retraining = False

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
        m = GetSubnet.apply(self.clamped_scores, self.prune_rate, self.mask, self.threshold)
        self.temp_mask.data = m
        w = self.weight * m
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class MagnitudeConv(nn.Conv2d):
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
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        subnet, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs[subnet == 0.0] = 0.0

        return grad_inputs, None


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

