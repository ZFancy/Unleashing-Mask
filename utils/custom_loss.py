import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from args import args

CrossEntropyLoss = nn.CrossEntropyLoss()

class CustomLoss(nn.Module):
    def __init__(self, criterion=None):
        super(CustomLoss, self).__init__()
        self.criterion = criterion
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        loss = args.beta * self.CrossEntropyLoss(outputs, labels)
        if self.criterion != "CrossEntropyLoss":
            loss = (loss - args.UM).abs() + args.UM
        return loss