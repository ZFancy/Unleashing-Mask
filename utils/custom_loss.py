import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from args import args

CrossEntropyLoss = nn.CrossEntropyLoss()

class CustomLoss(nn.Module):
    def __init__(self, criterion=None, model=None):
        super(CustomLoss, self).__init__()
        self.criterion = criterion
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        loss = args.beta * self.CrossEntropyLoss(outputs, labels)
        loss = (loss - args.UM_ce).abs() + args.UM_ce
        temperature = args.temperature
        alpha = args.alpha
        if self.criterion == "EnergyLoss":
            energy = alpha / (temperature * torch.logsumexp(outputs / temperature, 1).mean())
            energy = (energy - args.UM_e).abs() + args.UM_e
            if args.UMe:
                loss = (loss - energy).abs() + energy
            else:
                loss += energy
        if self.criterion == "LogEnergyLoss":
            energy = alpha / (temperature * torch.logsumexp(-outputs / temperature, 1).mean())
            loss += energy
        loss = (loss - args.UM).abs() + args.UM
        return loss

def EnergyLoss(outputs, labels):
    temperature = args.temperature
    alpha = args.alpha
    energy = temperature * torch.logsumexp(outputs / temperature, 1)
    loss = CrossEntropyLoss(outputs, labels) + alpha / energy
    print(loss.shape)
    return loss