import torch
from torch.nn import Module
from torch.nn import KLDivLoss
from torch.autograd import Variable
import numpy as np


class emd_loss(Module):

    def forward(self, input, target):
        input_cdf = torch.cumsum(input, dim=1)
        target_cdf = torch.cumsum(target, dim=1)
        wass_dist = torch.mean(torch.sum(torch.abs(input_cdf - target_cdf), dim=1))
        return wass_dist


class cramer_loss(Module):

    def forward(self, input, target):
        input_cdf = torch.cumsum(input, dim=1)
        target_cdf = torch.cumsum(target, dim=1)
        cramer_dist = torch.sum(torch.pow((input_cdf-target_cdf),2),dim=1)
        return torch.mean(cramer_dist)


def test_emd_loss(inputs, targets):
    loss_fn = emd_loss()
    loss = loss_fn(inputs, targets)
    print("Loss: {}".format(loss))
