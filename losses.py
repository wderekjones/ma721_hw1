import torch
from torch.nn import Module
from torch.nn import KLDivLoss
from torch.autograd import Variable
import numpy as np


# this should go into a torch_utils module?, found this on https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]


class emd_loss(Module):

    def forward(self, input, target):
        # may need to renorm vectors to length 1
        input_inv_cdf = torch.nn.ReLU()(torch.cumsum(input.renorm(dim=1, maxnorm=1, p=1), dim=1))
        target_inv_cdf = torch.nn.ReLU()(torch.cumsum(target.renorm(dim=1, maxnorm=1, p=1), dim=1))
        wass_dist = torch.abs(input_inv_cdf - target_inv_cdf)
        return torch.mean(wass_dist)


class cramer_loss(Module):
    def __init__(self): # why is this here?
        super(cramer_loss, self).__init__()

    def forward(self, input, target):
        input_cdf = torch.cumsum(input, dim=1)
        target_cdf = torch.cumsum(target, dim=1)
        cramer_loss = torch.sqrt(
            torch.sum(torch.pow(torch.abs(input_cdf - target_cdf), 2)))
        return torch.mean(cramer_loss)


# class dice_loss(Module):

def square_torch_variable(t1_var, t2_var):
    return torch.mul((t1_var), (t2_var))


def test_emd_loss(inputs, targets):
    loss_fn = emd_loss()
    loss = loss_fn(inputs, targets)
    print("Loss: {}".format(loss))

    # if __name__ == "__main__":

# N = 1
# m = 5
# X = torch.rand(N,m)
# y = torch.rand(N,1)
# test_emd_loss(X,y)
