import torch
import numpy as np

from model import Net


def ye_score(test_acc, n_params, n_epochs):
    return np.divide(test_acc,(n_params * n_epochs))


def init_network(model, gain=1):
    for child in model.children():
        for param in child.named_parameters():
            if 'weight' in param[0]:

                torch.nn.init.normal(param[1],1e-3,1)
            elif 'bias' in param[0]:
                torch.nn.init.uniform(param[1],1,1) # initialize biases to slightly positive values
    return model


def test_init(D_in, H, D_out, N):
    model = Net(D_in=D_in, H=H, D_out=D_out, N=N)
    init_network(model,1)


# test_init(500, 30, 2, 100)
