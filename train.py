# built using example code from pytorch getting started with examples documentation http://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
np.random.RandomState(0)
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn import KLDivLoss
from input_pipeline import parse_features
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from input_pipeline import KinaseDataset
from sklearn.preprocessing import Imputer
from losses import emd_loss, cramer_loss
from model import Net


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to data")
args = parser.parse_args()


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# n_bins is the number of class bins
# num_epochs is the number of complete iterations over the data in batch_size increments

num_epochs = 500
batch_size = 100
# look at loss performance by varying number of bins
n_bins = 2
input_shape = 92
N, D_in, H, D_out = batch_size, input_shape, 5, n_bins


# data = HousePriceDataset(args.data, n_bins)
features_list = parse_features("/Users/derekjones2025/workspace/protein_binding/data/all_kinase/with_pocket/Informative_features.csv")
data = KinaseDataset(data_path="/Users/derekjones2025/workspace/protein_binding/data/all_kinase/with_pocket/full_kinase_set.h5",
                     protein_name_list=["lck"], features_list=features_list)
# after loading the data, convert the labels to one-hot

dataloader = DataLoader(data, batch_size=batch_size, num_workers=4, sampler=WeightedRandomSampler(weights=[0.02,50],num_samples=batch_size,replacement=True))

# import the baseline model from model.py
model = Net(D_in=D_in, H=H, D_out=D_out, N=N)

# write functions to do these initializations seperately for weights and biases, write tests for ill-conditioning
# for child in model.children():
#     for param in child.named_parameters():
#         torch.nn.init.uniform(param[1], 1e-1, 5)

# loss_fn = emd_loss()
# learning_rate =1e-3
# loss_fn = cramer_loss()
# simulation_name = "cramer_loss_"+str(time.time())+".png"
loss_fn = emd_loss()
simulation_name = "emd_loss_"+str(time.time())+".png"
learning_rate = 1e-3
# loss_fn = torch.nn.MSELoss()
# loss_fn = KLDivLoss() # this is for debugging functionality independent of the custom loss functions
# learning_rate = 1e-3

# instantiate the adam optimizer with predefined learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# write a function that trains and returns a model to make it easier to compare results
training_losses = []
accuracy_scores = []
f1_scores = []
for epoch in range(num_epochs):

    epoch_losses = []
    epoch_accs = []
    epoch_f1 = []
    optimizer.zero_grad()

    for batch_number, batch in enumerate(dataloader):
        batch_xs = Variable(batch[0].float())
        batch_ys = Variable(batch[1].float())

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(batch_xs)

        # Compute loss.
        loss = loss_fn(y_pred, batch_ys)

        epoch_losses.append(loss.data.numpy())

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()


    print("Epoch: {} \t Loss: {} \t Accuracy: {} \t F1-Score: {}".format(epoch, np.mean(epoch_losses),
                    accuracy_score(np.argmax(y_pred.data.numpy(), axis=1),
                    np.argmax(batch_ys.data.numpy(),axis=1)),
                    f1_score(np.argmax(y_pred.data.numpy(),axis=1),
                    np.argmax(batch_ys.data.numpy(),axis=1))))
    epoch_accs.append(accuracy_score(np.argmax(y_pred.data.numpy(), axis=1),
                                                             np.argmax(batch_ys.data.numpy(),axis=1)))
    epoch_f1.append(f1_score(np.argmax(y_pred.data.numpy(),axis=1),
                    np.argmax(batch_ys.data.numpy(),axis=1)))
    training_losses.append(np.mean(epoch_losses))
    accuracy_scores.append(np.mean(epoch_accs))
    f1_scores.append(np.mean(epoch_f1))

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

# set vertical axis bounds at [0,1]
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(range(len(training_losses)),training_losses)
axarr[0].set_ylabel("loss")
axarr[1].plot(range(len(accuracy_scores)), accuracy_scores)
axarr[1].set_ylabel("accuracy")
axarr[2].plot(range(len(f1_scores)), f1_scores)
axarr[2].set_ylabel("f1")
plt.savefig(simulation_name)