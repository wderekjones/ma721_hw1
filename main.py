# built using example code from pytorch getting started with examples documentation http://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
np.random.RandomState(0)
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
from tensorboardX import SummaryWriter
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import multiprocessing
torch.manual_seed(0)
#using torch_extras may affect performance as it is not supported by anaconda
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn import KLDivLoss
from input_pipeline import parse_features
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split

from input_pipeline import KinaseDataset
from sklearn.preprocessing import OneHotEncoder
from losses import emd_loss, cramer_loss
from model import Net

# TODO: add argument for number of cores
# TODO: why isnt num_epochs updating correctly?
# TODO: save the models, implement saving the best so far
# TODO: fix the random number seeding...
# TODO: generate files that contain hyper-parameter values and or attach these to the generated figures
# TODO: keep track of training times
# TODO: find and implement at least one initialization method
# TODO: try different proteins and then the full set...perhaps this could be a function?
# TODO: add scalars for accuracy, precision, recall, f1-score, for the summary writer
# TODO: implement hyper-parameter optimization...
# TODO: write all hyperparameter values to the output figure
# TODO: plot distribution of probs for each class...should see skew away from .5
# TODO: train/test splits for to look at generalization performance
# TODO: look at performance with varying degrees of oversampling
# TODO: add argument for sampling technique, at least 2 options for comparison against no oversampling...this will need to be implemented in the dataloader
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to data")
parser.add_argument("--feats", type=str, help="path to features")
parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
parser.add_argument("--loss", type=str, help="loss function to use for training (emd or cramer). default is MSE.", default="mse")
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--batch_sz", type=int, help="batch size", default=1)
parser.add_argument("--np", type=int, help="number of processes", default=1)
parser.add_argument("--null", type=str, help="path to null features")
# parser.add_argument("--o", type=str, help="output name")


# write functions to do these initializations seperately for weights and biases, write tests for ill-conditioning
# for child in model.children():
#     for param in child.named_parameters():
#         torch.nn.init.uniform(param[1],9, 5)

# write a function that trains and returns a model to make it easier to compare results
# training_losses = []
# accuracy_scores = []
# f1_scores = []
# precision_scores = []
# recall_scores = []
#
# start_clock = time.clock()
#
# writer = SummaryWriter()
# for epoch in range(num_epochs):

    # epoch_losses = []
    # epoch_accs = []
    # epoch_f1s = []
    # epoch_precisions = []
    # epoch_recalls = []
    # optimizer.zero_grad()

    # for batch_number, batch in enumerate(dataloader):
        # batch doesn't need to be wrapped in a variable
        # batch_xs = Variable(batch[0].float())
        # batch_ys = Variable(batch[1].float())

        #need to find a more efficient way to do this...
        # batch_ys.data = torch.from_numpy(OneHotEncoder(sparse=False).fit_transform(batch_ys.data.numpy())).float()
        # Forward pass: compute predicted y by passing x to the model.

        # y_pred_probs = model(batch_xs)
        # y_pred = np.argmax(y_pred_probs.data.numpy(),axis=1)

        # Compute loss.
        # loss = loss_fn(y_pred_probs, batch_ys)

        # epoch_losses.append(loss.data.numpy())
        # epoch_accs.append(accuracy_score(np.argmax(batch_ys.data.numpy(),axis=1), y_pred))
        # epoch_f1s.append(f1_score(np.argmax(batch_ys.data.numpy(),axis=1), y_pred))
        # epoch_precisions.append(precision_score(np.argmax(batch_ys.data.numpy(),axis=1),y_pred))
        # epoch_recalls.append(recall_score(np.argmax(batch_ys.data.numpy(), axis=1), y_pred))

        # Backward pass: compute gradient of the loss with respect to model parameters
        # loss.backward()



            # training_losses.append(np.mean(epoch_losses))
    # accuracy_scores.append(np.mean(epoch_accs))
    # f1_scores.append(np.mean(epoch_f1s))
    # precision_scores.append(np.mean(epoch_precisions))
    # recall_scores.append(np.mean(epoch_recalls))
    # print("Epoch: {} \t Loss: {} \t Accuracy: {} \t F1-Score: {} \t Precision: {} \t Recall: {}".format(epoch, np.mean(epoch_losses),
    #                                                                      np.mean(epoch_accs), np.mean(epoch_f1s),
    #                                                                                       np.mean(epoch_precisions),
    #                                                                                             np.mean(epoch_recalls)))

#     # Calling the step function on an Optimizer makes an update to its
#     # parameters
#     optimizer.step()

# stop_clock = time.clock()


# print()
# print("Train time: ", (stop_clock-start_clock), " cpu seconds.")
# print("Test set performance:")

# test_y_probs = model(Variable(torch.from_numpy(data[test_idxs][0])))
# test_y_preds = np.argmax(test_y_probs.data.numpy(), axis=1)

# y_test = data[test_idxs][1]

# print("Accuracy: {} \t F1-Score: {} \t Precision: {} \t Recall: {}".format(accuracy_score(y_test,test_y_preds),f1_score(y_test,test_y_preds),
#                                                                            precision_score(y_test,test_y_preds),
#                                                                            recall_score(y_test,test_y_preds)))

# TODO: set vertical axis bounds at [0,1]
# plt.clf()
# plt.figure()
# f, axarr = plt.subplots(5, sharex=True)
# axarr[0].plot(range(len(training_losses)),training_losses)
# axarr[0].set_ylabel("loss")
# axarr[0].set_title("loss: {}  lr: {}  batch_sz: {}".format(args.loss, args.lr, args.batch_sz))
# axarr[1].plot(range(len(accuracy_scores)), accuracy_scores)
# axarr[1].set_ylabel("accuracy")
# axarr[2].plot(range(len(f1_scores)), f1_scores)
# axarr[2].set_ylabel("f1")
# axarr[3].plot(range(len(precision_scores)),precision_scores)
# axarr[3].set_ylabel("precision")
# axarr[4].plot(range(len(recall_scores)), recall_scores)
# axarr[4].set_ylabel("recall")
# plt.savefig("figures/"+simulation_name +"_"+str(time.time())+".png")


def train(model, dataloader, optimizer, epoch, queue):
    # optimizer.zero_grad()
    losses = []
    precisions = []
    f1s = []
    recalls = []
    accs = []
    for batch_number, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # batch doesn't need to be wrapped in a variable
        batch_xs = Variable(batch[0].float())
        batch_ys = Variable(batch[1].float())

        # convert the labels to one hot encoding to form a probability distribution so that the loss can be computed
        batch_ys.data = torch.from_numpy(OneHotEncoder(sparse=False).fit_transform(batch_ys.data.numpy())).float()

        # Forward pass: compute output of the network by passing x through the model.
        y_pred_probs = model(batch_xs)

        # Compute loss.
        loss = loss_fn(y_pred_probs, batch_ys)
        losses.append(loss.data.numpy())
        precisions.append(precision_score(batch_ys.data,np.argmax(y_pred_probs.data, axis=1)))
        f1s.append(f1_score(batch_ys.data, np.argmax(y_pred_probs.data, axis=1)))
        recalls.append(recall_score(batch_ys.data, np.argmax(y_pred_probs.data, axis=1)))
        accs.append(accuracy_score(batch_ys.data, np.argmax(y_pred_probs.data, axis=1)))
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()
    print("epoch: {} \t loss: {}".format(epoch,np.mean(losses)))
    accuracy_score(y_test, test_y_preds), f1_score(y_test, test_y_preds)
    precision_score(y_test, test_y_preds)
    recall_score(y_test, test_y_preds)
    # try putting a dictionary of loss and metrics instead to get multiple scalars to write to tensorboard

    queue.put({"loss": np.mean(losses), "precision": np.mean(precisions), "recall": np.mean(recalls), "f1": np.mean(f1s),
               "accuracy":np.mean(accs)})


if __name__ == '__main__':
    args = parser.parse_args()
    writer = SummaryWriter("test/"+str(time.time())+"/")

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    # n_bins is the number of class bins
    # num_epochs is the number of complete iterations over the data in batch_size increments

    # num_epochs = args.epochs
    num_epochs=100
    batch_size = args.batch_sz
    learning_rate = args.lr
    # look at loss performance by varying number of bins
    n_bins = 2
    # input_shape = 92
    input_shape = 5410
    N, D_in, H, D_out = batch_size, input_shape, 5, n_bins

    features_list = parse_features(args.feats,null_path=args.null)
    data = KinaseDataset(data_path=args.data,split="train",protein_name_list=["lck"], features_list=features_list)
    # data = KinaseDataset(data_path=args.data,split="train",protein_name_list=None, features_list=features_list)

    num_workers = 1
    idxs = np.arange(0, len(data))
    train_idxs, test_idxs = train_test_split(idxs)
    dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, sampler=train_idxs)
    num_processes = multiprocessing.cpu_count() - 2
    model = Net(D_in=D_in, H=H, D_out=D_out, N=N)
    if args.gpu is not None and args.gpu is 1:
        model.cuda()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    loss_fn = torch.nn.MSELoss()
    simulation_name = "mse"

    if args.loss == "cramer":
        loss_fn = cramer_loss()
        simulation_name = "cramer"

    elif args.loss == "emd":
        loss_fn = emd_loss()
        simulation_name = "emd"

    elif args.loss == "mse":
        loss_fn = torch.nn.MSELoss()
        simulation_name = "mse"

    print("loss function: {} \t lr: {} \t batch-sz: {}".format(args.loss, args.lr, args.batch_sz))
    # instantiate the adam optimizer with predefined learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_clock = time.clock()
    epoch=0
    q = Queue()
    # for step in range(int(np.ceil(num_epochs/num_processes))):
    for step in range(num_epochs):
        epoch_loss = 0
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(model, dataloader, optimizer, epoch,q))
            p.start()

            writer.add_scalar("loss",float(q.get()),epoch)
            epoch+=1

            processes.append(p)

        for p in processes:
            p.join()

    stop_clock = time.clock()
    print()
    print("Train time: ", (stop_clock-start_clock), " cpu seconds.")
    print("Validation set performance:")
    test_y_probs = model(Variable(torch.from_numpy(data[test_idxs][0])))
    test_y_preds = np.argmax(test_y_probs.data.numpy(), axis=1)
    y_test = data[test_idxs][1]
    print("Accuracy: {} \t F1-Score: {} \t Precision: {} \t Recall: {}".format(accuracy_score(y_test,test_y_preds),f1_score(y_test,test_y_preds),
                                                                               precision_score(y_test,test_y_preds),
                                                                               recall_score(y_test,test_y_preds)))
    # write a unique file for each run so that scalars can be collected and graphs can be created for the writeup
    # writer.export_scalars_to_json("all_scalars.json")
    writer.close()


    print()
    print("performance on testing set: ")
    data = KinaseDataset(data_path=args.data,split="test",protein_name_list=["lck"], features_list=features_list)
    test_y_probs = model(Variable(torch.from_numpy(data.data)))
    test_y_preds = np.argmax(test_y_probs.data.numpy(),axis=1)
    y_test = data.labels

    print("Accuracy: {} \t F1-Score: {} \t Precision: {} \t Recall: {}".format(accuracy_score(y_test,test_y_preds),f1_score(y_test,test_y_preds),
                                                                               precision_score(y_test,test_y_preds),
                                                                               recall_score(y_test,test_y_preds)))