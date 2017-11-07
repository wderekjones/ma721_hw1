# built using example code from pytorch getting started with examples documentation http://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import os
seed = 0
np.random.RandomState(seed)
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
from tensorboardX import SummaryWriter
import torch
import multiprocessing
torch.manual_seed(seed)
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from input_pipeline import KinaseDataset, parse_features
from model import Net
from utils import init_network

# need to add validation metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to data")
parser.add_argument("--feats", type=str, help="path to features")
parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--batch_sz", type=int, help="batch size", default=1)
parser.add_argument("--null", type=str, help="path to null features")
parser.add_argument("--ncores", type=int, help="number of cores to use for multiprocessing of training", default=multiprocessing.cpu_count())
parser.add_argument("--oversample", type=int, help="whether to oversample the minority class", default=1)


def train(model, dataloader, validation_data, optimizer, epoch):
    train_losses = []
    train_precisions = []
    train_f1s = []
    train_recalls = []
    train_accs = []

    start_train_clock = time.clock()
    for batch_number, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_xs = Variable(batch[0].float().cuda(), requires_grad=False)
        batch_ys = Variable(batch[1].long().cuda(), requires_grad=False)

        # Forward pass: compute output of the network by passing x through the model.
        y_pred_probs = model(batch_xs)

        y_pred = np.argmax(y_pred_probs.data.cpu().numpy(),axis=1)
        y_test = np.argmax(batch_ys.data.cpu().numpy(),axis=1)

        # Compute loss.
        train_loss = loss_fn(y_pred_probs, batch_ys.float())
        train_losses.append(train_loss.cpu().data.numpy())
        train_precisions.append(precision_score(y_test,y_pred))
        train_f1s.append(f1_score(y_test, y_pred))
        train_recalls.append(recall_score(y_test, y_pred))
        train_accs.append(accuracy_score(y_test, y_pred))
        # Backward pass: compute gradient of the loss with respect to model parameters
        train_loss.backward()
        optimizer.step()
    stop_train_clock = time.clock()
    # print("epoch: {} \t train_loss: {} \t train_accuracy: {} \t train_precision: {} \t train_recall: {} \t train_f1: {}".format(epoch,
    #                                                                                               np.mean(train_losses),
    #                                                                                               np.mean(train_accs),
    #                                                                                               np.mean(train_precisions),
    #                                                                                               np.mean(train_recalls),
    #                                                                                               np.mean(train_f1s)))

    # witch the model to evaluation mode (training=False) in order to evaluate on the validation set
    model.eval()

    val_losses = []
    val_precisions = []
    val_f1s = []
    val_recalls = []
    val_accs = []

    start_val_clock = time.clock()
    for val_batch_number, val_batch in enumerate(val_dataloader):
        val_xs = Variable(val_batch[0].float().cuda(), requires_grad=False)
        val_ys = Variable(val_batch[1].long().cuda(), requires_grad=False)

        # Forward pass: compute output of the network by passing x through the model.
        val_y_pred_probs = model(val_xs)

        val_y_pred = np.argmax(val_y_pred_probs.data.cpu().numpy(),axis=1)
        val_y_test = np.argmax(val_ys.data.cpu().numpy(),axis=1)

        # Compute loss.
        val_loss = loss_fn(val_y_pred_probs, val_ys.float())
        val_losses.append(val_loss.cpu().data.numpy())
        val_precisions.append(precision_score(val_y_test, val_y_pred))
        val_f1s.append(f1_score(val_y_test, val_y_pred))
        val_recalls.append(recall_score(val_y_test, val_y_pred))
        val_accs.append(accuracy_score(val_y_test, val_y_pred))
    stop_val_clock = time.clock()


    print("epoch: {} \t val_loss: {} \t val_accuracy: {} \t val_precision: {} \t val_recall: {} \t val_f1: {}".format(epoch,
                                                                                                    np.mean(val_losses),
                                                                                                    np.mean(val_accs),
                                                                                                    np.mean(val_precisions),
                                                                                                    np.mean(val_recalls),
                                                                                                    np.mean(val_f1s)))
    # put the model back into training mode so that it is ready to be updated during future epochs
    model.train()
    # return a tuple of dictionary objects containing the training and validation metrics respectively
    return ({"train_loss": np.mean(train_losses), "train_precision": np.mean(train_precisions), "train_recall": np.mean(train_recalls), "train_f1": np.mean(train_f1s),
               "train_accuracy": np.mean(train_accs), "train_time": (stop_train_clock-start_train_clock)},
        {"val_loss": np.mean(val_losses), "val_precision": np.mean(val_precisions), "val_recall": np.mean(val_recalls), "val_f1": np.mean(val_f1s),
               "val_accuracy": np.mean(val_accs), "val_time": (stop_val_clock - start_val_clock)})


if __name__ == '__main__':
    args = parser.parse_args()
    time_stamp = time.time()
    writer = SummaryWriter("logs/"+str(time_stamp)+"/")

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    # n_bins is the number of class bins
    # num_epochs is the number of complete iterations over the data in batch_size increments

    num_epochs = args.epochs
    batch_size = args.batch_sz
    learning_rate = args.lr
    n_bins = 2
    num_workers = 1


    # get the list of features by loading the full feature list and removing the nulls
    features_list = parse_features(args.feats, null_path=args.null)

    # load the training data, then further partition into training and validation sets, preserving the ratio of
    # positives to negative training examples
    data = KinaseDataset(data_path=args.data, split="train", oversample=False, features_list=features_list,
                         protein_name_list=['lck'])
    idxs = np.arange(0, len(data))
    train_idxs, val_idxs = train_test_split(idxs, stratify=data.labels.numpy(), random_state=seed)
    train_dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, sampler=train_idxs)
    val_dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, sampler=val_idxs)

    # define the network dimensions based on input data dimensionality
    N, D_in, H, D_out = batch_size, data.data.shape[1], 5, n_bins

    # load the model
    model = Net(D_in=D_in, H=H, D_out=D_out, N=N, p=0.7)
    # model = init_network(model)
    model.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_name = "bce"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_clock = time.clock()
    epoch = 0

    print("data: {}".format(args.data))
    print("features: {}".format(args.feats))
    print("null features: {}".format(args.null))
    print("ncores: {}".format(args.ncores))
    print("oversample: {}".format(args.oversample))

    # print network description
    print("batch size: {} \t # hidden units: {} \t total # params: {}".format(args.batch_sz, model.get_n_hidden_units(),
                                                                              model.get_n_params))
    # output optimization details
    regularization = "dropout"
    initialization = "uniform"

    print("optimizer: {} \t lr: {} \t initialization: {} \t regularization: {}".format("adam", args.lr, initialization,
                                                                                       regularization))
    #
    # train model
    #
    #

    for step in range(num_epochs):
        epoch_loss = 0

        train_dict, val_dict = train(model, train_dataloader, val_dataloader, optimizer, epoch)

        writer.add_scalar("train_loss", float(train_dict["train_loss"]), epoch)
        writer.add_scalar("train_accuracy", float(train_dict["train_accuracy"]), epoch)
        writer.add_scalar("train_precision", float(train_dict["train_precision"]), epoch)
        writer.add_scalar("train_recall", float(train_dict["train_recall"]), epoch)
        writer.add_scalar("train_f1", float(train_dict["train_f1"]), epoch)
        writer.add_scalar("train_time", float(train_dict["train_time"]), epoch)
        writer.add_scalar("val_loss", float(val_dict["val_loss"]), epoch)
        writer.add_scalar("val_accuracy", float(val_dict["val_accuracy"]), epoch)
        writer.add_scalar("val_precision", float(val_dict["val_precision"]), epoch)
        writer.add_scalar("val_recall", float(val_dict["val_recall"]), epoch)
        writer.add_scalar("val_f1", float(val_dict["val_f1"]), epoch)
        writer.add_scalar("val_time", float(val_dict["val_time"]), epoch)
        epoch += 1

    stop_clock = time.clock()
    print()
    print("Train time: ", (stop_clock-start_clock), " cpu seconds.")
    print()

    # serialize the scalar data to a .json
    scalar_path = "results/"+str(time_stamp)+"_all_scalars.json"
    if not os.path.exists("results/"):
        os.makedirs("results/")

    writer.export_scalars_to_json(scalar_path)
    writer.close()

    # pickle the model and save to a file
    model_path = "models/"+str(time_stamp)+"_saved_state.pkl"
    if not os.path.exists("models/"):
        os.makedirs("models/")
    torch.save(model.state_dict(), model_path)

    # evaluate on the testing data
    model.cpu().eval()
    data = KinaseDataset(data_path=args.data, oversample=False, split="test", features_list=features_list,
                         protein_name_list=["lck"])
    test_y_probs = model(Variable(data.data.float(), requires_grad=False))
    test_y_preds = np.argmax(test_y_probs.data.numpy(), axis=1)
    y_test = np.argmax(data.labels.numpy(), axis=1)

    print("Accuracy: {} \t F1-Score: {} \t Precision: {} \t Recall: {}".format(accuracy_score(y_test, test_y_preds),
                                                                               f1_score(y_test, test_y_preds),
                                                                               precision_score(y_test, test_y_preds),
                                                                               recall_score(y_test, test_y_preds)))
