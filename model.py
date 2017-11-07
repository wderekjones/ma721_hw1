import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, D_in, H, D_out, N, p):
        super(Net, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.N = N
        self.batch_norm1 = nn.BatchNorm1d(self.D_in)
        self.fc1 = nn.Linear(self.D_in, self.H)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p)
        self.batch_norm2 = nn.BatchNorm1d(self.H)
        self.fc2 = nn.Linear(self.H, self.D_out)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.batch_norm1(x) # make sure this is good place for batchnorm
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.batch_norm2(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

    def get_n_hidden_units(self):
        n_units = 0
        for child in self.children():
            for param in child.named_parameters():
                if 'weight' in param[0]:
                    n_units += param[1].data.size()[0]
                # elif 'bias' in param[0]:
        return n_units

    def get_n_params(self):
        n_params = self.get_n_hidden_units()
        for child in self.children():
            for param in child.named_parameters():
                if 'bias' in param[0]:
                    n_params += param[1].data.size()[0]
        return n_params


class deepNet(nn.Module):

    def __init__(self, D_in, H, D_out, N, p):
        super(deepNet, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.N = N
        self.batch_norm1 = nn.BatchNorm1d(self.D_in)
        self.fc1 = nn.Linear(self.D_in, self.H)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p)
        self.batch_norm2 = nn.BatchNorm1d(self.H)
        self.fc2 = nn.Linear(self.H, self.H)
        self.relu2 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(self.H)
        self.fc3 = nn.Linear(self.H, self.D_out)
        self.relu3 = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.batch_norm1(x)  # make sure this is good place for batchnorm
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.batch_norm2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.batch_norm3(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        x = self.softmax(x)
        return x

    def get_n_hidden_units(self):
        n_units = 0
        for child in self.children():
            for param in child.named_parameters():
                if 'weight' in param[0]:
                    n_units += param[1].data.size()[0]
                    # elif 'bias' in param[0]:
        return n_units

    def get_n_params(self):
        n_params = self.get_n_hidden_units()
        for child in self.children():
            for param in child.named_parameters():
                if 'bias' in param[0]:
                    n_params += param[1].data.size()[0]
        return n_params

