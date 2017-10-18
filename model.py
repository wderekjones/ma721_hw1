import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, D_in, H, D_out,N):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.N = N
        # noticed an issue with the result of the weight mult. by input giving nans...are the weights too large/small ?
        self.fc1 = nn.Linear(self.D_in, self.H)
        self.relu1 = nn.ReLU()
        # self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.H, self.D_out)
        self.relu2 = nn.Softplus()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.softmax(x)

        return x


def test_model(D_in=289, H=100, D_out=5, N=100):
    net = Net(D_in, H, D_out, N)
    print(net)
    print(net(torch.autograd.Variable(torch.zeros(N, D_in))))
    for parameter in net.parameters():
        print(parameter)

# test_model(D_in=100,H=50,D_out=2,N=100)