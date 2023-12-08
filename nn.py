# include our Python packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Make an MLP
class MLP(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        h_pred = F.relu(self.linear1(x))
        y_pred = self.linear2(h_pred)
        return y_pred

# Create a function to train the MLP
def train(model, x, y, optimizer, criterion):
    # Reset gradient
    optimizer.zero_grad()
    # Forward
    y_pred = model(x)
    # Loss
    loss = criterion(y_pred, y)
    # Backward
    loss.backward()
    # Update parameters
    optimizer.step()
    return loss\\