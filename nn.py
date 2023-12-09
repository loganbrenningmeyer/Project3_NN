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



# Create a function to split the data into training and testing sets
def split_data(X, y, train_ratio):
    # Shuffle the data
    indices = torch.randperm(X.size(0))
    # Split the data
    train_indices = indices[:int(train_ratio*X.size(0))]
    test_indices = indices[int(train_ratio*X.size(0)):]
    # Create the sets
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test



# Create a function to plot the data
def plot_data(X, y):
    # Create a figure
    fig = plt.figure()
    # Plot the data
    plt.scatter(X[:,0], X[:,1], c=y)
    # Show the plot
    plt.show()

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
    return loss

# Create a function to test the MLP
def test(model, x, y, criterion):
    # Forward
    y_pred = model(x)
    # Loss
    loss = criterion(y_pred, y)
    return loss

# Create a function to predict the MLP
def predict(model, x):
    # Forward
    y_pred = model(x)
    return y_pred



"""
Implement the following techniques:

Activation functions:
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- ELU

Batch Normalization

Regularization:
- L1
- L2
- Dropout

Loss Functions:
- MSE
- Cross Entropy

Data Scaling:
- Normalization
- Standardization (mean/std)
- Min-Max Scaling

Momentum

Skip Connections

Thought: Could automate different configurations and compare
their accuracies and convergence times
"""
# Load iris data set
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
# Convert to torch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# Split the data
X_train, y_train, X_test, y_test = split_data(X, y, 0.8)

# Print the data
# print('X_train: {}'.format(X_train))
# print('y_train: {}'.format(y_train))

# Create a model
model = MLP(4, 10, 3)

# Create an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

# Create a loss function
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(10000):
    # Train
    loss = train(model, X_train, y_train, optimizer, criterion)
    # Print loss
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

# Test the model
test_loss = test(model, X_test, y_test, criterion)
print('Test loss: {}'.format(test_loss.item()))

# Plot the data 
plot_data(X_train, y_train)

# Test plotting
plt.savefig('test.png')