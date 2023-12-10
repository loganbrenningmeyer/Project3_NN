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

    def __init__(self, D_in, H_layers, D_out, batch_norm=False, dropout_rate=0.0, activation="sigmoid"):
        super(MLP, self).__init__()

        # Activation function mapping
        activation_functions = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU()
        }

        # Create the layers
        layers = []

        # Inputs to first hidden layer
        layers.append(nn.Linear(D_in, H_layers[0]))
        # Batch norm
        if batch_norm:
            layers.append(nn.BatchNorm1d(H_layers[0]))
        # Activation
        layers.append(activation_functions[activation])
        # Dropout
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(H_layers)-1):
            # Hidden layer
            layers.append(nn.Linear(H_layers[i], H_layers[i+1]))
            # Batch norm
            if batch_norm:
                layers.append(nn.BatchNorm1d(H_layers[i+1]))
            # Activation
            layers.append(activation_functions[activation])
            # Dropout
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(H_layers[-1], D_out))

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Create a function to train the MLP
def train(model, train_loader, optimizer, criterion):
    # Set model in training mode
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Get the inputs and labels
        x, y = batch
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x)
        # Compute the loss
        loss = criterion(y_pred, y)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Accumulate this batches loss (avg loss * batch size)
        total_loss += loss.item() * x.size(0)
    # Return the average loss
    return total_loss / len(train_loader.dataset)

# Create a function to test the MLP
def test(model, test_loader, criterion):
    # Set model in evaluation mode
    model.eval()
    total_loss = 0
    # Don't compute gradients
    with torch.no_grad():
        for batch in test_loader:
            # Get the inputs and labels
            x, y = batch
            # Forward pass
            y_pred = model(x)
            # Compute the loss
            loss = criterion(y_pred, y)
            # Accumulate this batches loss (avg loss * batch size)
            total_loss += loss.item() * x.size(0)
    # Return the average loss
    return total_loss / len(test_loader.dataset)

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
'''
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
# Convert to torch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
'''
# Load MNIST data set
from keras.datasets import mnist
from keras.datasets import fashion_mnist

# Load the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Convert to torch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# Reshape the data
X_train = X_train.view(X_train.size(0), -1)
X_test = X_test.view(X_test.size(0), -1)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create TensorDatasets for train and test data
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders for train and test data
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Split the data
#X_train, y_train, X_test, y_test = split_data(X, y, 0.8)

# Print the data
# print('X_train: {}'.format(X_train))
# print('y_train: {}'.format(y_train))

configurations = {"activation": ["sigmoid", "tanh", "relu", "leaky_relu", "elu"],
                  "batch_norm": [False, True],
                  "regularization": [None, "l2", "dropout"],
                  "loss": ["cross_entropy"],
                  "scaling": [None, "normalization", "standardization", "min_max"],
                  "momentum": [0.0, 0.9],
                  "optimizer": ["sgd", "adam", "rmsprop"],
                  "dropout_rate": [0.0, 0.2, 0.5]}

# Create a model
model = MLP(784, [300, 100], 10, batch_norm=True, dropout_rate=0.3, activation="relu")

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# Train the model
epochs = 20
prev_loss = 0
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = test(model, test_loader, criterion)
    print('Epoch {}: train loss: {}, test loss: {}'.format(epoch, train_loss, test_loss))

true_labels = []
pred_labels = []

# Test the model for confusion matrix
with torch.no_grad():
    for batch in test_loader:
        # Get the inputs and labels
        x, y = batch
        # Forward pass
        y_pred = model(x)
        # Get the predicted labels
        _, predicted = torch.max(y_pred.data, 1)
        # Append the true and predicted labels
        true_labels += y.tolist()
        pred_labels += predicted.tolist()

# Create a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print F1 score
from sklearn.metrics import f1_score
print(f"F1 Score: {f1_score(true_labels, pred_labels, average='macro')}")

# Print accuracy
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(true_labels, pred_labels)}")

'''
# Test all configurations
for activation in configurations["activation"]:
    for batch_norm in configurations["batch_norm"]:
        for regularization in configurations["regularization"]:
            for loss_function in configurations["loss"]:
                for scaling in configurations["scaling"]:
                    for momentum in configurations["momentum"]:
                        for optimizer_function in configurations["optimizer"]:
                            for dropout_rate in configurations["dropout_rate"]:
                                # Create a model
                                model = MLP(4, [5], 3, batch_norm=batch_norm, dropout_rate=dropout_rate, activation=activation)

                                # Create an optimizer
                                if optimizer_function == "sgd":
                                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum)
                                elif optimizer_function == "adam":
                                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                                elif optimizer_function == "rmsprop":
                                    optimizer = optim.RMSprop(model.parameters(), lr=0.01, momentum=momentum)

                                # Create a loss function
                                if loss_function == "mse":
                                    criterion = nn.MSELoss()
                                elif loss_function == "cross_entropy":
                                    criterion = nn.CrossEntropyLoss()

                                # Train the model
                                epoch = 0
                                prev_loss = 0
                                while(True):
                                    # Train
                                    loss = train(model, X_train, y_train, optimizer, criterion)
                                    # Print loss
                                    if epoch % 2000 == 0:
                                        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

                                    # Check for convergence
                                    if abs(loss.item() - prev_loss) < 0.000001:
                                        print(f"Converged: {abs(loss.item() - prev_loss)}")
                                        print(f"Loss: {loss.item()}")
                                        print(f"Previous Loss: {prev_loss}")
                                        print(f"Epoch: {epoch}")
                                        break
                                    else:
                                        prev_loss = loss.item()

                                    epoch += 1

                                # Test the model
                                test_loss = test(model, X_test, y_test, criterion)
                                print("Activation: {activation}, Batch Norm: {batch_norm}, Regularization: {regularization}, Loss: {loss_function}, Scaling: {scaling}, Momentum: {momentum}, Optimizer: {optimizer_function}, Dropout Rate: {dropout_rate}".format(activation=activation, batch_norm=batch_norm, regularization=regularization, loss_function=loss_function, scaling=scaling, momentum=momentum, optimizer_function=optimizer_function, dropout_rate=dropout_rate))
                                print('Test loss: {}'.format(test_loss.item()))
'''
# Plot the data 
# plot_data(X_train, y_train)

# Test plotting
# plt.show()