import torch
import torchvision
import torch.nn as nn

class CNNModel(nn.Module):
    """
     A convolutional neural network (CNN) with a residual connection 
    for image classification.

    Attributes:
        conv1 (nn.Sequential): First convolutional layer block.
        conv2 (nn.Sequential): Second convolutional layer block.
        conv3 (nn.Sequential): Third convolutional layer block.
        conv4 (nn.Sequential): Fourth convolutional layer block.
        residual (nn.Sequential): 1x1 convolution for the residual connection.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Output/Last layer.
        dropout (nn.Dropout): Dropout layer with probability 0.5.
    """
    def __init__(self):
        """
        Initializes the CNN model with convolutional layers, residual connection, 
        fully connected layers, and activation functions.

        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.residual = nn.Sequential(
            nn.MaxPool2d(kernel_size=8),
            nn.Conv2d(3, 256, kernel_size=1))
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128, 3)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, X):
        """
        Defines the forward pass of the CNN model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, 3, 64, 64).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        residual = self.residual(X)

        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)

        X += residual[:, :, ::2, ::2]

        X = self.flatten(X)
        X = self.dropout(self.relu(self.fc1(X)))
        X = self.dropout(self.relu(self.fc2(X)))
        X = self.dropout(self.relu(self.fc3(X)))
        X = self.fc4(X)
        return X
