import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, activation_fn=nn.ReLU(), weight_decay=0.0001):
        """
        Custom VGG16 model with pluggable activation function.

        Args:
            activation_fn (nn.Module): PyTorch activation function (e.g., nn.ReLU()).
            weight_decay (float): Optional weight decay (not used here directly).
        """
        super(VGG16, self).__init__()
        self.activation = activation_fn

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.3)

        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.4)

        # Fourth convolutional block
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # First block
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Second block
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Third block
        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Fourth block
        x = self.activation(self.bn7(self.conv7(x)))
        x = self.activation(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.drop4(x)

        # Fully connected layers
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x



