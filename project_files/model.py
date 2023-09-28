# Definition of the Pytorch model

import torch
import torch.nn as nn
from torch.nn import Sequential

class CNN(nn.Module):
    def __init__(self, in_channels: int = 3, kernel_sizes: list = [(3, 3), (3, 3), (3, 3)], classes: int = 10):
        super(CNN, self).__init__()

        # Maybe make into parameters?...
        self.conv_1_layers = 16
        self.conv_2_layers = 32
        self.conv_3_layers = 64
        self.fc1_units = 512
        self.linear_input_factor = 20 # This will vary based on the data passed, set it somehow later!

        # TODO Change ReLu to... literally anything else.
        self.conv_layers = Sequential(
            nn.Conv2d(in_channels, self.conv_1_layers, kernel_sizes[0]),
            nn.BatchNorm2d(self.conv_1_layers),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Adding MaxPooling for down-sampling. AvgPool sounds cool but usually it's irrelevant.
            nn.Conv2d(self.conv_1_layers, self.conv_2_layers, kernel_sizes[1]),
            nn.BatchNorm2d(self.conv_2_layers),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.conv_2_layers, self.conv_3_layers, kernel_sizes[2]),
            nn.BatchNorm2d(self.conv_3_layers),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.conv_3_layers * self.linear_input_factor * self.linear_input_factor, self.fc1_units), # TODO check if this holds?..
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc1_units, classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
