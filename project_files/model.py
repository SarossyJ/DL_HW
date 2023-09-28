# Definition of the Pytorch model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)