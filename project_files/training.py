# Functions to train a given model on a given dataset.
# TODO All of this should be customizable, e.g loos_function should be a parameter.
# This is so hyperparameter search and other tasks can easily be achieved.

from project_files.utils import logging
import tensorboard as tb

import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

from project_files.data_handling import show_image, get_image_folder, get_CIFAR10_split
from project_files.model import CNN
from project_files.utils.py_utils import analyze, get_device

def run_training():
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    training_set, validation_set, test_set = get_CIFAR10_split(transform)
    num_of_classes = training_set.classes

    # Model init
    init_model = CNN(classes=num_of_classes)

    # Trained model
    print("BEGINNING OF TRAINING CYCLE")
    model = training_cycle()
    print("END OF TRAINING CYCLE")

    # Print metric
    accuracy = test_model(model)
    print("Accuracy of the trained model is {}.".format(accuracy))

def training_cycle(model, train_loader, val_loader, num_epochs, optimizer_fn, loss_fn, get_device):
    """
    Perform one full training cycle.
    """
    optimizer = optimizer_fn(model)
    device = get_device()
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validation_one_epoch(model, val_loader, loss_fn, device)
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}')
        val_losses.append(val_loss)

    return model


def training_cycle(model, train_loader, val_loader, test_loader, num_epochs, optimizer_fn, loss_fn):
    """
    One training cycle with the provided hyperparameters.
    """
    
    optimizer = optimizer_fn(model.parameters())
    device = get_device()
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validation_one_epoch(model, val_loader, loss_fn, device)

        print('Epoch {}: Train Loss = {:.4f}, Validation Loss = {:.4f}'.format(
            epoch+1,
            train_loss,
            val_loss
        ))

        val_losses.append(val_loss)

    test_model(model, test_loader, loss_fn, device)

# region epoch actions
# Functions dealing with 1 epoch / batch worth of tasks.
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Train the model for one epoch.

    Returns:
        float: The average loss for this epoch.
    """
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validation_one_epoch(model, dataloader, metrics, device):
    """
    Validate the model on a validation set.

    Returns:
        dict: Dictionary of computed metrics.
    """
    model.eval()
    metric_values = {metric_name: 0.0 for metric_name in metrics.keys()}

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            for metric_name, metric_fn in metrics.items():
                metric_values[metric_name] += metric_fn(outputs, targets)

    for metric_name in metrics.keys():
        metric_values[metric_name] /= len(dataloader)

    return metric_values

def test_model(model, test_loader, loss_fn, device):
    """
    Test the model on the test set.\n
    Only run once, at the end of all the training loops!

    Returns:
        float: The average loss on the test set.
    """
    device = device()
    test_loss = validation_one_epoch(model, test_loader, loss_fn, device)
    print('Test Loss = {:.4f}'.format(test_loss))
    return test_loss
# endregion
