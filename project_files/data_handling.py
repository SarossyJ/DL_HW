# Library dealing with the reading, manipulation and display of images.

import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from torchvision.transforms import ToTensor

from torch.utils.data import random_split

def get_image_folder(path:str='', transform=ToTensor):
    """
    Dummy function to hide ImageFolder here
    """
    return ImageFolder(path, transform)

def get_CIFAR10_split(train_transform, val_transform,valid_to_test_ratio = [.9, .1]):
    assert sum(valid_to_test_ratio) == 1, "Easier to give fractions that sum to 1. Ignore if not a problem."
    training_set =      CIFAR10('./data', train=True, transform=train_transform, download=True)
    validation_set =    CIFAR10('./data', train=False, transform=val_transform, download=True)

    numofval = int(len(validation_set) * valid_to_test_ratio[0])
    numoftest = len(validation_set) - numofval
    # Split validation_set into 2
    validation_set, test_set = random_split(validation_set, [numofval, numoftest])

    return training_set, validation_set, test_set




# Now, 'validation_set' contains the validation set, and 'test_set' contains the small test set


def show_image(img:Image=None) -> None:
    # TODO: Expand so tensor, pil, ndarray etc can be displayed without difficulty!
    """Primitive helper.

    Args:
        img (Image, optional): _description_. Defaults to None.
    """
    if(img):
        plt.imshow(img, cmap='gray')