# Functions to train a given model on a given dataset.
# TODO All of this should be customizable, e.g loos_function should be a parameter.
# This is so hyperparameter search and other tasks can easily be achieved.

from project_files.utils import logging
import tensorboard as tb

from project_files.data_handling import show_image
from project_files.model import CNN

def run_training():
    pass