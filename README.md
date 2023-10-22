# Deep Learning - Project

## Team Name
Model Mavericks

## Name and neptun
* Lakatos Bálint Gábor, T04ZVM
* Hegedűs András, BCFU8E
* Sárossy János, GUSXLY

## Task
The goal of this project is to **compare and demonstrate the advantages of using pretrained neural networks** vs randomly initialized ones for image classification. Build an image classification pipeline for a smaller dataset (e.g. CIFAR-10), and train both a randomly initialized and an ImageNet-pretrained network (e.g. ResNet) for the task. Compare their performance.

## Materials
Related materials:
* PyTorch implementation of a CIFAR-10 baseline model: [link to tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)
* **Deep Residual Learning for Image Recognition:** [link to paper](https://arxiv.org/abs/1512.03385)

* **Difficulty: Medium**

# Tasks:
- [x] Setup Project
- [x] Setup git
- [x] Setup figma
- [x] Setup Docker, envs

- [ ] Transfer Learning handle
- [x] Dataset (CIFAR?) handle

- [ ] Implement a basic training loop for a CNN on CIFAR-10
- [ ] Implement Hyperparameter Search
- [ ] Implement Visualization, metrics for models

# Files & structure
* project_files: all the necessary files to run the project
    * saved_models: directory to store the saved models
    * tensorboard_runs: directory to store tensorboard_runs
    * utils: directory to store python utilities related to the project
        * logging.py: (in progress) logging mechanisms should be implemented here
        * presistance.py: For classes and methods dealing with persisting items.
        * py_utils.py: useful utility functions (eg. check if GPU is available)
    * data_handling.py: data visualisation functionality
    * model.py: model definition
    * training.py: training related functionality
* docker-compose.yml: docker compose file to build and run the specified docker image needed to run the project
* Dockerfile: dockerfile to run the project in
* entrypoint.py: entrypoint for the project
* README.md: this document
* requirements.txt: list of required packages to run the project
* run_docker_commands.txt: necessary commands to run the project from a terminal in the predefined docker
* milestone1.ipynb: python notebook demo for the first milestone

# Notes
* Do not clutter up .gitignore if possible!
* Use comments, return typing, parameter typing for ease of developement.
* Try to build and keep an orderly project.
