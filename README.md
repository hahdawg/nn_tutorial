# Pytorch Tutorial

## Overview
In this tutorial, we'll cover the three main components of setting up a pytorch model.

1. Batching process: For datasets where loading a single batch takes more than a few milliseconds, torch's
DataSet and DataLoader tools are very useful. But for this tutorial, they're unnecessary.
2. Model definition: Use torch to define the model architecture.
3. Training: Use torch to train the model.

## Installation
The `setup.py file` lists the requriements needed to run this tutorial. If you don't already have
torch >= 1.8.0 installed, you'll want to install it (preferably in a virtual environment) using the
instructions from the [official torch website](https://pytorch.org/).

Once you have all the dependencies installed, run
```
pip install -e . --user --no-deps
```
in the project's root directory.

## Running the Model
The notebook `training-example.ipynb` contains an example of how to load data and train the model.

# Pytorch Basics

## Three Levels of Abstraction
1. Tensor: Basically a numpy array that can live on GPUs; e.g., the features you input into a neural network.
2. Variable: Node in a computational graph; e.g., the weights of a neural network.
    * Contains values (`data`) and gradients (`grad`)
    * Note: The `data` property of a Variable is a Tensor
3. Module: Neural network layer; e.g., affine (linear), RELU, Conv2D
    * Implements a forward and backward method
    * If module is composed of other torch modules, there's no need to implement the backward method.
    
## Autograd
Pytorch will calculate gradients for you.
    
## High Level APIs
1. torch.nn: Consists of layers (e.g., linear, transformer, LSTM) commonly used in deep learning models.
2. torch.optim: Contains optimizers (e.g., ADAM, SGD) that perform gradient update steps for models.

## DataSet and DataLoader
Useful for creating batch generators.