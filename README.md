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