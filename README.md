# Artificial Neural Network in CPP For Linear Regression

This project implements a neural network (ANN) model using C++ (including OOps concepts) to solve a regression problem (eg. [Pune House Rent Prediction](https://www.kaggle.com/datasets/rahulmishra5/pune-house-rent-prediction)). The code reads a dataset from a csv file, initializes network parameters, performs forward propagation, calculates errors, and updates parameters to minimize the loss function over multiple epochs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Dataset](#dataset)
- [Functions](#functions)

## Installation

To compile and run the code, ensure you have a C++ compiler installed on your system. The code also requires the `data_file.csv` file to be present in the same directory.

### Compilation
```sh
g++ main.cpp Dataset.cpp ANN.cpp -o my_program
```

### Running
```sh
./my_program
```

## Usage

The program reads a dataset from `data_file.csv` file, initializes the network parameters, trains the network over a given number of epochs, and prints the loss at each epoch.

## Code Explanation

### Dataset
The dataset should be in a file named `file_name.csv` with the following structure:

```
3,1.2,1,5.0
2,0.8,2,3.5
4,1.5,1,6.0
...
```

### Initialization
The network can be initialized with a given (user-defined) layer structure., for example:
```cpp
vector<int> layer_dims = {3, 2, 1};
```
This indicates:
- Input layer with 3 neurons
- One hidden layer with 2 neurons
- Output layer with 1 neuron

### Training
The network can be trained for given (user-defined) number of epochs (iterations over the dataset):
```cpp
int epochs = 20;
```
During each epoch, the network performs forward propagation to predict the rent, calculates the error, and updates the weights and biases to minimize the error.

## Functions

### `initialize`
Initializes a 2D vector (matrix) with specified rows, columns, and initial value.

### `initialize_parameters`
Initializes the weights and biases for each layer in the network.

### `linear_forward`
Performs the forward propagation for a single layer.

### `L_layer_forward`
Performs the forward propagation through all layers.

### `update_parameters`
Updates the weights and biases based on the calculated error.

## Example Output
```
Epoch:1, Loss: 2.74798
Epoch:2, Loss: 2.42892
Epoch:3, Loss: 2.10553
Epoch:4, Loss: 1.78152
Epoch:5, Loss: 1.46464
Epoch:6, Loss: 1.16569
Epoch:7, Loss: 0.89631
Epoch:8, Loss: 0.666201
Epoch:9, Loss: 0.480631
Epoch:10, Loss: 0.339477
Epoch:11, Loss: 0.237992
Epoch:12, Loss: 0.168721
Epoch:13, Loss: 0.123571
Epoch:14, Loss: 0.0952916
Epoch:15, Loss: 0.0781646
Epoch:16, Loss: 0.0680803
Epoch:17, Loss: 0.0622815
Epoch:18, Loss: 0.0590135
Epoch:19, Loss: 0.0572036
Epoch:20, Loss: 0.0562161
...
```

This output shows the loss (Mean Squared Error) at each epoch, indicating the training progress.
