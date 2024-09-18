# Artificial Neural Network in CPP For Linear Regression

This project implements a neural network (ANN) model using C++ (including OOps concepts) to solve a regression problem (eg. [Pune House Rent Prediction](https://www.kaggle.com/datasets/rahulmishra5/pune-house-rent-prediction)). The code reads a dataset from a csv file, initializes network parameters, performs forward propagation, calculates errors, and updates parameters to minimize the loss function over multiple epochs.
![image](https://github.com/user-attachments/assets/77f71c37-cac2-47e3-8afe-30e3f1c66af8)


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

## Important Functions

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

