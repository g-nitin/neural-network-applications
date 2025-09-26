# Neural Network from Scratch

This project provides a simple, from-scratch implementation of a feedforward neural network using only NumPy. The implementation is designed to be flexible, supporting both one-layer and two-layer architectures for classification and function approximation tasks.

## Options

- **Multiple Activation Functions**: Includes implementations for:
  - `tanh` (Hyperbolic Tangent)
  - `logsig` (Logistic Sigmoid)
  - `tansig` (Tangent Sigmoid)
  - `radialbasis` (Radial Basis Function)
  - `relu` (Rectified Linear Unit)
  - `linear` (for output layers in regression tasks)
- **Weight Initialization**: Supports different initialization strategies:
  - `random`: Small random values
  - `xavier`: Xavier/Glorot initialization
  - `he`: He initialization
- **Training**: Implements mini-batch gradient descent with Mean Squared Error (MSE) as the loss function.
- **Data Scaling**: Includes a `Scaler` class for standardizing data (zero mean, unit variance).
- **Visualization**: Comes with plotting utilities to visualize:
  - Decision boundaries for classification problems.
  - Function approximations for regression problems.
  - Training error (loss) over epochs.

## File Descriptions

- **`NeuralNetwork.py`**: Contains the core `NeuralNetwork` class, which encapsulates the network's structure, forward and backward propagation, and training logic. It also includes the `Scaler` helper class.
- **`main.py`**: The main script for running experiments. It defines two primary experiments:
  1.  **Question 1**: A classification task using a one-layer network to separate 2D data points into different classes.
  2.  **Question 2**: A function approximation (regression) task using a two-layer network to model a 1D function. This script also includes a hyperparameter tuning loop to find the best model configuration.
- **`plots.py`**: A collection of utility functions for generating and saving plots using Matplotlib.
