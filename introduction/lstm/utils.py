import numpy as np


# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(
        6 / (input_size + output_size)
    )


# Activation Functions
def sigmoid(input, derivative=False):
    if derivative:
        return input * (1 - input)

    return 1 / (1 + np.exp(-input))


def tanh(input, derivative=False):
    if derivative:
        return 1 - input**2

    return np.tanh(input)


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))
