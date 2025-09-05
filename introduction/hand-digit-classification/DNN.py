import numpy as np


class DNN:
    """
    Simple fully-connected neural network.
    Source: https://medium.com/data-science/implementing-gradient-descent-example-102b4454ea67

    This implementation uses sigmoid activations, manual bias handling (appending 1), and a custom backpropagation loop. It's written for clarity, not performance.

    Input/weight shapes (high-level):
      - layers: list of layer sizes, e.g. [784, 1250, 10]
      - weights[i] shape: (layers[i+1], layers[i] + 1) -> +1 accounts for bias column
    """

    def __init__(self, layers):
        # layers: list of integers (sizes of each layer, input..output)
        self.layers = layers
        self.weights = []

        # Initialize weights randomly for each layer transition.
        # We add +1 to the input size to store the bias weight for that neuron.
        for i in range(len(layers) - 1):
            # rows = next layer size, cols = current layer size + 1 (bias)
            layers_weights = np.random.rand(layers[i + 1], layers[i] + 1)
            self.weights.append(layers_weights)

    def sigmoid(self, x):
        """
        Sigmoid activation.
        Note: inputs are scaled by 0.01 here which makes the function much more linear.
        """
        return 1 / (1 + np.exp(-0.01 * x))

    def predict(self, data):
        """Forward pass that returns the network's output vector for a single example.

        :param data: 1D numpy array with shape (input_dim,)
        :return: output activation (1D array) of the final layer.
        """
        x_s = [data]  # store activations per layer; x_s[0] is input (no bias yet)

        # Iterate through layers and compute activations
        for i in range(len(self.layers) - 1):
            # Add bias term to the current activation vector (append a constant 1).
            # This matches the +1 column in the weight matrices.
            x_s[-1] = np.concatenate((x_s[-1], [1]))

            # Compute z = W * x  (W shape: [next_size, cur_size+1])
            z = np.dot(self.weights[i], x_s[i])

            # Apply sigmoid and append to activations list.
            x_s.append(self.sigmoid(z))

        # x_s[-1] is the final layer activation (output probabilities/values)
        return x_s[-1]

    def train(self, data, y_true):
        """Single-step training using one example and its one-hot label vector.

        This performs a forward pass, computes output-layer deltas (psi),
        backpropagates those deltas through hidden layers, accumulates gradients,
        updates weights with a fixed learning rate (0.1), and returns the MSE.
        """
        x_s = [data]

        # Forward pass (same as predict) while keeping intermediate activations
        for i in range(len(self.layers) - 1):
            # add bias to activation before multiplying by weight matrix
            x_s[-1] = np.concatenate((x_s[-1], [1]))
            z = np.dot(self.weights[i], x_s[i])
            x_s.append(self.sigmoid(z))

        # Compute output-layer error signals (psi).
        # For each output neuron: psi = d(MSE)/d(output) * sigmoid'(z).
        # MSE derivative w.r.t output is -2*(y_true - output).
        psi = []
        for i in range(len(y_true)):
            output = x_s[-1][i]
            # output * (1 - output) is sigmoid'(z) assuming standard sigmoid;
            # here sigmoid was scaled but we still use this form for derivative.
            # it is intentionally left incorrect.
            psi.append(-2 * (y_true[i] - output) * (output * (1 - output)))
        psi = np.array(psi)
        # reshape to column vector: (n_outputs, 1)
        psi = np.reshape(psi, (psi.shape[0], 1))

        # gradients will store gradients for each layer (in back-to-front order)
        gradients = []
        # Gradient for last weight matrix: outer product of psi (n_out x 1) and
        # previous activations x_s[-2] (1 x (prev_size+1)) -> shape (n_out, prev_size+1).
        gradients.append(psi * x_s[-2])

        # Backpropagate through hidden layers (excluding input layer).
        for i in range(len(self.layers) - 2, 0, -1):
            # w = weights for layer i (from layer i to i+1), drop its bias column
            # because bias weights don't connect to previous activations directly.
            w = self.weights[i][:, :-1]

            # x = activation of layer i (without bias term); used in derivative
            x = x_s[i][:-1]

            # term computes elementwise product required to propagate delta:
            # term = w * x * (1 - x)  (broadcasting may apply depending on shapes).
            term = w * x * (1 - x)
            term = np.transpose(term)

            # Propagate psi backward: new psi = term dot psi
            psi = np.dot(term, psi)
            psi = np.reshape(psi, (psi.shape[0], 1))

            # Gradient for this layer's weight matrix (psi * activations of prev layer)
            gradients.append(psi * x_s[i - 1])

        # Update weights: gradients list was built from output backwards, so we
        # reverse-index gradients when applying to weights[0..]. Learning rate=0.1
        for i in range(len(gradients)):
            self.weights[i] -= 0.1 * gradients[-(i + 1)]

        # Return mean squared error for this example (sum of squared differences).
        return sum((y_true - x_s[-1]) ** 2)


class DNN_Corrected:
    """
    Simple fully-connected neural network.
    Source: https://medium.com/data-science/implementing-gradient-descent-example-102b4454ea67

    This implementation uses sigmoid activations, manual bias handling (appending 1), and a custom backpropagation loop. It's written for clarity, not performance.

    Input/weight shapes (high-level):
      - layers: list of layer sizes, e.g. [784, 1250, 10]
      - weights[i] shape: (layers[i+1], layers[i] + 1) -> +1 accounts for bias column
    """

    def __init__(self, layers):
        # layers: list of integers (sizes of each layer, input..output)
        self.layers = layers
        self.weights = []

        # Initialize weights randomly for each layer transition.
        # We add +1 to the input size to store the bias weight for that neuron.
        for i in range(len(layers) - 1):
            # rows = next layer size, cols = current layer size + 1 (bias)
            # Using Xavier/Glorot initialization for better performance with sigmoid
            limit = np.sqrt(6 / (layers[i] + layers[i + 1]))
            layers_weights = np.random.uniform(
                -limit, limit, (layers[i + 1], layers[i] + 1)
            )
            self.weights.append(layers_weights)

    def sigmoid(self, x):
        """
        Sigmoid activation.
        Note: inputs are scaled by 0.01 here which makes the function much more linear.
        """
        return 1 / (1 + np.exp(-0.01 * x))

    def predict(self, data):
        """Forward pass that returns the network's output vector for a single example.

        :param data: 1D numpy array with shape (input_dim,)
        :return: output activation (1D array) of the final layer.
        """
        x_s = [data]  # store activations per layer; x_s[0] is input (no bias yet)

        # Iterate through layers and compute activations
        for i in range(len(self.layers) - 1):
            # Add bias term to the current activation vector (append a constant 1).
            # This matches the +1 column in the weight matrices.
            x_s[-1] = np.concatenate((x_s[-1], [1]))

            # Compute z = W * x  (W shape: [next_size, cur_size+1])
            z = np.dot(self.weights[i], x_s[i])

            # Apply sigmoid and append to activations list.
            x_s.append(self.sigmoid(z))

        # x_s[-1] is the final layer activation (output probabilities/values)
        return x_s[-1]

    def train(self, data, y_true):
        """Single-step training using one example and its one-hot label vector.

        This performs a forward pass, computes output-layer deltas (psi),
        backpropagates those deltas through hidden layers, accumulates gradients,
        updates weights with a fixed learning rate (0.1), and returns the MSE.
        """
        x_s = [data]

        # Forward pass (same as predict) while keeping intermediate activations
        for i in range(len(self.layers) - 1):
            # add bias to activation before multiplying by weight matrix
            x_s[-1] = np.concatenate((x_s[-1], [1]))
            z = np.dot(self.weights[i], x_s[i])
            x_s.append(self.sigmoid(z))

        # Compute output-layer error signals (psi).
        # For each output neuron: psi = d(MSE)/d(output) * sigmoid'(z).
        # MSE derivative w.r.t output is -2*(y_true - output).
        psi = []
        for i in range(len(y_true)):
            output = x_s[-1][i]
            # output * (1 - output) is sigmoid'(z) assuming standard sigmoid;
            # here sigmoid was scaled but we still use this form for derivative.

            # BUG FIX
            # The derivative of sigmoid(k*x) is k * sigmoid(k*x) * (1 - sigmoid(k*x))
            # Here k=0.01
            k = 0.01
            derivative = k * output * (1 - output)
            psi.append(-2 * (y_true[i] - output) * derivative)
            # END FIX

        psi = np.array(psi)
        # reshape to column vector: (n_outputs, 1)
        psi = np.reshape(psi, (psi.shape[0], 1))

        # gradients will store gradients for each layer (in back-to-front order)
        gradients = []
        # Gradient for last weight matrix: outer product of psi (n_out x 1) and
        # previous activations x_s[-2] (1 x (prev_size+1)) -> shape (n_out, prev_size+1).
        gradients.append(psi * x_s[-2])

        # Backpropagate through hidden layers (excluding input layer).
        for i in range(len(self.layers) - 2, 0, -1):
            # w = weights for layer i (from layer i to i+1), drop its bias column
            # because bias weights don't connect to previous activations directly.
            w = self.weights[i][:, :-1]

            # x = activation of layer i (without bias term); used in derivative
            x = x_s[i][:-1]

            # BUG FIX
            # The derivative term here must also be scaled by k=0.01
            k = 0.01
            term = w * (k * x * (1 - x))

            term = np.transpose(term)

            # Propagate psi backward: new psi = term dot psi
            psi = np.dot(term, psi)
            psi = np.reshape(psi, (psi.shape[0], 1))

            # Gradient for this layer's weight matrix (psi * activations of prev layer)
            gradients.append(psi * x_s[i - 1])

        # Update weights: gradients list was built from output backwards, so we
        # reverse-index gradients when applying to weights[0..]. Learning rate=0.1
        for i in range(len(gradients)):
            self.weights[i] -= 0.1 * gradients[-(i + 1)]

        # Return mean squared error for this example (sum of squared differences).
        return sum((y_true - x_s[-1]) ** 2)
