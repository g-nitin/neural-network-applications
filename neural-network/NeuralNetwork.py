import numpy as np


class NeuralNetwork:
    """
    A flexible Neural Network class that can handle 1 or 2 layers.
    """

    def __init__(self, layer_dims, activations, initialization_method="random"):
        """
        Initializes the Neural Network.

        layer_dims (list): A list of integers representing the number of neurons in each layer.
            e.g., [input_dim, output_dim] for a 1-layer network.
            e.g., [input_dim, hidden_dim, output_dim] for a 2-layer network.
        activations (list): A list of strings for the activation function of each layer (except input).
            e.g., ['sigmoid'] or ['tanh', 'radialbasis'].
        initialization_method (str): Method for initializing weights ('random', 'xavier', 'he').
        """
        self.parameters = {}
        self.num_layers = len(layer_dims) - 1  # Number of weighted layers
        self.activations = activations
        self.beta = 1.0  # For RBF

        # Activation functions and their derivatives
        # Stored in a dictionary for easy lookup, using only the requested function names.
        # Note: Derivatives are often calculated using the activated output 'A' for stability,
        # except for ReLU and RBF where 'Z' is more direct.
        self._activation_functions = {
            "tanh": (self._tanh, self._tanh_derivative),
            "logsig": (self._logsig, self._logsig_derivative),
            "tansig": (self._tansig, self._tansig_derivative),
            "radialbasis": (self._radial_basis, self._radial_basis_derivative),
            "relu": (self._relu, self._relu_derivative),
            "linear": (
                self._linear,
                self._linear_derivative,
            ),  # Kept for Q2 output layer
        }

        # Initialize weights and biases for each layer
        for layer in range(1, self.num_layers + 1):
            n_in = layer_dims[layer - 1]
            n_out = layer_dims[layer]

            # Weight Initialization Logic
            if initialization_method == "xavier":
                # For tanh/sigmoid, symmetric around 0
                limit = np.sqrt(6 / (n_in + n_out))
                self.parameters[f"W{layer}"] = np.random.uniform(
                    -limit, limit, size=(n_out, n_in)
                )
            elif initialization_method == "he":
                # For ReLU, asymmetric
                std_dev = np.sqrt(2 / n_in)
                self.parameters[f"W{layer}"] = np.random.randn(n_out, n_in) * std_dev
            else:  # 'random' (default, small random values)
                self.parameters[f"W{layer}"] = np.random.randn(n_out, n_in) * 0.1

            # Biases initialized to zero
            self.parameters[f"b{layer}"] = np.zeros((n_out, 1))

    # Activation Functions
    def _tanh(self, Z):
        # Formula: (e^Z - e^-Z) / (e^Z + e^-Z)
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    def _tanh_derivative(self, A):
        return 1 - np.power(A, 2)

    def _logsig(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _logsig_derivative(self, A):
        return A * (1 - A)

    def _tansig(self, Z):
        # Formula: 2 / (1 + e^(-2*Z)) - 1
        return 2 / (1 + np.exp(-2 * Z)) - 1

    def _tansig_derivative(self, A):
        return 1 - np.power(A, 2)

    def _radial_basis(self, Z):
        return np.exp(-self.beta * np.power(Z, 2))

    def _radial_basis_derivative(self, Z):
        # Derivative of exp(-beta * Z^2) is -2 * beta * Z * exp(-beta * Z^2)
        return -2 * self.beta * Z * np.exp(-self.beta * np.power(Z, 2))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        dZ = np.ones_like(Z)
        dZ[Z <= 0] = 0
        return dZ

    def _linear(self, Z):
        return Z

    def _linear_derivative(self, Z):
        return np.ones_like(Z)

    def forward(self, X):
        """
        Performs the forward pass through the network.

        X (np.array): Input data of shape (input_dim, num_samples).

        Returns:
        - A_final (np.array): The final output of the network.
        - cache (dict): A dictionary storing intermediate values (Z, A, W) needed for backpropagation.
        """
        cache = {}
        cache["A0"], A = X, X

        for layer in range(1, self.num_layers + 1):
            A_prev = A  # Save previous layer
            W = self.parameters[f"W{layer}"]
            b = self.parameters[f"b{layer}"]
            activation_name = self.activations[layer - 1]
            activation_func, _ = self._activation_functions[activation_name]

            # Linear step: Z = W * A_prev + b
            Z = np.dot(W, A_prev) + b
            # Activation step: A = g(Z)
            A = activation_func(Z)

            # Store values in cache for backpropagation
            cache[f"Z{layer}"] = Z
            cache[f"A{layer}"] = A
            cache[f"W{layer}"] = W

        return A, cache

    def compute_loss(self, Y_pred, Y_true):
        """
        Computes the Mean Squared Error (MSE) loss.

        Y_pred (np.array): The predicted output from the network.
        Y_true (np.array): The ground truth labels.

        Return: (float) The calculated MSE loss.
        """
        m = Y_true.shape[1]  # Number of samples in the current batch
        loss = (1 / m) * np.sum((Y_pred - Y_true) ** 2)
        return loss

    def backward(self, Y_pred, Y_true, cache):
        """
        Performs the backward pass (backpropagation) to compute gradients.

        Y_pred (np.array): The predicted output.
        Y_true (np.array): The ground truth labels.
        cache (dict): The cache from the forward pass.

        Return: grads (dict): A dictionary containing the gradients (dW, db) for each layer.
        """
        grads = {}
        m = Y_true.shape[1]  # Number of samples in the current batch
        L = self.num_layers

        # Start with the last layer (L)
        # Derivative of the MSE loss with respect to the final output A
        dAL = 2 * (Y_pred - Y_true) / m

        # Loop backwards from the last layer to the first
        for layer in reversed(range(1, L + 1)):
            Zl = cache[f"Z{layer}"]  # Pre-activation
            Al = cache[f"A{layer}"]  # Post-activation
            A_prev = cache[f"A{layer - 1}"]  # Activation from previous layer
            Wl = cache[f"W{layer}"]  # Weights of current layer
            activation_name = self.activations[layer - 1]  # Activation function name
            _, activation_deriv = self._activation_functions[activation_name]

            # Calculate dZ for the current layer
            # Note on derivative calculation:
            # For logsig/tanh/tansig, it's easier to use A. For relu/rbf, we use Z.
            # This is because the first functions' derivatives can be expressed in terms of A,
            # since they are smooth and bounded functions:
            # - $g'(Z) = g(Z)(1 - g(Z))$ for logsig,
            # - $g'(Z) = 1 - g(Z)^2$ for tanh,
            # - $g'(Z) = 1 - g(Z)^2$ for tansig.
            # For ReLU, the derivative is piecewise and depends directly on Z (0 for Z <= 0, 1 for Z > 0): $g'(Z) = 1_{Z > 0}$.
            if activation_name in ["logsig", "tanh", "tansig", "linear"]:
                dZl = dAL * activation_deriv(Al)
            else:  # relu, radialbasis
                dZl = dAL * activation_deriv(Zl)

            # Calculate gradients for W and b
            # $dW = dZ * A_prev^T$; shape (neurons_current, neurons_prev)
            grads[f"dW{layer}"] = np.dot(dZl, A_prev.T)
            # $db = sum(dZ)$; shape (neurons_current, 1); axis=1 is sum over samples; keepdims to maintain 2D shape
            grads[f"db{layer}"] = np.sum(dZl, axis=1, keepdims=True)

            # Calculate the gradient for the previous layer's activation (dAL_prev)
            # This is the error to be propagated back.
            if layer > 1:
                dAL = np.dot(Wl.T, dZl)

        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Updates the network's weights and biases using gradient descent.

        grads (dict): The gradients calculated during backpropagation.
        learning_rate (float): The step size for the update.
        """
        for layer in range(1, self.num_layers + 1):
            self.parameters[f"W{layer}"] -= learning_rate * grads[f"dW{layer}"]
            self.parameters[f"b{layer}"] -= learning_rate * grads[f"db{layer}"]

    def train(
        self,
        X,
        Y,
        epochs,
        learning_rate,
        batch_size=None,  # For mini-batch
        activation=None,  # For plotting labels
        plot_epochs=None,
        plot_callback=None,
        plot_filename_prefix="plot",
        verbose=True,
        # Parameters for plotting callback to handle scaling
        X_original_unscaled=None,
        Y_original_unscaled=None,
        x_scaler=None,
        y_scaler=None,
    ):
        """
        The main training loop (with mini-batch gradient descent).
        """
        loss_history = []
        m_total = X.shape[1]  # Total number of samples

        # If batch_size is None or larger than total samples, use full-batch GD
        if batch_size is None or batch_size >= m_total:
            batch_size = m_total
        num_batches = int(np.ceil(m_total / batch_size))

        for i in range(epochs):
            # Shuffle data at the beginning of each epoch
            permutation = np.random.permutation(m_total)
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation]

            epoch_loss_sum = 0  # To calculate average loss for the epoch

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, m_total)
                X_batch = shuffled_X[:, start_idx:end_idx]
                Y_batch = shuffled_Y[:, start_idx:end_idx]

                # Forward pass on batch
                Y_pred_batch, cache = self.forward(X_batch)

                # Compute loss for the batch
                batch_loss = self.compute_loss(Y_pred_batch, Y_batch)
                epoch_loss_sum += (
                    batch_loss * X_batch.shape[1]
                )  # Sum of squared errors for the batch

                # Backward pass on batch
                grads = self.backward(Y_pred_batch, Y_batch, cache)

                # Update parameters using batch gradients
                self.update_parameters(grads, learning_rate)

            # Calculate average epoch loss
            epoch_loss = epoch_loss_sum / m_total
            loss_history.append(epoch_loss)

            if verbose and (i + 1) % 100 == 0:
                print(f"Epoch {i + 1}/{epochs} - Loss: {epoch_loss:.6f}")

            if plot_epochs and (i + 1) in plot_epochs:
                if plot_callback:
                    # Create a unique filename for each plot
                    filename = f"{plot_filename_prefix}_epoch_{i + 1}.png"
                    # Pass all necessary data for plotting, including scalers
                    plot_callback(
                        self,
                        X,  # Scaled X for model.forward
                        Y,  # Scaled Y for model.forward
                        i + 1,
                        activation,
                        filename,
                        X_original_unscaled,
                        Y_original_unscaled,
                        x_scaler,
                        y_scaler,
                    )

        return loss_history


# Helper class for data standardization
class Scaler:
    """
    A simple scaler for standardizing data (zero mean, unit variance).
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """Calculates mean and standard deviation from data."""
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.std = np.std(data, axis=1, keepdims=True)
        # Handle cases where std is zero (constant data) to avoid division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """Applies standardization using fitted mean and std."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / self.std

    def inverse_transform(self, scaled_data):
        """Inverse transforms scaled data back to original scale."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return scaled_data * self.std + self.mean
