import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

DEBUG = False
SEED = 13
np.random.seed(SEED)

# Define CIFAR-10 class names for visualization
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cifar10(data_dir):
    """
    Loads the CIFAR-10 dataset from the pickled batch files.
    """
    print(f"Loading CIFAR-10 data from {data_dir}...")

    def unpickle(file):
        with open(os.path.join(data_dir, file), "rb") as fo:
            data_dict = pickle.load(fo, encoding="latin1")
        return data_dict

    # Load training data from all 5 batches
    train_images = []
    train_labels = []
    for i in range(1, 6):
        batch_dict = unpickle(f"data_batch_{i}")
        train_images.append(batch_dict["data"])
        train_labels.extend(batch_dict["labels"])
    train_images = np.concatenate(train_images)
    train_labels = np.array(train_labels)

    # Load test data
    test_dict = unpickle("test_batch")
    test_images = test_dict["data"]
    test_labels = np.array(test_dict["labels"])

    # Reshape data into 3x32x32 images
    train_images = train_images.reshape(-1, 3, 32, 32)
    test_images = test_images.reshape(-1, 3, 32, 32)

    return (train_images, train_labels), (test_images, test_labels)


def preprocess_data(images, labels):
    """
    Normalizes images and one-hot encodes labels.
    """
    # Normalize images to be between -0.5 and 0.5
    images = (images.astype("float32") / 255.0) - 0.5
    # One-hot encode labels
    num_classes = 10
    one_hot_labels = np.eye(num_classes)[labels]
    return images, one_hot_labels


# Visualization Utilities
def plot_training_metrics(
    epoch_losses, epoch_accuracies, save_path="figures_cifar/training_metrics.png"
):
    """
    Plots the training loss and accuracy per epoch and saves to file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(epoch_losses) + 1)
    ax1.plot(epochs, epoch_losses, label="Training Loss", color="#E74C3C", marker="o")
    ax1.set_title("Cross-Entropy Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax2.plot(
        epochs, epoch_accuracies, label="Training Accuracy", color="#3498DB", marker="s"
    )
    ax2.set_title("Classification Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training metrics saved to {save_path}")


def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    class_names,
    save_path="figures_cifar/confusion_matrix.png",
):
    """
    Creates and saves a confusion matrix visualization.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_classes = len(class_names)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        conf_matrix[true, pred] += 1

    _, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(conf_matrix, cmap="Blues")
    _ = plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - CIFAR-10 CNN Classification")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def visualize_predictions(
    images,
    true_labels,
    model,
    class_names,
    num_to_show=10,
    save_path="figures_cifar/predictions.png",
):
    """
    Selects random images, makes predictions, and displays them.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    processed_images, _ = preprocess_data(images, true_labels)

    indices = np.random.choice(len(images), num_to_show, replace=False)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    fig.suptitle("Sample Predictions from Test Set", fontsize=16, y=0.98)
    axes = axes.flatten()

    for idx, i in enumerate(indices):
        img = images[i]
        processed_img = processed_images[i]  # Shape: (3, 32, 32)
        true_label_idx = true_labels[i]
        probs = model.forward(processed_img).flatten()
        predicted_label_idx = np.argmax(probs)
        is_correct = predicted_label_idx == true_label_idx
        color = "green" if is_correct else "red"

        # Un-normalize and transpose for display: (3, 32, 32) -> (32, 32, 3)
        display_img = np.transpose(img, (1, 2, 0))

        axes[idx].imshow(display_img)
        axes[idx].set_title(
            f"True: {class_names[true_label_idx]}\nPred: {class_names[predicted_label_idx]}",
            fontsize=9,
            color=color,
        )
        axes[idx].axis("off")

    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Predictions visualization saved to {save_path}")


# Activation Functions
@jit(nopython=True)
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


@jit(nopython=True)
def relu_derivative(x):
    """Derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)


def softmax(x):
    """Softmax activation function."""
    # Subtract max for numerical stability
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# Layer Implementations
class ConvLayer:
    """A 2D Convolutional Layer that handles multiple input channels."""

    def __init__(self, input_channels, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels

        # He initialization
        fan_in = input_channels * filter_size * filter_size
        std_dev = np.sqrt(2.0 / fan_in)
        self.filters = (
            np.random.randn(num_filters, input_channels, filter_size, filter_size)
            * std_dev
        )

        self.biases = np.zeros(num_filters)
        self.last_input = None

    def forward(self, input_data):
        """
        Performs the forward pass of the convolution.
        input_data is a 3D numpy array (channels, height, width).
        """
        self.last_input = input_data
        return self._numba_forward(
            input_data, self.filters, self.biases, self.filter_size, self.num_filters
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_forward(input_data, filters, biases, filter_size, num_filters):
        _, height, width = input_data.shape

        output_height = height - filter_size + 1
        output_width = width - filter_size + 1
        output = np.zeros((num_filters, output_height, output_width))

        for f in range(num_filters):
            for y in range(output_height):
                for x in range(output_width):
                    patch = input_data[:, y : y + filter_size, x : x + filter_size]
                    output[f, y, x] = np.sum(patch * filters[f]) + biases[f]
        return output

    def backward(self, d_L_d_out, learning_rate):
        """
        Performs the backward pass (backpropagation).
        """
        d_L_d_filters, d_L_d_input = self._numba_backward(
            d_L_d_out, self.filters, self.last_input, self.filter_size, self.num_filters
        )

        # Update filters and biases (biases gradient is simpler)
        d_L_d_biases = np.sum(d_L_d_out, axis=(1, 2))
        self.filters -= learning_rate * d_L_d_filters
        self.biases -= learning_rate * d_L_d_biases

        return d_L_d_input

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_backward(d_L_d_out, filters, last_input, filter_size, num_filters):
        d_L_d_filters = np.zeros(filters.shape)
        d_L_d_input = np.zeros(last_input.shape)
        _, output_height, output_width = d_L_d_out.shape

        for f in range(num_filters):
            for y in range(output_height):
                for x in range(output_width):
                    patch = last_input[:, y : y + filter_size, x : x + filter_size]
                    d_L_d_filters[f] += d_L_d_out[f, y, x] * patch
                    d_L_d_input[:, y : y + filter_size, x : x + filter_size] += (
                        d_L_d_out[f, y, x] * filters[f]
                    )
        return d_L_d_filters, d_L_d_input


class MaxPoolingLayer:
    """A Max Pooling Layer."""

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.last_input = None

    def forward(self, input_data):
        """Performs the forward pass of max pooling."""
        self.last_input = input_data
        return self._numba_forward(input_data, self.pool_size)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_forward(input_data, pool_size):
        num_channels, height, width = input_data.shape
        output_height = height // pool_size
        output_width = width // pool_size
        output = np.zeros((num_channels, output_height, output_width))

        for c in range(num_channels):
            for y in range(output_height):
                for x in range(output_width):
                    patch = input_data[
                        c,
                        y * pool_size : y * pool_size + pool_size,
                        x * pool_size : x * pool_size + pool_size,
                    ]
                    output[c, y, x] = np.max(patch)
        return output

    def backward(self, d_L_d_out, learning_rate=None):
        """Performs the backward pass for max pooling."""
        return self._numba_backward(d_L_d_out, self.last_input, self.pool_size)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_backward(d_L_d_out, last_input, pool_size):
        d_L_d_input = np.zeros(last_input.shape)
        num_channels, output_height, output_width = d_L_d_out.shape

        for c in range(num_channels):
            for y in range(output_height):
                for x in range(output_width):
                    patch = last_input[
                        c,
                        y * pool_size : y * pool_size + pool_size,
                        x * pool_size : x * pool_size + pool_size,
                    ]
                    max_val = np.max(patch)
                    # Find the index of the max value
                    for i in range(patch.shape[0]):
                        for j in range(patch.shape[1]):
                            if patch[i, j] == max_val:
                                d_L_d_input[c, y * pool_size + i, x * pool_size + j] = (
                                    d_L_d_out[c, y, x]
                                )
                                break  # Break inner loops once max is found
                        else:
                            continue
                        break
        return d_L_d_input


class DenseLayer:
    """A Fully Connected (Dense) Layer."""

    def __init__(self, input_size, output_size):
        # He initialization
        std_dev = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * std_dev

        self.biases = np.zeros(output_size)
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None

    def forward(self, input_data):
        """Performs the forward pass."""
        self.last_input_shape = input_data.shape
        # Flatten the input
        self.last_input = input_data.flatten()
        self.last_output = np.dot(self.last_input, self.weights) + self.biases
        return self.last_output

    def backward(self, d_L_d_out, learning_rate):
        """Performs the backward pass."""
        # Gradient of loss w.r.t. weights, biases, and input
        d_L_d_weights = np.dot(self.last_input.reshape(-1, 1), d_L_d_out.reshape(1, -1))  # type: ignore
        d_L_d_biases = d_L_d_out
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_biases

        # Reshape gradient to match original input shape
        return d_L_d_input.reshape(self.last_input_shape)


class DropoutLayer:
    """A Dropout Layer for regularization."""

    def __init__(self, rate=0.5):
        self.rate = rate  # The probability of *keeping* a neuron
        self.mask = None

    def forward(self, input_data, training=True):
        """
        Applies dropout during training.
        """
        if training:
            # Inverted dropout: scale during training, not testing
            self.mask = (np.random.rand(*input_data.shape) < self.rate) / self.rate
            return input_data * self.mask
        else:
            # During testing, do nothing
            return input_data

    def backward(self, d_L_d_out, learning_rate=None):
        """
        Backpropagates the gradient through the dropout mask.
        """
        return d_L_d_out * self.mask


# CNN Model for CIFAR-10
class CNN:
    """A CNN model adapted for CIFAR-10 classification."""

    def __init__(self):
        # ConvLayer(input_channels, num_filters, filter_size)
        self.conv1 = ConvLayer(input_channels=3, num_filters=16, filter_size=3)
        self.conv2 = ConvLayer(input_channels=16, num_filters=32, filter_size=3)
        self.pool1 = MaxPoolingLayer(pool_size=2)
        self.conv3 = ConvLayer(input_channels=32, num_filters=32, filter_size=3)
        self.pool2 = MaxPoolingLayer(pool_size=2)
        self.dropout = DropoutLayer(rate=0.5)

        # Calculate dense layer input size:
        # Input: 3x32x32
        # Conv1 (16f, 3x3): 16x30x30
        # Conv2 (32f, 3x3): 32x28x28
        # Pool1 (2x2): 32x14x14
        # Conv3 (32f, 3x3): 32x12x12
        # Pool2 (2x2): 32x6x6
        # Flattened size = 32 * 6 * 6 = 1152
        self.dense = DenseLayer(input_size=1152, output_size=10)

    def forward(self, image):
        """Complete forward pass through the network."""
        out = relu(self.conv1.forward(image))
        out = relu(self.conv2.forward(out))
        out = self.pool1.forward(out)
        out = relu(self.conv3.forward(out))
        out = self.pool2.forward(out)

        # Dropout with training=False for inference/testing
        out = self.dropout.forward(out, training=False)

        out = self.dense.forward(out)
        return softmax(out.reshape(1, -1))

    def train(self, image, label, learning_rate=0.001):
        """Performs a full training step (forward and backward pass)."""
        # Forward Pass (storing intermediate results)
        conv1_out = self.conv1.forward(image)
        conv1_relu = relu(conv1_out)
        conv2_out = self.conv2.forward(conv1_relu)
        conv2_relu = relu(conv2_out)
        pool1_out = self.pool1.forward(conv2_relu)
        conv3_out = self.conv3.forward(pool1_out)
        conv3_relu = relu(conv3_out)
        pool2_out = self.pool2.forward(conv3_relu)

        dropout_out = self.dropout.forward(pool2_out, training=True)

        dense_out = self.dense.forward(dropout_out)
        final_probs = softmax(dense_out.reshape(1, -1)).flatten()

        # Calculate Loss and Initial Gradient
        gradient = final_probs - label

        # Backward Pass
        gradient = self.dense.backward(gradient, learning_rate)

        # Backpropagate through dropout
        gradient = self.dropout.backward(gradient)

        gradient = self.pool2.backward(gradient)
        gradient *= relu_derivative(conv3_out)
        gradient = self.conv3.backward(gradient, learning_rate)
        gradient = self.pool1.backward(gradient)
        gradient *= relu_derivative(conv2_out)
        gradient = self.conv2.backward(gradient, learning_rate)
        gradient *= relu_derivative(conv1_out)
        self.conv1.backward(gradient, learning_rate)

        # Calculate loss for reporting
        loss = -np.sum(label * np.log(final_probs + 1e-9))  # Add epsilon for stability
        return loss, np.argmax(final_probs) == np.argmax(label)


# Main Training Loop
def main():

    cifar_data_dir = os.path.join("data", "cifar-10-batches-py")
    (
        (raw_train_images, raw_train_labels),
        (raw_test_images, raw_test_labels),
    ) = load_cifar10(cifar_data_dir)

    train_images, train_labels = preprocess_data(raw_train_images, raw_train_labels)
    test_images, test_labels = preprocess_data(raw_test_images, raw_test_labels)

    if DEBUG:
        train_images, train_labels = train_images[:200], train_labels[:200]
        print(f"\nDEBUG: Using subset of {len(train_images)} samples for training.")

    model = CNN()

    # Learning Rate Decay Parameters
    initial_lr = 0.002  # Start with a slightly higher rate
    learning_rate = initial_lr
    lr_decay_factor = 0.5  # Factor to reduce LR by
    lr_decay_epochs = 2  # Decay LR every 2 epochs

    epochs = 1 if DEBUG else 6
    num_samples = len(train_images)

    print(f"\nStarting training for {epochs} epochs...")
    print(
        f"Initial LR: {initial_lr}, Decay Factor: {lr_decay_factor}, Decay every {lr_decay_epochs} epochs."
    )
    epoch_losses, epoch_accuracies = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # LR Decay Logic
        if (epoch > 0) and (epoch % lr_decay_epochs == 0):
            learning_rate *= lr_decay_factor
            print(f"  Learning rate decayed to: {learning_rate:.6f}")

        total_loss, num_correct = 0, 0
        permutation = np.random.permutation(num_samples)
        train_images, train_labels = (
            train_images[permutation],
            train_labels[permutation],
        )

        for i in range(num_samples):
            # Pass the current (potentially decayed) learning rate to the train method
            loss, correct = model.train(train_images[i], train_labels[i], learning_rate)
            total_loss += loss
            if correct:
                num_correct += 1
            if (i + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch + 1}, Step {i + 1}/{num_samples}, Avg Loss: {total_loss / (i + 1):.3f}, Accuracy: {num_correct / (i + 1):.3f}"
                )
        epoch_losses.append(total_loss / num_samples)
        epoch_accuracies.append(num_correct / num_samples)

    print("\nTraining Complete")
    plot_training_metrics(epoch_losses, epoch_accuracies)

    if DEBUG:
        test_images = test_images[:100]
        test_labels = test_labels[:100]
        print(f"\nDEBUG: Using subset {len(test_images)} samples for testing.")

    print("\nTesting the model...")
    test_correct, test_predictions, test_true_labels = 0, [], []
    for i in range(len(test_images)):
        print(f"Testing sample {i + 1}/{len(test_images)}", end="\r")
        probs = model.forward(test_images[i]).flatten()
        pred_label = np.argmax(probs)
        true_label = np.argmax(test_labels[i])
        test_predictions.append(pred_label)
        test_true_labels.append(true_label)
        if pred_label == true_label:
            test_correct += 1
    print(f"Test Accuracy: {test_correct / len(test_images):.3f}")

    print("\nGenerating visualizations...")
    plot_confusion_matrix(test_true_labels, test_predictions, CIFAR10_CLASSES)
    visualize_predictions(
        raw_test_images, raw_test_labels, model, CIFAR10_CLASSES, num_to_show=10
    )
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
