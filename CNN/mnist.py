import os

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

DEBUG = False
SEED = 13
np.random.seed(SEED)


def load_mnist_from_csv(train_path, test_path):
    """
    Loads the MNIST dataset from CSV files.
    Assumes the first column is the label and the rest are 784 pixel values.
    The CSV files are expected to have a header row, which is skipped.
    """
    # Load training data
    print(f"Loading training data from {train_path}...")
    train_data = np.loadtxt(train_path, delimiter=",", skiprows=1)
    train_labels = train_data[:, 0].astype(np.uint8)
    train_images = train_data[:, 1:].reshape(-1, 28, 28)

    # Load test data
    print(f"Loading test data from {test_path}...")
    test_data = np.loadtxt(test_path, delimiter=",", skiprows=1)
    test_labels = test_data[:, 0].astype(np.uint8)
    test_images = test_data[:, 1:].reshape(-1, 28, 28)

    return (train_images, train_labels), (test_images, test_labels)


def preprocess_data(images, labels):
    """
    Normalizes images and one-hot encodes labels.
    """
    # Normalize images to be between -0.5 and 0.5
    images = (images.astype("float32") / 255.0) - 0.5
    # Add a channel dimension
    images = np.expand_dims(images, axis=1)
    # One-hot encode labels
    num_classes = 10
    one_hot_labels = np.eye(num_classes)[labels]
    return images, one_hot_labels


# Visualization Utilities
def plot_training_metrics(
    epoch_losses, epoch_accuracies, save_path="figures_mnist/training_metrics.png"
):
    """
    Plots the training loss and accuracy per epoch and saves to file.
    """
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training Metrics", fontsize=16, fontweight="bold")

    # Create epoch numbers (1-based)
    epochs = range(1, len(epoch_losses) + 1)

    # Plotting Loss
    ax1.plot(
        epochs,
        epoch_losses,
        label="Training Loss",
        color="#E74C3C",
        linewidth=2.5,
        marker="o",
    )
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Cross-Entropy Loss per Epoch", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#f8f9fa")

    # Plotting Accuracy
    ax2.plot(
        epochs,
        epoch_accuracies,
        label="Training Accuracy",
        color="#3498DB",
        linewidth=2.5,
        marker="s",
    )
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title("Classification Accuracy per Epoch", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim([0, 1])
    ax2.set_facecolor("#f8f9fa")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training metrics saved to {save_path}")


def plot_confusion_matrix(
    true_labels, predicted_labels, save_path="figures_mnist/confusion_matrix.png"
):
    """
    Creates and saves a confusion matrix visualization.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create confusion matrix
    num_classes = 10
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        conf_matrix[true, pred] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(conf_matrix, cmap="Blues", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Predictions", fontsize=12, fontweight="bold")

    # Set ticks and labels
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(range(num_classes), fontsize=11)  # type: ignore
    ax.set_yticklabels(range(num_classes), fontsize=11)  # type: ignore

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title(
        "Confusion Matrix - MNIST CNN Classification",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text_color = (
                "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            )
            ax.text(
                j,
                i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

    return conf_matrix


def plot_per_class_accuracy(
    true_labels, predicted_labels, save_path="figures_mnist/per_class_accuracy.png"
):
    """
    Creates a bar chart showing accuracy for each digit class.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_classes = 10
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for true, pred in zip(true_labels, predicted_labels):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1

    accuracies = correct_per_class / total_per_class

    # Create bar plot
    _, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, num_classes))  # type: ignore
    bars = ax.bar(
        range(num_classes), accuracies, color=colors, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for _, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Digit Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title(
        "Per-Class Classification Accuracy", fontsize=15, fontweight="bold", pad=15
    )
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(range(num_classes), fontsize=11)  # type: ignore
    ax.set_ylim([0, 1.05])  # type: ignore
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy saved to {save_path}")


def visualize_feature_maps(
    model, sample_image, save_path="figures_mnist/feature_maps.png"
):
    """
    Visualizes the feature maps (activations) from convolutional layers for a sample image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Forward pass through layers to get activations
    conv1_out = model.conv1.forward(sample_image)
    conv1_relu = relu(conv1_out)
    conv2_out = model.conv2.forward(conv1_relu)
    conv2_relu = relu(conv2_out)
    conv3_out = model.conv3.forward(conv2_relu)
    conv3_relu = relu(conv3_out)

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # Original image
    ax = plt.subplot(4, 8, 1)
    ax.imshow(sample_image[0], cmap="gray")
    ax.set_title("Input Image", fontsize=10, fontweight="bold")
    ax.axis("off")

    # Conv1 feature maps
    for i in range(8):
        ax = plt.subplot(4, 8, i + 1)
        if i == 0:
            ax.imshow(sample_image[0], cmap="gray")
            ax.set_title("Input", fontsize=9, fontweight="bold")
        else:
            ax.imshow(conv1_relu[i - 1], cmap="viridis")
            ax.set_title(f"Conv1-{i}", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Conv2 feature maps
    for i in range(8):
        ax = plt.subplot(4, 8, i + 9)
        ax.imshow(conv2_relu[i], cmap="viridis")
        ax.set_title(f"Conv2-{i + 1}", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Conv3 feature maps
    for i in range(8):
        ax = plt.subplot(4, 8, i + 17)
        ax.imshow(conv3_relu[i], cmap="viridis")
        ax.set_title(f"Conv3-{i + 1}", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Pooled output
    pool_out = model.pool.forward(conv3_relu)
    for i in range(8):
        ax = plt.subplot(4, 8, i + 25)
        ax.imshow(pool_out[i], cmap="viridis")
        ax.set_title(f"Pool-{i + 1}", fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Feature Maps Through CNN Layers", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature maps visualization saved to {save_path}")


def visualize_predictions(
    images,
    true_labels,
    model,
    num_to_show=10,
    save_path="figures_mnist/predictions.png",
):
    """
    Selects random images from the dataset, makes predictions, and displays them in a single figure.
    """
    print(f"\nGenerating visualization with {num_to_show} random test predictions...")
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Preprocess images for the model
    processed_images, _ = preprocess_data(images, true_labels)

    # Get random indices
    indices = np.random.choice(len(images), num_to_show, replace=False)

    # Calculate grid dimensions (2 rows x 5 columns for 10 images)
    rows = 2
    cols = 5

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle(
        "Sample Predictions from Test Set", fontsize=16, fontweight="bold", y=0.98
    )

    axes = axes.flatten()

    for idx, i in enumerate(indices):
        img = images[i]
        processed_img = processed_images[i]
        true_label = true_labels[i]

        # Get model prediction
        probs = model.forward(processed_img).flatten()
        predicted_label = np.argmax(probs)
        confidence = probs[predicted_label]

        # Determine if prediction is correct
        is_correct = predicted_label == true_label
        color = "green" if is_correct else "red"

        # Display image
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(
            f"True: {true_label} | Pred: {predicted_label}\n(Conf: {confidence:.2f})",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        axes[idx].axis("off")

    plt.tight_layout()
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

        # Filters are now 4D: (num_filters, input_channels, height, width)
        self.filters = np.random.randn(
            num_filters, input_channels, filter_size, filter_size
        ) / (filter_size * filter_size)
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
        num_channels, height, width = input_data.shape

        output_height = height - filter_size + 1
        output_width = width - filter_size + 1
        output = np.zeros((num_filters, output_height, output_width))

        for f in range(num_filters):
            for y in range(output_height):
                for x in range(output_width):
                    image_patch = input_data[
                        :, y : (y + filter_size), x : (x + filter_size)
                    ]
                    output[f, y, x] = np.sum(image_patch * filters[f]) + biases[f]
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
                    image_patch = last_input[
                        :, y : (y + filter_size), x : (x + filter_size)
                    ]
                    # Gradient for filters
                    d_L_d_filters[f] += d_L_d_out[f, y, x] * image_patch
                    # Gradient for input
                    d_L_d_input[:, y : (y + filter_size), x : (x + filter_size)] += (
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
                        (y * pool_size) : (y * pool_size + pool_size),
                        (x * pool_size) : (x * pool_size + pool_size),
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
                        (y * pool_size) : (y * pool_size + pool_size),
                        (x * pool_size) : (x * pool_size + pool_size),
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
        self.weights = np.random.randn(input_size, output_size) * 0.1
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


# CNN Model
class CNN:
    """A simple CNN model for MNIST classification."""

    def __init__(self):
        # ConvLayer(input_channels, num_filters, filter_size)
        self.conv1 = ConvLayer(input_channels=1, num_filters=8, filter_size=3)
        self.conv2 = ConvLayer(input_channels=8, num_filters=8, filter_size=3)
        self.conv3 = ConvLayer(input_channels=8, num_filters=8, filter_size=3)
        self.pool = MaxPoolingLayer(pool_size=2)
        # The input size for the dense layer depends on the output of the conv/pool layers
        # Input: 1x28x28 -> Conv1(8 filters, 3x3): 8x26x26 -> Conv2(8 filters, 3x3): 8x24x24 -> Conv3(8 filters, 3x3): 8x22x22 -> Pool(2x2): 8x11x11
        # So, flattened size is 8 * 11 * 11 = 968
        self.dense = DenseLayer(input_size=8 * 11 * 11, output_size=10)

    def forward(self, image):
        """Complete forward pass through the network."""
        out = self.conv1.forward(image)
        out = relu(out)
        out = self.conv2.forward(out)
        out = relu(out)
        out = self.conv3.forward(out)
        out = relu(out)
        out = self.pool.forward(out)
        out = self.dense.forward(out)
        return softmax(out.reshape(1, -1))

    def train(self, image, label, learning_rate=0.005):
        """
        Performs a full training step (forward and backward pass).
        """
        #  Forward Pass
        # We need to store intermediate outputs for backpropagation
        conv1_out = self.conv1.forward(image)
        conv1_out_relu = relu(conv1_out)
        conv2_out = self.conv2.forward(conv1_out_relu)
        conv2_out_relu = relu(conv2_out)
        conv3_out = self.conv3.forward(conv2_out_relu)
        conv3_out_relu = relu(conv3_out)
        pool_out = self.pool.forward(conv3_out_relu)
        dense_out = self.dense.forward(pool_out)
        final_probs = softmax(dense_out.reshape(1, -1)).flatten()

        #  Calculate Loss and Initial Gradient
        # Cross-entropy loss derivative
        gradient = final_probs - label

        #  Backward Pass
        gradient = self.dense.backward(gradient, learning_rate)
        gradient = self.pool.backward(gradient)
        gradient = gradient * relu_derivative(conv3_out)
        gradient = self.conv3.backward(gradient, learning_rate)
        gradient = gradient * relu_derivative(conv2_out)
        gradient = self.conv2.backward(gradient, learning_rate)
        gradient = gradient * relu_derivative(conv1_out)
        self.conv1.backward(gradient, learning_rate)

        # Calculate loss for reporting
        loss = -np.sum(label * np.log(final_probs + 1e-9))  # Add epsilon for stability
        return loss, np.argmax(final_probs) == np.argmax(label)


# Main Training Loop
def main():
    # Define paths to the CSV files within the 'data' subdirectory
    train_csv_path = os.path.join("data", "mnist_train.csv")
    test_csv_path = os.path.join("data", "mnist_test.csv")

    print("Loading MNIST dataset from CSV files...")
    (
        (raw_train_images, raw_train_labels),
        (
            raw_test_images,
            raw_test_labels,
        ),
    ) = load_mnist_from_csv(train_csv_path, test_csv_path)

    # Preprocess data for the network
    train_images, train_labels = preprocess_data(raw_train_images, raw_train_labels)
    test_images, test_labels = preprocess_data(raw_test_images, raw_test_labels)

    if DEBUG:
        train_images = train_images[:100]
        train_labels = train_labels[:100]
        print(f"\nDEBUG: Using subset {len(train_images)} samples for training.")

    model = CNN()
    learning_rate = 0.005
    epochs = 1 if DEBUG else 3
    num_samples = 100 if DEBUG else len(train_images)

    print(f"\nStarting training for {epochs} epochs...")

    # For plotting
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(epochs):
        print(f" Epoch {epoch + 1}/{epochs} ")
        total_loss = 0
        num_correct = 0

        # Shuffle training data
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        for i in range(num_samples):
            image = train_images[i]
            label = train_labels[i]

            loss, correct = model.train(image, label, learning_rate)
            total_loss += loss
            if correct:
                num_correct += 1

            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                accuracy = num_correct / (i + 1)
                print(
                    f"Epoch {epoch + 1}, Step {i + 1}/{num_samples}, Avg Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}"
                )

        # Record metrics for the epoch
        epoch_losses.append(total_loss / num_samples)
        epoch_accuracies.append(num_correct / num_samples)

    print("\nTraining Complete")

    # Plot training metrics
    plot_training_metrics(epoch_losses, epoch_accuracies)

    if DEBUG:
        test_images = test_images[:100]
        test_labels = test_labels[:100]
        print(f"\nDEBUG: Using subset {len(test_images)} samples for testing.")

    print("\nTesting the model")
    test_correct = 0
    test_loss = 0
    test_predictions = []
    test_true_labels = []

    for i in range(len(test_images)):
        print(f"Testing sample {i + 1}/{len(test_images)}", end="\r")
        image = test_images[i]
        label = test_labels[i]
        probs = model.forward(image).flatten()
        test_loss -= np.sum(label * np.log(probs + 1e-9))

        pred_label = np.argmax(probs)
        true_label = np.argmax(label)
        test_predictions.append(pred_label)
        test_true_labels.append(true_label)

        if pred_label == true_label:
            test_correct += 1

    print(f"\nTest Loss: {test_loss / len(test_images):.3f}")
    print(f"Test Accuracy: {test_correct / len(test_images):.3f}")

    # Generate comprehensive visualizations for the report
    print("\nGenerating visualizations for report...")

    # 1. Training metrics (already done)

    # 2. Confusion matrix
    plot_confusion_matrix(test_true_labels, test_predictions)

    # 3. Per-class accuracy
    plot_per_class_accuracy(test_true_labels, test_predictions)

    # 4. Feature maps for a sample image
    sample_idx = np.random.randint(0, len(test_images))
    visualize_feature_maps(
        model,
        test_images[sample_idx],
        save_path=f"figures_mnist/feature_maps_digit_{test_true_labels[sample_idx]}.png",
    )

    # 5. Sample predictions
    visualize_predictions(raw_test_images, raw_test_labels, model, num_to_show=10)

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
