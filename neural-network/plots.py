import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Plotting Utility Functions
def plot_error_vs_epoch(loss_history, filename):
    """Saves a plot of training error vs. epoch."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title("Training Error vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory


def plot_combined_error_vs_epoch(loss_histories_dict, title, filename):
    """
    Plots multiple training error curves on a single graph for comparison.

    loss_histories_dict (dict): A dictionary where keys are activation function names (str) and values are their respective loss history lists (list of floats).
    title (str): The title for the plot.
    """
    plt.figure(figsize=(10, 6))
    for activation_name, history in loss_histories_dict.items():
        plt.plot(history, label=activation_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory


def plot_decision_boundary(model, X, Y, epoch, activation, filename, *args):
    """
    Saves the decision boundary plot for the classification problem.
    Plots lines where the output neurons cross the 0.5 threshold.
    """
    plt.figure(figsize=(8, 6))

    # Define the grid for plotting the decision boundary
    # Add some padding to the min/max values for better visualization
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.05  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T

    # Get the network's raw outputs (before rounding) for the grid points
    # Z here refers to the output of the activation function of the last layer,
    # which is Y_pred in the context of the forward pass.
    Y_pred_grid, _ = model.forward(grid_points)

    # Extract the two output dimensions and reshape them to the meshgrid shape
    y1_output = Y_pred_grid[0, :].reshape(xx.shape)
    y2_output = Y_pred_grid[1, :].reshape(xx.shape)

    # Plot the decision boundaries for y1=0.5 and y2=0.5
    # These lines define where the output neuron's classification flips
    plt.contour(
        xx,
        yy,
        y1_output,
        levels=[0.5],
        colors="red",
        linestyles="dashed",
        linewidths=2,
    )
    plt.contour(
        xx,
        yy,
        y2_output,
        levels=[0.5],
        colors="blue",
        linestyles="dotted",
        linewidths=2,
    )

    # Plot the original data points
    # Map binary codes to integer labels for coloring
    group_map = {(1, 0): 0, (0, 0): 1, (1, 1): 2, (0, 1): 3}
    # Ensure Y is binary and integer for consistent mapping
    y_labels_binary = np.round(Y).astype(int)
    data_group_indices = [
        group_map.get(tuple(y_labels_binary[:, i]), -1) for i in range(Y.shape[1])
    ]

    # Use a colormap for the data points and ensure they are on top (zorder=2)
    plt.scatter(
        X[0, :],
        X[1, :],
        c=data_group_indices,
        cmap=plt.cm.RdYlBu,  # type: ignore
        edgecolors="k",
        s=80,
        zorder=2,
    )

    # Create a custom legend for the decision boundaries
    custom_lines = [
        Line2D([0], [0], color="red", linestyle="dashed", lw=2),
        Line2D([0], [0], color="blue", linestyle="dotted", lw=2),
    ]
    plt.legend(custom_lines, ["y1=0.5 boundary", "y2=0.5 boundary"], loc="upper left")

    plt.title(f"Decision Boundaries for {activation} ({epoch} Epochs)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_function_approximation(
    model,
    X_scaled,
    Y_scaled,
    epoch,
    activation,
    filename,
    X_original_unscaled,
    Y_original_unscaled,
    x_scaler,
    y_scaler,
):
    """
    Saves the function approximation plot for the regression problem.
    Uses original unscaled data for plotting and inverse transforms predictions.
    """
    plt.figure(figsize=(10, 6))

    # Generate a smoother x-range for the approximation curve
    x_smooth_original = np.linspace(
        X_original_unscaled.min(), X_original_unscaled.max(), 300
    ).reshape(1, -1)
    # Scale the smooth x-range for model prediction
    x_smooth_scaled = x_scaler.transform(x_smooth_original)

    # Get predictions from the model (on scaled data)
    y_pred_smooth_scaled, _ = model.forward(x_smooth_scaled)
    # Inverse transform predictions back to original scale
    y_pred_smooth_original = y_scaler.inverse_transform(y_pred_smooth_scaled)

    plt.scatter(
        X_original_unscaled,
        Y_original_unscaled,
        label="Actual Data Points",
        color="blue",
    )
    plt.plot(
        x_smooth_original.flatten(),
        y_pred_smooth_original.flatten(),
        label="NN Approximation",
        color="red",
        linewidth=2,
    )

    plt.title(f"Function Approximation for {activation} ({epoch} Epochs))")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
