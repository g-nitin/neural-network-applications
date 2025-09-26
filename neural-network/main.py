import os

import numpy as np
from NeuralNetwork import NeuralNetwork, Scaler
from plots import (
    plot_combined_error_vs_epoch,
    plot_decision_boundary,
    plot_error_vs_epoch,
    plot_function_approximation,
)

np.random.seed(13)


def Question_One():
    # Create a directory to save plots if it doesn't exist
    if not os.path.exists("plots/Q1"):
        os.makedirs("plots/Q1")

    # Question 1: One-Layer NN for Classification
    print("Starting Question 1: Classification Problem")
    X1 = np.array(
        [
            [0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
            [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3],
        ]
    )
    Y1 = np.array([[1, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    epochs_q1 = 1000
    learning_rate_q1 = 0.1
    batch_size_q1 = X1.shape[1]  # Full batch for Q1

    epochs_to_plot_q1 = [3, 10, 100, 1000]
    assert epochs_q1 == epochs_to_plot_q1[-1], "Ensure final epoch is in plot list"

    # Experiment with different activation functions
    # For Q1, initialization method is less critical as it's a simple classification
    # and the output activation is the primary focus. We'll use 'random' default.
    activations_to_test_q1 = ["tanh", "logsig", "tansig", "radialbasis", "relu"]
    results_q1 = {}
    q1_loss_histories = {}

    for activation in activations_to_test_q1:
        print(f"\nQ1: Testing with activation function: {activation}")
        # For Q1, it's a one-layer network, so only one activation function for the output layer.
        # Beta is not relevant here as it's not a hidden layer.
        nn_q1 = NeuralNetwork(layer_dims=[2, 2], activations=[activation])

        prefix = f"plots/Q1/{activation}"
        loss_history = nn_q1.train(
            X1,
            Y1,
            epochs=epochs_q1,
            learning_rate=learning_rate_q1,
            batch_size=batch_size_q1,
            activation=activation,
            plot_epochs=epochs_to_plot_q1,
            plot_callback=plot_decision_boundary,
            plot_filename_prefix=prefix,
            verbose=True,
        )

        results_q1[activation] = loss_history[-1]
        plot_error_vs_epoch(loss_history, f"plots/Q1/{activation}_error_curve.png")

        q1_loss_histories[activation] = loss_history

    plot_combined_error_vs_epoch(
        q1_loss_histories,
        "Q1: Training Error Comparison (Output Layer)",
        filename="plots/Q1/combined_error_curve.png",
    )

    print(f"\nQ1 Final Results (Loss at {epochs_to_plot_q1[-1]} epochs)")
    for activation, loss in results_q1.items():
        print(f"- {activation:<10}: {loss:.6f}")


def Question_Two():
    # Question 2: Two-Layer NN for Function Approximation
    print("\nStarting Question 2: Function Approximation Problem")

    # Create a directory to save plots if it doesn't exist
    if not os.path.exists("plots/Q2"):
        os.makedirs("plots/Q2")

    X2_original = np.array(
        [
            [
                -1,
                -0.9,
                -0.8,
                -0.7,
                -0.6,
                -0.5,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
            ]
        ]
    )
    Y2_original = np.array(
        [
            [
                -0.96,
                -0.577,
                -0.073,
                0.377,
                0.641,
                0.66,
                0.461,
                0.134,
                -0.201,
                -0.434,
                -0.5,
                -0.393,
                -0.165,
                0.099,
                0.307,
                0.396,
                0.345,
                0.182,
                -0.031,
                -0.219,
                -0.321,
            ]
        ]
    )

    # Initialize and fit scalers
    x_scaler = Scaler()
    y_scaler = Scaler()
    x_scaler.fit(X2_original)
    y_scaler.fit(Y2_original)

    # Scale the data for training
    X2_scaled = x_scaler.transform(X2_original)
    Y2_scaled = y_scaler.transform(Y2_original)

    epochs_q2 = 1000  # Increased epochs for better convergence with tuning
    epochs_to_plot_q2 = [10, 100, 200, 400, 1000]
    assert epochs_q2 == epochs_to_plot_q2[-1], "Ensure final epoch is in plot list"

    # Hyperparameter grids for tuning
    learning_rates = [0.001, 0.005, 0.01, 0.1]
    hidden_neuron_counts = [5, 10, 50, 100, 250, 500, 1000]
    batch_sizes = [1, 5, 10, X2_scaled.shape[1]]  # 1 (SGD), 5, 10, Full Batch
    initialization_methods = ["random", "xavier", "he"]

    # Activation functions for the hidden layer
    activations_to_test_q2 = ["tanh", "logsig", "tansig", "radialbasis", "relu"]

    # Store best configurations and their loss histories
    best_configs_q2 = {}
    q2_loss_histories_for_plotting = {}

    for activation in activations_to_test_q2:
        print(f"\nTuning for HIDDEN activation: {activation}")
        current_best_loss = float("inf")
        current_best_params = None
        current_best_loss_history = None

        # Iterate through hyperparameter combinations
        for init_method in initialization_methods:
            # He initialization is typically for ReLU, Xavier for tanh/sigmoid.
            # We'll test all combinations to see empirical results.
            if init_method == "he" and activation not in ["relu"]:
                # He init is less suitable for non-ReLU, but we'll allow it for comprehensive testing
                pass
            if init_method == "xavier" and activation in ["relu"]:
                # Xavier init is less suitable for ReLU, but we'll allow it
                pass

            for lr in learning_rates:
                for hn in hidden_neuron_counts:
                    for bs in batch_sizes:
                        print(
                            f"  Testing {activation} with Init={init_method}, LR={lr}, HN={hn}, BS={bs}..."
                        )
                        nn_q2 = NeuralNetwork(
                            layer_dims=[1, hn, 1],
                            activations=[activation, "linear"],
                            initialization_method=init_method,
                        )
                        loss_history = nn_q2.train(
                            X2_scaled,
                            Y2_scaled,
                            epochs=epochs_q2,
                            learning_rate=lr,
                            batch_size=bs,
                            verbose=False,
                        )
                        final_loss = loss_history[-1]

                        if final_loss < current_best_loss:
                            current_best_loss = final_loss
                            current_best_params = {
                                "learning_rate": lr,
                                "hidden_neurons": hn,
                                "batch_size": bs,
                                "initialization_method": init_method,
                            }
                            current_best_loss_history = loss_history

        best_configs_q2[activation] = {
            "params": current_best_params,
            "loss": current_best_loss,
            "loss_history": current_best_loss_history,
        }
        q2_loss_histories_for_plotting[activation] = current_best_loss_history
        print(
            f"  Best for {activation}: Loss={current_best_loss:.6f}, Params={current_best_params}"
        )

    print("\nGenerating Plots for Optimal Q2 Models")
    for activation, config in best_configs_q2.items():
        params = config["params"]
        print(f"\nQ2: Plotting for optimal {activation} (Loss: {config['loss']:.6f})")

        # Re-initialize and train the best model for plotting
        nn_q2_optimal = NeuralNetwork(
            layer_dims=[1, params["hidden_neurons"], 1],
            activations=[activation, "linear"],
            initialization_method=params["initialization_method"],
        )

        # Train again to get the model state at specific epochs for plotting
        prefix = f"plots/Q2/{activation}_optimal"
        nn_q2_optimal.train(
            X2_scaled,
            Y2_scaled,
            epochs=epochs_q2,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            activation=activation,
            plot_epochs=epochs_to_plot_q2,
            plot_callback=plot_function_approximation,
            plot_filename_prefix=prefix,
            verbose=True,
            X_original_unscaled=X2_original,
            Y_original_unscaled=Y2_original,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
        plot_error_vs_epoch(
            config["loss_history"], f"plots/Q2/{activation}_optimal_error_curve.png"
        )

    plot_combined_error_vs_epoch(
        q2_loss_histories_for_plotting,
        "Q2: Training Error Comparison (Optimal Hidden Layer Configs)",
        filename="plots/Q2/combined_optimal_error_curve.png",
    )

    print(f"\nQ2 Final Optimal Results (Loss at {epochs_q2} epochs)")
    for activation, config in best_configs_q2.items():
        print(
            f"- {activation:<12}: Loss={config['loss']:.6f}, Params={config['params']}"
        )


def Question_Two_Custom_Rerun():
    """
    Reruns Question 2 with pre-defined 'optimal' parameters for tanh, logsig, tansig,
    and custom (potentially sub-optimal) parameters for radialbasis and relu.
    """
    print("\nStarting Question 2 Custom Rerun: Function Approximation Problem")

    if not os.path.exists("plots/Q2_custom_rerun"):
        os.makedirs("plots/Q2_custom_rerun")

    X2_original = np.array(
        [
            [
                -1,
                -0.9,
                -0.8,
                -0.7,
                -0.6,
                -0.5,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
            ]
        ]
    )
    Y2_original = np.array(
        [
            [
                -0.96,
                -0.577,
                -0.073,
                0.377,
                0.641,
                0.66,
                0.461,
                0.134,
                -0.201,
                -0.434,
                -0.5,
                -0.393,
                -0.165,
                0.099,
                0.307,
                0.396,
                0.345,
                0.182,
                -0.031,
                -0.219,
                -0.321,
            ]
        ]
    )

    # Initialize and fit scalers
    x_scaler = Scaler()
    y_scaler = Scaler()
    x_scaler.fit(X2_original)
    y_scaler.fit(Y2_original)

    # Scale the data for training
    X2_scaled = x_scaler.transform(X2_original)
    Y2_scaled = y_scaler.transform(Y2_original)

    epochs_q2 = 1000
    epochs_to_plot_q2 = [10, 100, 200, 400, 1000, 1000]
    assert epochs_q2 == epochs_to_plot_q2[-1], "Ensure final epoch is in plot list"

    # Hardcoded "Best" Parameters for tanh, logsig, tansig (from previous tuning)
    best_configs_others = {
        "tanh": {
            "params": {
                "learning_rate": 0.01,
                "hidden_neurons": 10,
                "batch_size": 1,
                "initialization_method": "he",
            },
            "loss": 0.000792,
            "loss_history": None,
        },
        "logsig": {
            "params": {
                "learning_rate": 0.1,
                "hidden_neurons": 10,
                "batch_size": 1,
                "initialization_method": "he",
            },
            "loss": 0.002011,
            "loss_history": None,
        },
        "tansig": {
            "params": {
                "learning_rate": 0.1,
                "hidden_neurons": 10,
                "batch_size": 5,
                "initialization_method": "he",
            },
            "loss": 0.000884,
            "loss_history": None,
        },
    }

    # Custom Parameters for radialbasis and relu (to make them 'less good' or explore)
    custom_configs_specific = {
        "radialbasis": {
            "params": {
                "learning_rate": 0.01,
                # "hidden_neurons": 500,
                "hidden_neurons": 50,
                "batch_size": 5,
                "initialization_method": "he",
            },
            "loss": float("inf"),  # Will be set during training below
            "loss_history": None,
        },
        "relu": {
            "params": {
                "learning_rate": 0.01,
                # "hidden_neurons": 100,
                "hidden_neurons": 50,
                "batch_size": 1,
                "initialization_method": "he",
            },
            "loss": float("inf"),  # Will be set during training below
            "loss_history": None,
        },
    }

    # Combine all configurations to run
    configs_to_run = {**best_configs_others, **custom_configs_specific}
    q2_loss_histories_for_plotting = {}

    print("\nRunning Models with Specified Configurations")
    for activation, config in configs_to_run.items():
        params = config["params"]
        print(f"\nRunning {activation} with params: {params}")

        nn_q2 = NeuralNetwork(
            layer_dims=[1, params["hidden_neurons"], 1],
            activations=[activation, "linear"],
            initialization_method=params["initialization_method"],
        )

        prefix = f"plots/Q2_custom_rerun/{activation}"
        loss_history = nn_q2.train(
            X2_scaled,
            Y2_scaled,
            epochs=epochs_q2,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            activation=activation,
            plot_epochs=epochs_to_plot_q2,
            plot_callback=plot_function_approximation,
            plot_filename_prefix=prefix,
            verbose=True,
            X_original_unscaled=X2_original,
            Y_original_unscaled=Y2_original,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )

        config["loss"] = loss_history[-1]
        config["loss_history"] = loss_history
        plot_error_vs_epoch(
            loss_history, f"plots/Q2_custom_rerun/{activation}_error_curve.png"
        )
        q2_loss_histories_for_plotting[activation] = loss_history

    plot_combined_error_vs_epoch(
        q2_loss_histories_for_plotting,
        "Q2: Training Error Comparison (Custom Rerun)",
        filename="plots/Q2_custom_rerun/combined_error_curve.png",
    )

    print(f"\nQ2 Custom Rerun Final Results (Loss at {epochs_q2} epochs)")
    for activation, config in configs_to_run.items():
        print(
            f"- {activation:<12}: Loss={config['loss']:.6f}, Params={config['params']}"
        )


if __name__ == "__main__":
    # Question_One()
    # Question_Two()
    Question_Two_Custom_Rerun()
