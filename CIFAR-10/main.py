from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, random
from resnet import eval_step, init_model, update_step
from utils import load_cifar10


def train_experiment(use_residual=True, epochs=15):
    print(f"\nStarting Training (Residual={use_residual})...")
    key = random.key(13)
    train_loader, test_loader = load_cifar10(batch_size=128)

    params, state, config = init_model(key)
    # Initialize optimizer velocity as zeros matching params structure
    opt_state = jax.tree.map(lambda x: jnp.zeros_like(x), params)

    # Create JIT-compiled functions with static config baked in via partial.
    # This prevents JAX from trying to trace the 'config' dictionary or hash it incorrectly.
    update_fn = jit(partial(update_step, config=config, use_residual=use_residual))
    eval_fn = jit(partial(eval_step, config=config))

    train_accs, val_accs = [], []
    grad_norms = []

    for epoch in range(epochs):
        batch_accs = []
        batch_grads = []

        for x, y in train_loader:
            # Call the specialized jitted function
            params, state, opt_state, loss, acc, g_norm = update_fn(
                params, state, opt_state, x, y
            )
            batch_accs.append(acc)
            batch_grads.append(g_norm)

        # Validation
        val_batch_accs = []
        for x_val, y_val in test_loader:
            val_acc = eval_fn(params, state, x_val, y_val)
            val_batch_accs.append(val_acc)

        epoch_train_acc = np.mean(batch_accs)
        epoch_val_acc = np.mean(val_batch_accs)
        epoch_grad_norm = np.mean(batch_grads)

        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        grad_norms.append(epoch_grad_norm)

        print(
            f"Ep {epoch + 1} | Train: {epoch_train_acc:.3f} | Val: {epoch_val_acc:.3f} | Grad: {epoch_grad_norm:.3f}"
        )

    return train_accs, val_accs, grad_norms


if __name__ == "__main__":
    # Run Training
    # We train two models: One with Residual connections, one without (PlainNet) to demonstrate the effect on gradient flow.
    epochs = 20
    print("Training ResNet...")
    res_train, res_val, res_grads = train_experiment(use_residual=True, epochs=epochs)

    print("\nTraining PlainNet (No Skip Connections)...")
    plain_train, plain_val, plain_grads = train_experiment(
        use_residual=False, epochs=epochs
    )

    # Generate Report Figures
    print("\nGenerating Plots...")
    # Figure 1: Accuracy Comparison
    plt.figure(figsize=(6, 4))
    plt.plot(
        res_val,
        label="ResNet (Validation)",
        color="#1f77b4",
        linestyle="--",
    )
    plt.plot(
        plain_val,
        label="PlainNet (Validation)",
        color="#ff7f0e",
        linestyle="--",
    )
    plt.plot(res_train, label="ResNet (Train)", color="#1f77b4", alpha=0.7)
    plt.plot(plain_train, label="PlainNet (Train)", color="#ff7f0e", alpha=0.7)
    plt.title("Impact of Residual Connections on Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/accuracy_curve.png", dpi=300)

    # Figure 2: Gradient Flow
    plt.figure(figsize=(6, 4))
    plt.plot(res_grads, label="ResNet", color="#1f77b4", linewidth=2)
    plt.plot(
        plain_grads, label="PlainNet", color="#ff7f0e", linewidth=2, linestyle="--"
    )
    plt.title("Gradient Flow (L2 Norm of First Layer Gradients)")
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/grad_flow.png", dpi=300)

    print("\nPlots saved to figures/")
