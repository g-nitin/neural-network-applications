import jax
import jax.numpy as jnp
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# Data Loading (using Torchvision only for IO)
def load_cifar10(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.permute(1, 2, 0).numpy(),  # Convert CHW to HWC for JAX
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
        drop_last=True,
    )

    return train_loader, test_loader


def relu(x):
    return jnp.maximum(0, x)


# Layer Definitions (Functional Approach)
def init_conv(key, in_c, out_c, k_size, stride):
    # He Initialization
    w_key, b_key = jax.random.split(key)
    scale = jnp.sqrt(2.0 / (in_c * k_size * k_size))
    w = jax.random.normal(w_key, (out_c, in_c, k_size, k_size)) * scale
    b = jnp.zeros((out_c,))

    params = {"w": w, "b": b}
    config = {"stride": stride, "padding": "SAME"}
    return params, config


def init_bn(channels):
    # Gamma=1, Beta=0
    params = {"gamma": jnp.ones((channels,)), "beta": jnp.zeros((channels,))}
    # Running mean=0, Running var=1
    state = {"mean": jnp.zeros((channels,)), "var": jnp.ones((channels,))}
    return params, state


def init_dense(key, in_dim, out_dim):
    # Xavier/Glorot
    w_key, b_key = jax.random.split(key)
    scale = jnp.sqrt(6.0 / (in_dim + out_dim))
    w = jax.random.uniform(w_key, (in_dim, out_dim), minval=-scale, maxval=scale)
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


# Forward Pass Helpers
def conv_forward(params, x, config):
    # x: (B, H, W, C) -> JAX expects (B, C, H, W) usually for NCHW,
    # but lax.conv_general_dilated defaults are flexible.
    # We will stick to NHWC (standard for TF/JAX usually) to minimize transposes.
    # Input x is (B, H, W, C). Weights are (Out, In, K, K).

    w, b = params["w"], params["b"]
    stride = config["stride"]

    # dimension_numbers=('NHWC', 'OIHW', 'NHWC') means:
    # Input: NHWC, Kernel: OIHW, Output: NHWC
    out = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=config["padding"],
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    return out + b


def bn_forward(params, state, x, is_training, momentum=0.9):
    # x shape: (B, H, W, C)
    gamma, beta = params["gamma"], params["beta"]

    if is_training:
        # Calculate batch stats
        mean = jnp.mean(x, axis=(0, 1, 2))
        var = jnp.var(x, axis=(0, 1, 2))

        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + 1e-5)

        # Update running stats (Exponential Moving Average)
        # Note: We use (1-m)*old + m*new convention or similar.
        # Standard is: new_running = mom * old_running + (1-mom) * batch_stat
        new_mean = momentum * state["mean"] + (1 - momentum) * mean
        new_var = momentum * state["var"] + (1 - momentum) * var
        new_state = {"mean": new_mean, "var": new_var}
    else:
        # Use running stats
        mean = state["mean"]
        var = state["var"]
        x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
        new_state = state

    # Scale and Shift
    out = x_norm * gamma + beta
    return out, new_state


def dense_forward(params, x):
    return jnp.dot(x, params["w"]) + params["b"]


def global_avg_pool(x):
    # x: (B, H, W, C) -> (B, C)
    return jnp.mean(x, axis=(1, 2))
