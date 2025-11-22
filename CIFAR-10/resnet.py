import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from utils import (
    bn_forward,
    conv_forward,
    dense_forward,
    global_avg_pool,
    init_bn,
    init_conv,
    init_dense,
    relu,
)


# ResNet Architecture
def init_res_block(key, in_c, out_c, stride=1):
    k1, k2, k3 = random.split(key, 3)

    # Main path
    conv1, cfg1 = init_conv(k1, in_c, out_c, 3, stride)
    bn1, st1 = init_bn(out_c)
    conv2, cfg2 = init_conv(k2, out_c, out_c, 3, 1)
    bn2, st2 = init_bn(out_c)

    params = {"c1": conv1, "b1": bn1, "c2": conv2, "b2": bn2}
    state = {"b1": st1, "b2": st2}
    config = {"c1": cfg1, "c2": cfg2}

    # Shortcut path (if dimensions change)
    if stride > 1 or in_c != out_c:
        conv_s, cfg_s = init_conv(k3, in_c, out_c, 1, stride)
        bn_s, st_s = init_bn(out_c)
        params["cs"] = conv_s
        params["bs"] = bn_s
        state["bs"] = st_s
        config["cs"] = cfg_s
        config["use_shortcut"] = True
    else:
        config["use_shortcut"] = False

    return params, state, config


def res_block_forward(params, state, config, x, is_training, use_residual=True):
    # Save input for residual
    residual = x

    # First Conv
    out = conv_forward(params["c1"], x, config["c1"])
    out, st1 = bn_forward(params["b1"], state["b1"], out, is_training)
    out = relu(out)

    # Second Conv
    out = conv_forward(params["c2"], out, config["c2"])
    out, st2 = bn_forward(params["b2"], state["b2"], out, is_training)

    new_state = {"b1": st1, "b2": st2}

    # Shortcut handling
    if config["use_shortcut"]:
        residual = conv_forward(params["cs"], residual, config["cs"])
        residual, st_s = bn_forward(params["bs"], state["bs"], residual, is_training)
        new_state["bs"] = st_s

    # Addition (Skip Connection)
    if use_residual:
        out = out + residual

    out = relu(out)
    return out, new_state


# Initialize Full Model
def init_model(key):
    # Architecture:
    # Conv(16) -> ResBlock(16) -> ResBlock(32) -> ResBlock(64) -> GAP -> Dense(10)
    keys = random.split(key, 5)

    p_c1, cfg_c1 = init_conv(keys[0], 3, 16, 3, 1)
    p_b1, st_b1 = init_bn(16)

    p_r1, st_r1, cfg_r1 = init_res_block(keys[1], 16, 16, 1)
    p_r2, st_r2, cfg_r2 = init_res_block(keys[2], 16, 32, 2)
    p_r3, st_r3, cfg_r3 = init_res_block(keys[3], 32, 64, 2)

    # init_dense returns a single dict, do not unpack with `p_fc, _`
    p_fc = init_dense(keys[4], 64, 10)

    params = {"c1": p_c1, "b1": p_b1, "r1": p_r1, "r2": p_r2, "r3": p_r3, "fc": p_fc}
    state = {"b1": st_b1, "r1": st_r1, "r2": st_r2, "r3": st_r3}
    config = {"c1": cfg_c1, "r1": cfg_r1, "r2": cfg_r2, "r3": cfg_r3}

    return params, state, config


def model_forward(params, state, config, x, is_training=True, use_residual=True):
    # Initial Conv
    x = conv_forward(params["c1"], x, config["c1"])
    x, st_b1 = bn_forward(params["b1"], state["b1"], x, is_training)
    x = relu(x)

    # ResBlocks
    x, st_r1 = res_block_forward(
        params["r1"], state["r1"], config["r1"], x, is_training, use_residual
    )
    x, st_r2 = res_block_forward(
        params["r2"], state["r2"], config["r2"], x, is_training, use_residual
    )
    x, st_r3 = res_block_forward(
        params["r3"], state["r3"], config["r3"], x, is_training, use_residual
    )

    # Classifier
    x = global_avg_pool(x)
    logits = dense_forward(params["fc"], x)

    new_state = {"b1": st_b1, "r1": st_r1, "r2": st_r2, "r3": st_r3}
    return logits, new_state


# Optimization & Training Loop
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, 10)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


def compute_loss(params, state, config, x, y, use_residual):
    logits, new_state = model_forward(params, state, config, x, is_training=True, use_residual=use_residual)
    loss = cross_entropy_loss(logits, y)
    # L2 Regularization
    l2_loss = 1e-4 * sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
    return loss + l2_loss, (new_state, logits)


def update_step(params, state, opt_state, x, y, config, use_residual=True):
    # We explicitly pass arguments to compute_loss in the order it expects
    (loss, (new_state, logits)), grads = value_and_grad(compute_loss, has_aux=True)(
        params, state, config, x, y, use_residual
    )

    # Manual SGD with Momentum
    lr = 0.01
    momentum = 0.9

    # Update velocity: v = m * v + g
    new_opt_state = jax.tree.map(lambda v, g: momentum * v + g, opt_state, grads)
    # Update params: w = w - lr * v
    new_params = jax.tree.map(lambda p, v: p - lr * v, params, new_opt_state)

    # Calculate accuracy
    acc = jnp.mean(jnp.argmax(logits, -1) == y)

    # Extract gradient norm of the very first layer for analysis
    grad_norm_first_layer = jnp.linalg.norm(grads["c1"]["w"])

    return new_params, new_state, new_opt_state, loss, acc, grad_norm_first_layer


def eval_step(params, state, x, y, config):
    logits, _ = model_forward(params, state, config, x, is_training=False)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return acc
