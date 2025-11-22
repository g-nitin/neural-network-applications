import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Create directories for figures
os.makedirs("figures/task_a", exist_ok=True)
os.makedirs("figures/task_b", exist_ok=True)


# 1. Math & Activations
def glorot_init(fin, fout):
    """Xavier/Glorot Initialization"""
    limit = np.sqrt(6.0 / (fin + fout))
    return np.random.uniform(-limit, limit, (fin, fout))


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)


def softmax(x):
    """Stable Softmax"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dropout(x, drop_prob=0.5, training=True):
    """Inverted Dropout implementation"""
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > drop_prob) / (1 - drop_prob)
    return x * mask


# 2. Graph Utils
def get_normalized_adj(edge_index, num_nodes):
    """
    Computes D^(-1/2) * (A + I) * D^(-1/2)
    """
    # Create dense adjacency matrix
    A = np.zeros((num_nodes, num_nodes))
    for src, dst in edge_index.T:
        A[src, dst] = 1
        A[dst, src] = 1  # Undirected

    # Add self-loops
    A_hat = A + np.eye(num_nodes)

    # Compute Degree Matrix D
    D_hat_diag = np.sum(A_hat, axis=1)

    # Compute D^(-1/2)
    # Handle division by zero for isolated nodes
    with np.errstate(divide="ignore"):
        D_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)

    # Symmetric normalization
    return D_inv_sqrt_mat @ A_hat @ D_inv_sqrt_mat


# 3. Optimizer (Adam + L2 Regularization)
class AdamOptimizer:
    def __init__(self, params, lr=0.01, weight_decay=5e-4):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Add L2 Regularization term to gradient: grad += lambda * param
            grad = grad + self.weight_decay * param

            self.m[i] = 0.9 * self.m[i] + 0.1 * grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * (grad**2)

            m_hat = self.m[i] / (1 - 0.9**self.t)
            v_hat = self.v[i] / (1 - 0.999**self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# Visualization Utils
def plot_loss(losses, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()


def plot_tsne(embeddings, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename)
    plt.close()
