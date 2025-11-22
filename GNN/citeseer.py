import numpy as np
from helpers import (
    AdamOptimizer,
    d_relu,
    dropout,
    get_normalized_adj,
    glorot_init,
    plot_loss,
    plot_tsne,
    relu,
    softmax,
)
from torch_geometric.datasets import Planetoid


def task_a_citeseer_node_classification():
    print("\n****** TASK A: Citeseer Node Classification ******")

    # 1. Load Data
    dataset = Planetoid(root="/tmp/Citeseer", name="Citeseer")
    data = dataset[0]

    # Convert to NumPy
    X = data.x.numpy()
    Y = data.y.numpy()
    edge_index = data.edge_index.numpy()
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    num_nodes, input_dim = X.shape
    num_classes = dataset.num_classes

    # 2. Preprocessing (Adjacency Normalization)
    print("Normalizing Adjacency Matrix...")
    A_norm = get_normalized_adj(edge_index, num_nodes)

    # Architecture: Input(3703) -> Dropout -> GCN(16) -> ReLU -> Dropout -> GCN(6) -> Softmax

    # 3. Initialize Weights
    hidden_dim = 16
    W0 = glorot_init(input_dim, hidden_dim)
    W1 = glorot_init(hidden_dim, num_classes)

    # Optimizer with Weight Decay
    optimizer = AdamOptimizer([W0, W1], lr=0.01, weight_decay=5e-4)

    # 4. Training Loop
    epochs = 300
    losses = []

    for epoch in range(epochs):
        # Forward Pass
        # Dropout on Input
        X_drop = dropout(X, 0.5, training=True)

        # Layer 1: H1 = ReLU(A_norm @ X @ W0)
        Z0 = A_norm @ X_drop @ W0
        H1 = relu(Z0)
        H1_drop = dropout(H1, 0.5, training=True)

        # Layer 2: Z = Softmax(A_norm @ H1 @ W1)
        Z1 = A_norm @ H1_drop @ W1
        probs = softmax(Z1)

        # Loss
        probs_train = probs[train_mask]
        y_train = Y[train_mask]

        # CE Loss: -sum(y * log(p))
        # Create one-hot for labels
        y_one_hot = np.eye(num_classes)[y_train]
        loss = -np.sum(y_one_hot * np.log(probs_train + 1e-15)) / len(y_train)
        losses.append(loss)

        # Backward Pass
        # Gradient of Loss w.r.t Z1 (Logits) = (Probs - OneHot) / N
        dZ1 = probs.copy()
        dZ1[train_mask] -= y_one_hot
        dZ1[~train_mask] = 0  # No gradient from non-train nodes
        dZ1 /= len(y_train)

        # Gradient w.r.t W1: (AH1)^T @ dZ1
        dW1 = (A_norm @ H1_drop).T @ dZ1

        # Gradient w.r.t H1: (dZ1 @ W1^T) -> backprop through A_norm
        dAH1 = dZ1 @ W1.T
        dH1_drop = A_norm.T @ dAH1
        dH1 = dropout(dH1_drop, 0.5, training=True)  # Backprop through dropout

        # Gradient w.r.t Z0 (through ReLU): dH1 * (Z0 > 0)
        dZ0 = dH1 * d_relu(Z0)

        # Gradient w.r.t W0: (AX)^T @ dZ0
        dW0 = (A_norm @ X_drop).T @ dZ0

        # Optimization Step
        optimizer.step([dW0, dW1])

        if epoch % 30 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

    # 5. Evaluation
    # Forward pass on full graph
    H1_eval = relu(A_norm @ X @ W0)
    logits = A_norm @ H1_eval @ W1
    preds = np.argmax(logits, axis=1)

    # Calculate Accuracy on Test Mask
    acc = np.mean(preds[test_mask] == Y[test_mask])
    print(f"Final Test Accuracy: {acc:.4f}")

    # Visualizations
    plot_loss(losses, "Citeseer Training Loss", "figures/task_a/loss_curve.png")
    plot_tsne(
        H1_eval, Y, "Citeseer Hidden Representations (t-SNE)", "figures/task_a/tsne.png"
    )
