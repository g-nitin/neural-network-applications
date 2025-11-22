# CIFAR-10 Residual Networks

Small JAX project that compares a shallow residual network against a plain CNN on CIFAR-10. It highlights how skip connections improve convergence and gradient flow.

## Running the experiment

From this directory run:

```bash
python main.py
```

The script downloads CIFAR-10 (if needed), trains both models for 20 epochs, and prints accuracy/gradient statistics each epoch.

## Outputs

Training curves and gradient norm plots are saved to `figures/accuracy_curve.png` and `figures/grad_flow.png`. Downloaded dataset batches live under `data/cifar-10-batches-py/`, _which is omitted on GitHub due to space constraints_.
