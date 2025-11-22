# GNN

This folder contains small Graph Neural Network (GNN) experiments and utilities for the Cora and Citeseer datasets. The scripts implement dataset-specific experiments and helpers to load data, run models, and save result figures.

## Files

- `cora.py` — Experiment script for the Cora dataset.
- `citeseer.py` — Experiment script for the Citeseer dataset.
- `helpers.py` — Helper functions for data loading, preprocessing, and plotting.
- `main.py` — Optional entry point to run experiments or utilities (project-specific).
- `figures/` — Output folder where result plots are saved (`task_a/`, `task_b/`).

## Quick start

1. From this folder, run `python main.py` which, by default runs both citeseer and cora tasks. Change the `main` function in `main.py` to choose specific tasks or configurations.

2. Check `figures/task_a/` and `figures/task_b/` for generated plots and results.
