#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
# Point this at your actual run directory:
base_dir = "runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_jse"
epoch    = 100

# Which layers to plot
layers   = ["fc1", "fc2", "fc3"]

# ─────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────
for layer in layers:
    # Build the CSV file paths
    csv_raw  = os.path.join(base_dir, f"recon_{layer}_acc_epoch{epoch}.csv")
    csv_vec  = os.path.join(base_dir, f"recon_{layer}_vec_epoch{epoch}.csv")
    csv_full = os.path.join(base_dir, f"recon_{layer}_full_epoch{epoch}.csv")

    # Skip if any file is missing
    if not all(os.path.exists(p) for p in (csv_raw, csv_vec, csv_full)):
        print(f"[WARNING] Skipping {layer}: missing CSV for epoch {epoch}")
        continue

    # Load accuracies
    acc_raw  = np.loadtxt(csv_raw)[20:-1]
    acc_vec  = np.loadtxt(csv_vec)[20:-1]
    acc_full = np.loadtxt(csv_full)[20:-1]

    # X‐axis: rank k = 1 … K
    ks = np.arange(20+1, len(acc_raw) + 20+1)

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.plot(ks, acc_raw,  '-', label="Baseline (top-k)")
    plt.plot(ks, acc_vec,  '--', label="JS-vec")
    plt.plot(ks, acc_full, ':', label="JS-full")

    plt.xlabel("Rank k")
    plt.ylabel("Test Accuracy")
    plt.title(f"Reconstruction Comparison at Epoch {epoch} — {layer}")
    plt.legend()
    plt.grid(True)

    # Save figure
    out_path = os.path.join(base_dir, f"recon_compare_{layer}_ep{epoch}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")
