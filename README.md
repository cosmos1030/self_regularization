# Project Overview

This repository contains a set of scripts for **CIFAR-10** training and analysis using a custom **miniAlexNet** model. Key features include:

- **Training** miniAlexNet with various optimizers (SGD, Adam) and various hyperparameters on CIFAR-10.
- **Injecting noisy labels** for experiments (configurable noise ratio and random seeds).
- **Extracting leading eigenvectors** from convolution and linear layers every epoch.
- **Rank-k approximation experiments** on linear layers.
- **Riemannian optimization** with **Geoopt** and **Geomstats** for **Geodesic PCA** (GPCA).
- **Multi-seed experiments** to gather distributions of results (fitting scores, RSS, etc.).
- **Comparisons of different settings** (e.g., SGD vs. Adam) via boxplots.

## Folder/File Structure (Example)

```
.
├─ data/                          # CIFAR-10 data & (optionally) noisy label files
├─ runs/                          # Saved outputs for single-run experiments
├─ multiseed_runs/                # Saved outputs for multi-seed experiments
├─ multiseed_output_test/         # Aggregated CSV results from multi-seed experiments
├─ minialexnet_finished.py
├─ minialexnet_finished_noise.py
├─ minialexnet_multifactor.py
├─ label_noise_generator.py
├─ combined_dim_reduction.py
├─ combined_dim_reduction.sh
├─ multiseed_experiment_reviesed.py
├─ multiseed_experiment_noise_revised.py
├─ compare_settings.py
└─ README.md
```

Below is a brief explanation of each major script.

---

## 1. **label_noise_generator.py**
- **Purpose**:  
  - Generates or reloads noisy labels for the CIFAR-10 dataset. 
  - Allows specifying a `noise_ratio` (e.g., 0.7 = 70% noisy labels) and a random `seed` for reproducibility.
  - Saves or reloads the noisy labels in a `.npy` file (e.g., `noisy_labels50_seed5.npy`).
- **Key Points**:
  - Uses `introduce_label_noise()` to shuffle a portion of the labels to random incorrect classes.
  - If `reload=True` and the `.npy` file exists, it will load those labels instead of generating new ones.
- **Usage**:
  1. Adjust `noise_ratio` and `seed` as desired.
  2. Run:
     ```bash
     python label_noise_generator.py
     ```
  3. The script saves the noisy labels to `./data/noisy_labels{noise_level}_seed{seed}.npy`.

---

## 2. **minialexnet_finished.py**
- **Purpose**:  
  - Trains miniAlexNet on CIFAR-10 using **SGD** (default) or Adam (if you switch the optimizer code).
  - Saves leading eigenvectors and biases of the convolution and linear layers every epoch to `.csv` files.
  - Evaluates and plots test accuracy over epochs, and performs PCA on biases.
  - (Optional) Performs **rank-k approximation** experiments on the linear layers at specific epochs to see how test accuracy changes.
- **Key Steps**:
  1. Load CIFAR-10 dataset and define `miniAlexNet`.
  2. Train for `epochs`, logging training loss and test accuracy.
  3. After each epoch, perform SVD on the weights of each layer, save the first (leading) eigenvector, and also save each linear layer’s bias.
  4. Plot overall accuracy (`accuracy.csv` -> `accuracy.png`).
  5. Perform PCA on biases across epochs and save the figure (`bias_pca_analysis.png`).
  6. (If enabled) do rank-k factorization at certain epochs and record the normalized reconstruction accuracy.
- **Usage**:
  ```bash
  python minialexnet_finished.py
  ```
  (Inside the script, you can modify hyperparameters such as `epochs`, `learning_rate`, etc.)

---

## 3. **minialexnet_finished_noise.py**
- **Purpose**:  
  - Similar to `minialexnet_finished.py`, but trains with **noisy labels**. 
  - Assumes you already generated a `.npy` file with noisy labels (e.g., using `label_noise_generator.py`).
  - Loads these noisy labels and trains the network, storing leading eigenvectors, biases, and performance metrics.
- **Usage**:
  1. Make sure you have a `.npy` file with noisy labels.
  2. In the script, set `save_path` to point to that `.npy` file (and `noise_ratio`/`noise_seed` should match).
  3. Run:
     ```bash
     python minialexnet_finished_noise.py
     ```
  4. The script will log epoch-wise leading eigenvectors and biases in similar fashion.

---

## 4. **minialexnet_multifactor.py**
- **Purpose**:
  - Extends `minialexnet_finished.py` by conducting additional rank-k reconstruction tests for the linear layers at specific epochs.
  - Examines how test accuracy degrades (or not) for different SVD ranks `k`.
- **Usage**:
  ```bash
  python minialexnet_multifactor.py
  ```
  - Adjust the relevant code inside for the set of ranks/epochs at which you want to measure accuracy.

---

## 5. **combined_dim_reduction.py**
- **Purpose**:
  - Performs **Geodesic PCA (GPCA)** on the stored eigenvectors from each epoch. 
  - Uses [Geoopt](https://github.com/leoiri/geoopt) and [Geomstats](https://geomstats.github.io/) to fit the data on S¹ (1D) or S² (2D) on the hypersphere.
  - After fitting, it calculates the RSS (residual sum of squares) and a fitting score (1 − RSS / total variance).
  - Produces visualizations:
    - For 1D GPCA: geodesic parameter vs. epoch plots.
    - For 2D GPCA: sphere scatter, heatmaps of epochs/residuals, etc.
- **Key Arguments**:
  - `--dirs`: One or more `runs/...` directories to process.
  - `--layers`: Which layers (e.g., `fc1`, `fc2`, `fc3`) to analyze.
  - `--start-epoch`, `--end-epoch`: Range of epochs to consider.
  - `--gpca-dim`: 1 or 2 (project to S¹ or S²).
- **Usage Example**:
  ```bash
  python combined_dim_reduction.py \
    --dirs runs/alex_seed100_batch64_sgd_lr0.0001_epochs100 \
    --layers fc1 fc2 fc3 \
    --start-epoch 1 --end-epoch 100 \
    --gpca-dim 2
  ```

---

## 6. **combined_dim_reduction.sh**
- **Purpose**:
  - A shell script that ensures `geoopt` and `geomstats` are installed.
  - Then runs `combined_dim_reduction.py` for multiple directories in a batch manner.
- **Usage**:
  ```bash
  chmod +x combined_dim_reduction.sh
  ./combined_dim_reduction.sh
  ```
  - You can edit `DIRS=(runs/*/)` or other arrays in the file to batch-process multiple runs.

---

## 7. **multiseed_experiment_reviesed.py**
- **Purpose**:
  - Automates running the same experiment over multiple random seeds (e.g., seeds 1 to 30) without label noise.
  - For each seed:
    1. Trains miniAlexNet, saving the leading eigenvectors each epoch in `runs/...`.
    2. Performs 1D and 2D GPCA to compute `S1_fit`, `S2_fit`, `S1_RSS`, and `S2_RSS`.
    3. Appends results to CSV files (e.g., `fc1.csv`, `fc2.csv`, `fc3.csv`) under `multiseed_output_test/...`.
  - If the script detects that some seeds are already processed, it resumes from the next seed.
- **Usage**:
  ```bash
  python multiseed_experiment_reviesed.py
  ```
  - Modify `total_seeds`, `batch_size`, `epochs`, or other parameters in the script as needed.

---

## 8. **multiseed_experiment_noise_revised.py**
- **Purpose**:
  - Similar to `multiseed_experiment_reviesed.py`, but applies **noisy labels** (reloaded from a `.npy` file).
  - Trains each seed with the specified noise ratio. 
  - Saves `fc1.csv`, `fc2.csv`, `fc3.csv` results with GPCA fitting scores and RSS in `multiseed_output_test/...`.
- **Usage**:
  ```bash
  python multiseed_experiment_noise_revised.py
  ```
  - Ensure you have the appropriate `noisy_labelsX_seedY.npy` file and adjust `noise_ratio` + `noise_seed`.

---

## 9. **compare_settings.py**
- **Purpose**:
  - Loads the aggregated CSV results from multi-seed experiments (e.g., from `multiseed_output_test/...`) and compares across different conditions (optimizers, etc.) by generating boxplots.
  - For example, it might compare `test_acc`, `S1_fit`, `S2_fit`, or other metrics for `sgd` vs. `adam`.
- **Usage**:
  1. Ensure the CSV files from multi-seed experiments (e.g., `fc1.csv`, `fc2.csv`, `fc3.csv`) exist in the `multiseed_output_test/...` folder(s).
  2. Adjust the path to those CSVs inside `compare_settings.py`.
  3. Run:
     ```bash
     python compare_settings.py
     ```
  4. Boxplot images are saved (e.g., in a `visualization/` folder).

---

# Quick Start

1. **Generate or use existing CIFAR-10 data**:  
   - By default, running any training script will download CIFAR-10 if not present.

2. **(Optional) Generate noisy labels** (if you want to test noise experiments):
   ```bash
   python label_noise_generator.py
   ```
   - Adjust `noise_ratio` and `seed` as desired.

3. **Run a standard training** (example: `minialexnet_finished.py`):
   ```bash
   python minialexnet_finished.py
   ```
   - This produces a `runs/xxx/` directory with epoch-wise eigenvectors, biases, and accuracy logs.

4. **(Optional) Perform Geodesic PCA** on the saved eigenvectors:
   ```bash
   python combined_dim_reduction.py --dirs runs/my_experiment --layers fc1 fc2 fc3 --start-epoch 1 --end-epoch 100 --gpca-dim 2
   ```
   - Or run `./combined_dim_reduction.sh` to process multiple directories.

5. **Multi-seed experiments**:
   - Without label noise: `multiseed_experiment_reviesed.py`
   - With label noise: `multiseed_experiment_noise_revised.py`
   ```bash
   python multiseed_experiment_reviesed.py
   ```
   - Collects fitting scores (S1_fit, S2_fit, etc.) in `multiseed_output_test/...`.

6. **Compare settings** (`compare_settings.py`):
   - Generates boxplots for different metrics across optimizers or other hyperparameters. 
   ```bash
   python compare_settings.py
   ```
   - Check the `visualization/` folder for the resulting plots.

---

# Notes & Tips

- **GPU Usage**:  
  All scripts default to using `cuda` if available (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).  
- **Parameter Adjustment**:  
  Many hyperparameters (e.g., learning rates, epochs, seeds) are set inside the scripts. Adjust them to suit your experiments. Some scripts also accept command-line arguments, which you can see via `--help`.
- **Data Analysis**:  
  - `runs/.../eigenvectors/`: Contains epoch-wise leading eigenvector CSV files.  
  - `runs/.../biases/`: Contains epoch-wise biases for linear layers.  
  - `accuracy.csv`: epoch-wise test accuracy.  
  - `bias_pca_analysis.png`: PCA on biases across epochs.  
  - For GPCA analysis, check newly generated `.png` files for 1D or 2D geodesic embeddings and residuals.