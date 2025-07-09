#!/bin/bash

# Function to check if a Python package is installed
check_and_install_package() {
    package_name=$1
    if ! python -c "import $package_name" &> /dev/null; then
        echo "$package_name is not installed. Installing..."
        pip install $package_name
    else
        echo "$package_name is already installed."
    fi
}

# Ensure Python dependencies are installed
check_and_install_package geomstats
check_and_install_package geoopt

# Define Directories, Layers, and Epoch Ranges
#DIRS=("runs/alex_seed100_batch64_sgd_lr0.0001_epochs100" "runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise100" "runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise50.0" "runs/alex_seed100_batch64_sgd_lr0.01_epochs100" "runs/alex_seed100_batch8_sgd_lr0.0001_epochs100" "runs/alex_seed100_batch8_sgd_lr0.001_epochs100")
# DIRS=("runs/alex_seed100_batch64_sgd_lr0.0001_epochs100")
#DIRS=("runs/alex_seed100_batch8_sgd_lr0.001_epochs100")
#DIRS=("runs/alex_seed200_batch64_adam_lr0.0001_epochs100" "runs/alex_seed200_batch64_sgd_lr0.0001_epochs100" "runs/alex_seed200_batch64_sgd_lr0.0001_epochs100_noise100_noiseseed5" "runs/alex_seed200_batch64_adam_lr0.0001_epochs100_noise50.0_noiseseed5" "runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise100_noiseseed5" "runs/alex_seed200_batch64_adam_lr0.0001_epochs100_ noise50.0_noiseseed5")
#DIRS=("runs/alex_seed100_batch64_sgd_lr0.001_epochs100" "runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise50.0_noiseseed5")
#DIRS=("runs/alex_seed100_batch8_sgd_lr0.0001_epochs100")
# DIRS=(runs/*/)
# DIRS=("runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise0_seed5_sub100_binary")
DIRS=("/clifford-data/home/doyoonkim/projects/R1-V_archive/self_regularization/runs/alex_seed100_batch8_sgd_lr0.001_epochs100_noise0_seed5_sub100_jse_eigen_various")

# LAYERS=("fc1" "fc2" "fc3")
LAYERS=("fc2")
START_EPOCH=1
END_EPOCH=100
GPCA_DIM=2

# Run Python script
python combined_dim_reduction_multi_trajectory.py --dirs "${DIRS[@]}" --layers "${LAYERS[@]}" --start-epoch $START_EPOCH --end-epoch $END_EPOCH --gpca-dim $GPCA_DIM --c