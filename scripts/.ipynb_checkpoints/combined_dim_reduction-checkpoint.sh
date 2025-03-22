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
# DIRS=("runs/alex_seed100_batch64_sgd_lr0.0001_epochs100" "runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise100" "runs/alex_seed100_batch64_sgd_lr0.01_epochs100" "runs/alex_seed100_batch8_sgd_lr0.0001_epochs100" "runs/alex_seed100_batch8_sgd_lr0.001_epochs100")
# DIRS=("runs/alex_seed100_batch64_sgd_lr0.0001_epochs100")
#DIRS=("runs/alex_seed100_batch8_sgd_lr0.001_epochs100")
DIRS=("runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise50.0")


LAYERS=("fc1" "fc2" "fc3")
START_EPOCH=1
END_EPOCH=100

# Run Python script
python leading_eigenvector_dim_reduction_final.py --dirs "${DIRS[@]}" --layers "${LAYERS[@]}" --start-epoch $START_EPOCH --end-epoch $END_EPOCH
