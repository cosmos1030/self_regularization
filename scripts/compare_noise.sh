#!/bin/bash
# run_normal.sh
# Make sure to make it executable: chmod +x run_normal.sh

python compare_settings.py \
  --base_path multiseed_output \
  --experiments normal:alex_batch64_sgd_lr0.0001_epochs100 noise30:alex_batch64_sgd_lr0.0001_epochs100_noise30 noise50:alex_batch64_sgd_lr0.0001_epochs100_noise50 noise100:alex_batch64_sgd_lr0.0001_epochs100_noise100 \
  --fc_files fc1.csv fc2.csv fc3.csv \
  --plot_type normal \
  --columns S1_fit S1_RSS S2_fit S2_RSS \
  --output_dir visualization/compare_noise
