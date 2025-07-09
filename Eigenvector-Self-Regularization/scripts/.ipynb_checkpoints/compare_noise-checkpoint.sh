#!/bin/bash
# run_normal.sh
# Make sure to make it executable: chmod +x run_normal.sh

python compare_settings.py \
  --base_path multiseed_output_test \
  --experiments sgd:alex_batch64_sgd_lr0.0001_epochs100 adam:alex_batch64_adam_lr0.0001_epochs100 \
  --fc_files fc1.csv fc2.csv fc3.csv \
  --plot_type normal \
  --columns S1_fit S1_RSS S2_fit S2_RSS \
  --output_dir visualization/compare_optimizer
