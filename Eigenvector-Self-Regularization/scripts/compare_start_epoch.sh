#!/bin/bash
# run_normal.sh
# Make sure to make it executable: chmod +x run_normal.sh

python compare_settings.py \
  --base_path multiseed_output_test \
  --experiments sgd:alex_batch64_sgd_lr0.0001_epochs100 \
  --fc_files fc1.csv fc2.csv fc3.csv \
  --plot_type epoch \
  --epoch_bases S1_fit S1_RSS S2_fit S2_RSS \
  --epochs 5 \
  --output_dir visualization/compare_start_epoch
