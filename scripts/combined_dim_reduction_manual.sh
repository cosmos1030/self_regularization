
# DIRS=("/clkifford-data/home/doyoonkim/projects/R1-V_archive/self_regularization/runs/alex_seed100_batch8_sgd_lr0.001_epochs100_noise0_seed5_sub100_jse_eigen_final")
# DIRS=("/clifford-data/home/doyoonkim/projects/R1-V_archive/self_regularization/runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise0_seed5_sub100_jse_eigen_final")
# DIRS=("/clifford-data/home/doyoonkim/projects/R1-V_archive/self_regularization/runs/alex_seed100_batch64_sgd_lr0.0001_epochs100_noise0_seed5_sub100_jse_eigen_final")
DIRS=("/clifford-data/home/doyoonkim/projects/R1-V_archive/self_regularization/runs/alex_seed100_batch8_sgd_lr0.001_epochs100_noise0_seed5_sub100_jse_eigen_final")


# LAYERS=("fc1_last" "fc2_last" "fc3_last")
# LAYERS=("fc1_subleading" "fc2_subleading" "fc3_subleading")
# LAYERS=("fc1" "fc2" "fc3")
LAYERS=("fc2")

#LAYERS=("fc3")
START_EPOCH=46
END_EPOCH=80
GPCA_DIM=2

# 추적을 시작할 초기 인덱스
 INITIAL_EIG_INDEX=0

 # 수동으로 전환할 지점들 (공백으로 구분)
 # 형식: "에포크:새로운_인덱스"
 # 예: 35번째 에포크부터는 2번 인덱스를, 60번째 에포크부터는 1번 인덱스를 추적
# MANUAL_SWITCHES="35:2 60:1"
# MANUAL_SWITCHES="81:2"
# MANUAL_SWITCHES="22:0"



 # Run Python script 
 python combined_dim_reduction_manual.py \
     --dirs "${DIRS[@]}" \
     --layers "${LAYERS[@]}" \
     --start-epoch $START_EPOCH \
     --end-epoch $END_EPOCH \
     --gpca-dim $GPCA_DIM \
     --eig-index $INITIAL_EIG_INDEX \
     --manual-switch $MANUAL_SWITCHES