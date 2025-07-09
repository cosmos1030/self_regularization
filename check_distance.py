import numpy as np
import pandas as pd
import os

# 경로 설정
# run_name = 'alex_seed100_batch64_sgd_lr0.0001_epochs100_noise0_seed5_sub100_jse_eigen_various'
run_name ='alex_seed200_batch8_sgd_lr0.001_epochs100_noise0_seed5_sub100_jse_eigen_various'
base_dir = f'/clifford-data/home/doyoonkim/projects/R1-V_archive/self_regularization/runs/{run_name}/eigenvectors'
lead_path = os.path.join(base_dir, 'fc2_leading.csv')
sub_path  = os.path.join(base_dir, 'fc2_subleading.csv')

# CSV 로드
lead_vecs = pd.read_csv(lead_path, sep='\s+', header=None).values
sub_vecs  = pd.read_csv(sub_path,  sep='\s+', header=None).values

# 부호 정규화 (sign alignment)
for i in range(1, len(lead_vecs)):
    if np.dot(lead_vecs[i], lead_vecs[i - 1]) < 0:
        lead_vecs[i] *= -1
    if np.dot(sub_vecs[i], sub_vecs[i - 1]) < 0:
        sub_vecs[i] *= -1

# epoch별 leading → next leading vs next subleading 유사도 비교
switch_epochs = []
for t in range(len(lead_vecs) - 1):
    v_prev = lead_vecs[t]
    v_lead_next = lead_vecs[t + 1]
    v_sub_next  = sub_vecs[t + 1]

    sim_lead = np.abs(np.dot(v_prev, v_lead_next)) / (np.linalg.norm(v_prev) * np.linalg.norm(v_lead_next))
    sim_sub  = np.abs(np.dot(v_prev, v_sub_next))  / (np.linalg.norm(v_prev) * np.linalg.norm(v_sub_next))

    if sim_sub > sim_lead:
        print(f"🔀 Possible switch: epoch {t} → {t+1} (subleading more similar)")
        switch_epochs.append(t + 1)
