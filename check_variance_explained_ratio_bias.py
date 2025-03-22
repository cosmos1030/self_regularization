import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

bias_dir = "runs/alex_seed100_batch64_sgd_lr0.001_epochs100/biases"  # 경로 수정 필요
files = sorted([f for f in os.listdir(bias_dir) if f.endswith('.csv')])

pca_results = {}

for file in files:
    file_path = os.path.join(bias_dir, file)
    bias_data = np.loadtxt(file_path, delimiter=' ')

    if len(bias_data.shape) == 1:  # Handle 1D case (single epoch)
        bias_data = bias_data.reshape(1, -1)

    # 전체 bias 데이터에 PCA 적용
    pca = PCA(n_components=1)
    pca.fit(bias_data)  # 전체 데이터로 PCA 학습

    # 첫 번째 주성분이 전체 분산을 얼마나 설명하는지 확인
    explained_variance_ratio = pca.explained_variance_ratio_[0]

    # 각 epoch마다 첫 번째 주성분(PC1) 방향으로 투영된 크기 저장
    principal_components = np.abs(pca.transform(bias_data)).flatten()

    pca_results[file] = (explained_variance_ratio, principal_components)

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 5))
for layer, (variance_ratio, pc_values) in pca_results.items():
    layer_name = layer.split('.')[0]  # 확장자 제거
    ax.plot(pc_values, label=f"{layer_name} (Var: {variance_ratio:.2f})")  # 레전드에 variance 추가

ax.set_xlabel('Epochs')
ax.set_ylabel('PC1 Magnitude')
ax.set_title('Bias PCA First Principal Component Magnitude Over Epochs')
ax.legend()
plt.grid()
plt.savefig(os.path.join(bias_dir, 'bias_pca_analysis.png'))
plt.show()
