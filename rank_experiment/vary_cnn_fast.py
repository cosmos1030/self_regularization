#!/usr/bin/env python
# =================================================================
#   실험 2: 정규화된 SVD 기반 랭크 재구성에 따른 성능 비교
#   (CNN_0, CNN_1 모델만)
# =================================================================

import os
import shutil
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# 0. GPU 설정 (필요에 따라 수정)
# -----------------------------------------------------------------
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------
# 1. 하이퍼파라미터 및 경로 설정
# -----------------------------------------------------------------
batch_size    = 64
seed          = 100
learning_rate = 0.001
epochs        = 20
regularization_alpha = 0.01

results_dir = "experiment_regularized_svd"
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

# -----------------------------------------------------------------
# 2. CIFAR-10 데이터 로딩
# -----------------------------------------------------------------
print("==> Preparing CIFAR-10 data...")
tfm = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
])
trainset = torchvision.datasets.CIFAR10("./data", train=True,  transform=tfm, download=True)
testset  = torchvision.datasets.CIFAR10("./data", train=False, transform=tfm, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=4)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4)
print("==> Data preparation complete.")

# -----------------------------------------------------------------
# 3. 모델 정의
# -----------------------------------------------------------------
class ClassifierHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(),  # 타겟 레이어
            nn.Linear(192, 10)
        )
    def forward(self, x):
        return self.layers(x)

class CNN_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = ClassifierHead(in_features=3*32*32)
    def forward(self, x):
        return self.classifier(torch.flatten(x,1))

class CNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,96,5,1,2), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.classifier = ClassifierHead(in_features=96*16*16)
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x),1))

# -----------------------------------------------------------------
# 4. 정규화된 SVD 재구성 함수
# -----------------------------------------------------------------
def regularized_svd_recon(W_cpu, k, alpha):
    W = W_cpu.to(torch.float32)
    p, n = W.shape
    # 원본 특이값
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    # 공분산 행렬에 정규화 추가
    C_u_reg = W @ W.T + alpha * torch.eye(p, device=W.device)
    C_v_reg = W.T @ W   + alpha * torch.eye(n, device=W.device)
    # 정규화 공분산 고유벡터
    _, U_reg = torch.linalg.eigh(C_u_reg)
    U_reg = torch.flip(U_reg, dims=[-1])
    _, V_reg = torch.linalg.eigh(C_v_reg)
    V_reg = torch.flip(V_reg, dims=[-1])
    # rank-k 근사
    Vh_reg = V_reg.T
    return U_reg[:, :k] @ torch.diag(S[:k]) @ Vh_reg[:k, :]

# -----------------------------------------------------------------
# 5. 정확도 측정 함수
# -----------------------------------------------------------------
def accuracy(model, loader, device,
             k=None, original_weights=None,
             target_idx=None, target_weight_cpu=None,
             alpha=None):
    if k is not None:
        # other-layer 원본 복원
        for i, w in original_weights.items():
            if i != target_idx:
                model.classifier.layers[i].weight.data = w.to(device)
        # 타겟 레이어 재구성
        recon_w = regularized_svd_recon(target_weight_cpu, k, alpha).to(device)
        model.classifier.layers[target_idx].weight.data = recon_w

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total

# -----------------------------------------------------------------
# 6. 실험 대상 모델 설정 (CNN_0, CNN_1만)
# -----------------------------------------------------------------
experiment_models = {
    "0_CNN_Layers": {'class': CNN_0, 'target_idx': 2},
    "1_CNN_Layer":  {'class': CNN_1, 'target_idx': 2},
}

# -----------------------------------------------------------------
# 7. 메인 루프: 학습 → full-acc → rank별 raw/norm acc 저장
# -----------------------------------------------------------------
for model_name, cfg in experiment_models.items():
    print(f"\n=== Running model: {model_name} ===")
    torch.manual_seed(seed)
    model = cfg['class']().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- (1) 학습 ---
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(trainloader, desc=f"Train Ep {ep}/{epochs} [{model_name}]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    # --- (2) full-rank 정확도 ---
    full_acc = accuracy(model, testloader, device)
    print(f"Full-rank Test Accuracy ({model_name}): {full_acc:.4f}")

    # --- (3) rank-k 재구성 평가 ---
    target_idx = cfg['target_idx']
    original_weights = {
        i: layer.weight.data.clone()
        for i, layer in enumerate(model.classifier.layers)
        if isinstance(layer, nn.Linear)
    }
    target_weight_cpu = original_weights[target_idx].cpu()
    max_k = min(target_weight_cpu.shape)

    raw_accs = []
    for k in tqdm(range(1, max_k+1), desc=f"Recon [{model_name}]"):
        acc_k = accuracy(model, testloader, device,
                         k=k,
                         original_weights=original_weights,
                         target_idx=target_idx,
                         target_weight_cpu=target_weight_cpu,
                         alpha=regularization_alpha)
        raw_accs.append(acc_k)

    raw_accs = np.array(raw_accs)
    norm_accs = raw_accs / full_acc if full_acc > 0 else np.zeros_like(raw_accs)

    # --- (4) CSV 저장 ---
    np.savetxt(os.path.join(results_dir, f"{model_name}_raw_acc.csv"),  raw_accs,  delimiter=',')
    np.savetxt(os.path.join(results_dir, f"{model_name}_norm_acc.csv"), norm_accs, delimiter=',')

print("\nAll analyses complete.")

# -----------------------------------------------------------------
# 8. 최종 그래프: Raw vs Normalized Accuracy
# -----------------------------------------------------------------
print("Generating final comparison plot...")
plt.figure(figsize=(14,8))

for model_name in experiment_models.keys():
    raw_path  = os.path.join(results_dir, f"{model_name}_raw_acc.csv")
    norm_path = os.path.join(results_dir, f"{model_name}_norm_acc.csv")

    if os.path.exists(raw_path):
        y_raw = np.loadtxt(raw_path, delimiter=',')
        plt.plot(range(1, len(y_raw)+1), y_raw,
                 label=f"{model_name} (Raw)", linestyle='--', alpha=0.8)

    if os.path.exists(norm_path):
        y_norm = np.loadtxt(norm_path, delimiter=',')
        plt.plot(range(1, len(y_norm)+1), y_norm,
                 label=f"{model_name} (Normalized)", linestyle='-', alpha=0.8)

plt.title(f"Rank vs Accuracy (Regularized SVD, α={regularization_alpha})", fontsize=16)
plt.xlabel("Rank (k)", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"comparison_raw_and_norm_alpha_{regularization_alpha}.png"))
plt.close()
print(f"Plot saved in '{results_dir}'.")
