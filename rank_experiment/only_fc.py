#!/usr/bin/env python
# =================================================================
#   실험 2 수정: 전체가 FC 레이어로 구성된 모델의 깊이에 따른 랭크 재구성 성능 비교
# =================================================================
# 0. 라이브러리 및 기본 설정
# -----------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm

# --- GPU 설정 ---
# 사용 가능한 GPU가 있는 경우 자동으로 'cuda'를 사용합니다.
# 특정 GPU를 지정하려면 os.environ['CUDA_VISIBLE_DEVICES'] = '0' 과 같이 설정하세요.

# -----------------------------------------------------------------
# 1. 하이퍼파라미터 및 경로 설정
# -----------------------------------------------------------------
# 학습 하이퍼파라미터
batch_size    = 128
seed          = 100
learning_rate = 0.002
epochs        = 50
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer_type= 'adam'
recon_epoch   = epochs

# --- 결과 저장 기본 디렉토리 ---
results_dir = "experiment_2_mlp_depth"
if os.path.exists(results_dir): shutil.rmtree(results_dir)
os.makedirs(results_dir)

# -----------------------------------------------------------------
# 2. CIFAR-10 데이터 로딩
# -----------------------------------------------------------------
print("==> Preparing CIFAR-10 data...")
tfm = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
trainset = torchvision.datasets.CIFAR10("./data", train=True, transform=tfm, download=True)
testset  = torchvision.datasets.CIFAR10("./data", train=False, transform=tfm, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
print(f"==> Data preparation complete. Using device: {device}")

# -----------------------------------------------------------------
# 3. 모델 정의 (전체 FC 레이어, CNN 없음)
# -----------------------------------------------------------------
# CIFAR-10 이미지(3x32x32)를 flatten하면 3072개의 입력 피처가 됩니다.
input_features = 3 * 32 * 32

class MLP_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), # 타겟 레이어
            nn.Linear(192, 10)
        )
    def forward(self, x): return self.classifier(x)

class MLP_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 1024), nn.ReLU(),
            nn.Linear(1024, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), # 타겟 레이어
            nn.Linear(192, 10)
        )
    def forward(self, x): return self.classifier(x)

class MLP_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), # 타겟 레이어
            nn.Linear(192, 10)
        )
    def forward(self, x): return self.classifier(x)

# -----------------------------------------------------------------
# 4. 헬퍼 함수 (정확도, SVD 재구성)
# -----------------------------------------------------------------
def accuracy(model, loader, device, k=None, original_weights=None, target_idx=None, target_weight_cpu=None):
    if k is not None: # 분석 모드
        # 모든 Linear 레이어의 가중치를 원본으로 복원
        for i, w in original_weights.items():
            model.classifier[i].weight.data = w.to(device)
        # 타겟 레이어의 가중치만 SVD 재구성된 가중치로 교체
        recon_w = topk_recon(target_weight_cpu, k).to(device)
        model.classifier[target_idx].weight.data = recon_w

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def topk_recon(W_cpu, k):
    W = W_cpu.to(torch.float32)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # k개의 특이값만 사용하여 행렬 재구성
    return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]

# -----------------------------------------------------------------
# 5. 실험 설정
# -----------------------------------------------------------------
experiment_models = {
    "3-Layer MLP": {'class': MLP_3, 'target_idx': 3}, # Flatten(0), Linear(1), ReLU(2), Linear(3)
    "4-Layer MLP": {'class': MLP_4, 'target_idx': 5}, # ... Linear(5)
    "5-Layer MLP": {'class': MLP_5, 'target_idx': 7}, # ... Linear(7)
}

# -----------------------------------------------------------------
# 6. 메인 실험 실행 루프
# -----------------------------------------------------------------
for model_name, config in experiment_models.items():
    print(f"\n{'='*60}\n--- Running for model: {model_name} ---\n{'='*60}")
    torch.manual_seed(seed)
    model = config['class']().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == 'adam' else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # 학습 루프
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(trainloader, desc=f"Train Ep {ep}/{epochs} [{model_name}]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    # 최종 정확도 측정
    final_acc = accuracy(model, testloader, device)
    print(f"Final Test Accuracy for {model_name}: {final_acc:.4f}")

    # 랭크 재구성 분석
    if recon_epoch > 0:
        print(f"--- Running Rank Reconstruction for {model_name} ---")
        target_layer_idx = config['target_idx']
        
        # Linear 레이어의 가중치만 복사하여 저장
        original_weights = {i: layer.weight.data.clone() for i, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear)}
        
        # 타겟 레이어의 가중치를 CPU로 가져옴
        target_weight_saved = model.classifier[target_layer_idx].weight.data.cpu()
        
        max_k = min(target_weight_saved.shape)
        
        # k를 1부터 max_k까지 변화시키며 재구성 후 정확도 측정
        raw_accs = [accuracy(model, testloader, device, k, original_weights, target_layer_idx, target_weight_saved) 
                    for k in tqdm(range(1, max_k + 1), desc=f"Recon [{model_name}]")]
        
        raw_accs = np.array(raw_accs)
        # 원본 정확도(k=max_k일 때)로 정규화
        norm_accs = raw_accs / raw_accs[-1] if raw_accs[-1] > 0 else np.zeros_like(raw_accs)
        
        sanitized_model_name = model_name.replace(" ", "_")
        os.makedirs(results_dir, exist_ok=True)
        np.savetxt(os.path.join(results_dir, f"{sanitized_model_name}_norm_acc.csv"), norm_accs, delimiter=',')

print("\nAll analyses for Experiment 2 (MLP version) are complete.")

# -----------------------------------------------------------------
# 7. 최종 비교 그래프 생성
# -----------------------------------------------------------------
print("\nGenerating final comparison plot...")
plt.figure(figsize=(12, 8))
for model_name in experiment_models.keys():
    sanitized_model_name = model_name.replace(" ", "_")
    file_path = os.path.join(results_dir, f"{sanitized_model_name}_norm_acc.csv")
    if os.path.exists(file_path):
        y = np.loadtxt(file_path, delimiter=',')
        plt.plot(range(1, len(y) + 1), y, label=model_name, marker='.', markersize=4, alpha=0.8)

plt.title("MLP Depth vs. Normalized Accuracy after Rank Reconstruction", fontsize=16)
plt.xlabel("Rank (k)", fontsize=12)
plt.ylabel("Normalized Test Accuracy", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(results_dir, f"__comparison_normalized_accuracy_mlp_depth.png"))
plt.close()

print(f"Plot saved in '{results_dir}' directory.")