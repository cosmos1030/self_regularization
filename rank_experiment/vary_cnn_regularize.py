#!/usr/bin/env python
# =================================================================
#   실험 2: 정규화된 SVD 기반 랭크 재구성에 따른 성능 비교
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
# 사용 가능한 GPU가 여러 개인 경우, 사용할 GPU 번호를 지정하세요.
# 예: os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'


# -----------------------------------------------------------------
# 1. 하이퍼파라미터 및 경로 설정
# -----------------------------------------------------------------
# 학습 하이퍼파라미터
batch_size    = 64
seed          = 100
learning_rate = 0.001
epochs        = 20
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer_type= 'adam'
recon_epoch   = epochs

# --- 실험 2번을 위한 하이퍼파라미터 ---
# 공분산 행렬 정규화 강도 (alpha)
regularization_alpha = 0.01

# --- 결과 저장 기본 디렉토리 ---
results_dir = "experiment_regularized_svd"
if os.path.exists(results_dir): shutil.rmtree(results_dir)
os.makedirs(results_dir)

# -----------------------------------------------------------------
# 2. CIFAR-10 데이터 로딩 (기존과 동일)
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
print("==> Data preparation complete.")


# -----------------------------------------------------------------
# 3. 모델 정의 (기존과 동일)
# -----------------------------------------------------------------

# --- 고정된 3-레이어 FC 분류기 헤드 ---
class ClassifierHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), # 타겟 레이어
            nn.Linear(192, 10)
        )
    def forward(self, x):
        return self.layers(x)

# --- CNN 깊이가 다른 모델들 ---
class CNN_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = ClassifierHead(in_features=3 * 32 * 32)
    def forward(self, x):
        return self.classifier(torch.flatten(x, 1))

class CNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = ClassifierHead(in_features=96 * 16 * 16)
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

class CNN_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = ClassifierHead(in_features=256 * 8 * 8)
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

class CNN_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = ClassifierHead(in_features=384 * 4 * 4)
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


# -----------------------------------------------------------------
# 4. 헬퍼 함수 (정규화된 SVD 재구성 함수 추가)
# -----------------------------------------------------------------

def regularized_svd_recon(W_cpu, k, alpha):
    """
    실험 2: 정규화된 공분산 행렬의 고유벡터를 이용한 재구성
    """
    W = W_cpu.to(torch.float32)
    p, n = W.shape

    # 1. 원본 가중치 행렬 W에서 특이값(S)을 미리 계산합니다.
    _, S, _ = torch.linalg.svd(W, full_matrices=False)

    # 2. 공분산 행렬 WW^T 와 W^T W 를 계산합니다.
    C_u = W @ W.T   # p x p
    C_v = W.T @ W   # n x n

    # 3. 대각 행렬(alpha * I)을 더해 공분산 행렬을 정규화합니다.
    #    torch.eye는 단위 행렬(Identity matrix)을 생성합니다.
    C_u_reg = C_u + alpha * torch.eye(p, device=W.device)
    C_v_reg = C_v + alpha * torch.eye(n, device=W.device)

    # 4. 정규화된 공분산 행렬의 고유벡터를 계산합니다. (eigh는 대칭행렬용)
    #    eigh는 고유값을 오름차순으로 반환하므로, SVD 관례에 맞게 뒤집어줍니다.
    _, U_reg = torch.linalg.eigh(C_u_reg)
    U_reg = torch.flip(U_reg, dims=[-1])

    _, V_reg = torch.linalg.eigh(C_v_reg)
    V_reg = torch.flip(V_reg, dims=[-1])

    # 5. 원본 특이값과 정규화된 고유벡터를 사용해 행렬을 재구성합니다.
    Vh_reg = V_reg.T
    recon_W = U_reg[:, :k] @ torch.diag(S[:k]) @ Vh_reg[:k, :]
    
    return recon_W


def accuracy(model, loader, device, k=None, original_weights=None, target_idx=None, target_weight_cpu=None, alpha=None):
    if k is not None: # 분석 모드
        # 원본 가중치로 복원 (타겟 레이어 제외)
        for i, w in original_weights.items():
            if i != target_idx:
                model.classifier.layers[i].weight.data = w.to(device)
        
        # <<-- 핵심 변경점: 정규화된 SVD 재구성 함수 호출 -->>
        recon_w = regularized_svd_recon(target_weight_cpu, k, alpha).to(device)
        model.classifier.layers[target_idx].weight.data = recon_w

    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total

# -----------------------------------------------------------------
# 5. 실험 설정 (기존과 동일)
# -----------------------------------------------------------------
experiment_models = {
    "0 CNN Layers": {'class': CNN_0, 'target_idx': 2},
    "1 CNN Layer":  {'class': CNN_1, 'target_idx': 2},
    "2 CNN Layers": {'class': CNN_2, 'target_idx': 2},
    "3 CNN Layers": {'class': CNN_3, 'target_idx': 2},
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

    # 모델 학습
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(trainloader, desc=f"Train Ep {ep}/{epochs} [{model_name}]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device); optimizer.zero_grad()
            loss = criterion(model(x), y); loss.backward(); optimizer.step()
            pbar.set_postfix(loss=loss.item())

    print(f"Final Test Accuracy for {model_name}: {accuracy(model, testloader, device):.4f}")

    # 재구성 실험
    if recon_epoch > 0:
        print(f"--- Running Regularized Rank Reconstruction for {model_name} (alpha={regularization_alpha}) ---")
        target_layer_idx = config['target_idx']
        original_weights = {i: layer.weight.data.clone() for i, layer in enumerate(model.classifier.layers) if isinstance(layer, nn.Linear)}
        target_weight_saved = original_weights[target_layer_idx].cpu()
        max_k = min(target_weight_saved.shape)
        
        # accuracy 함수에 alpha 값을 전달
        raw_accs = [accuracy(model, testloader, device, k, original_weights, target_layer_idx, target_weight_saved, alpha=regularization_alpha) for k in tqdm(range(1, max_k + 1), desc=f"Recon [{model_name}]")]

        raw_accs, norm_accs = np.array(raw_accs), np.array(raw_accs) / raw_accs[-1] if raw_accs[-1] > 0 else np.zeros_like(raw_accs)
        sanitized_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        np.savetxt(os.path.join(results_dir, f"{sanitized_model_name}_norm_acc_regularized.csv"), norm_accs, delimiter=',')

print("\nAll analyses for the experiment are complete.")

# -----------------------------------------------------------------
# 7. 최종 비교 그래프 생성
# -----------------------------------------------------------------
print("\nGenerating final comparison plot...")
plt.figure(figsize=(12, 8))
for model_name in experiment_models.keys():
    sanitized_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    file_path = os.path.join(results_dir, f"{sanitized_model_name}_norm_acc_regularized.csv")
    if os.path.exists(file_path):
        y = np.loadtxt(file_path, delimiter=',')
        plt.plot(range(1, len(y) + 1), y, label=model_name, marker='.', markersize=4, alpha=0.8)

plt.title(f"Experiment 2: Rank vs. Normalized Accuracy (Regularized SVD, alpha={regularization_alpha})", fontsize=16)
plt.xlabel("Rank (k)", fontsize=12); plt.ylabel("Normalized Test Accuracy", fontsize=12)
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(results_dir, f"__comparison_regularized_svd_alpha_{regularization_alpha}.png"))
plt.close()

print(f"Plot saved in '{results_dir}' directory.")