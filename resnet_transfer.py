import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

# ------------------------------
# 1. 환경 설정 및 디렉토리 준비
# ------------------------------
print("1. 설정 및 디렉토리 준비 시작")

# GPU/CPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 결과 저장을 위한 디렉토리 설정
base_dir = "runs/resnet18_cifar10_spectrum_analysis"
if os.path.exists(base_dir):
    print(f"기존 디렉토리 '{base_dir}'를 삭제합니다.")
    shutil.rmtree(base_dir)
spectrum_dir = os.path.join(base_dir, 'spectrum')
os.makedirs(spectrum_dir, exist_ok=True)
print(f"결과는 '{base_dir}'에 저장됩니다.")

# ------------------------------
# 2. 데이터 준비
# ------------------------------
print("2. 데이터 준비 시작")
# CIFAR-10 데이터셋을 위한 전처리
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# CIFAR-10 데이터셋 다운로드 및 로더 생성
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
print("2. 데이터 준비 완료")

# ------------------------------
# 3. 모델 준비
# ------------------------------
print("\n3. 모델 준비 시작")
# ImageNet으로 사전 훈련된 ResNet18 모델 로드
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 컨볼루션 레이어들의 가중치 동결
for param in model.parameters():
    param.requires_grad = False

# 마지막 FC 레이어를 3개의 새로운 FC 레이어로 교체
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 10)
)

model = model.to(device)
print("3. 모델 준비 완료\n")
print(model.fc)

# ------------------------------
# 4. 모델 훈련 및 스펙트럼 분석
# ------------------------------
print("\n4. 모델 훈련 및 스펙트럼 분석 시작")
# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
epochs = 5 # 에포크 수 (필요 시 조정)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
    for i, data in enumerate(train_pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}'})

    # --- 에포크 종료 후 평가 및 스펙트럼 분석 ---
    model.eval()
    with torch.no_grad():
        # FC 레이어 스펙트럼 분석 및 저장
        fc_layer_index = 1
        for layer in model.fc:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().numpy()
                
                # SVD 및 스펙트럼 계산
                U, S, Vh = np.linalg.svd(weight, full_matrices=False)
                n, p = weight.shape # n: 출력 뉴런 수, p: 입력 뉴런 수
                spec = np.square(S) / p # 논문에 따라 p로 정규화 (또는 n)
                
                # Marchenko-Pastur(MP) 상한 계산
                gamma = n / p
                sigma2 = np.var(weight) # 가중치 행렬 원소들의 분산
                lambda_plus = sigma2 * (1 + np.sqrt(gamma))**2
                
                # MP 상한을 넘는 특이값(스파이크) 개수 계산
                spike_count = np.sum(spec > lambda_plus)
                
                # 스펙트럼 시각화 및 저장
                fig_spec, ax_spec = plt.subplots(figsize=(10, 6))
                ax_spec.hist(spec, bins=200, density=True, alpha=0.7, label='Empirical Spectrum')
                ax_spec.axvline(x=lambda_plus, color='r', linestyle='--', lw=2, label=f'MP Upper Bound ($\lambda_+$)')
                ax_spec.set_title(f"FC{fc_layer_index} Spectrum at Epoch {epoch+1}", fontsize=16)
                ax_spec.set_xlabel("Normalized Squared Singular Value ($\lambda$)", fontsize=12)
                ax_spec.set_ylabel("Density", fontsize=12)
                ax_spec.text(0.65, 0.95, f"Spike Count: {spike_count}", transform=ax_spec.transAxes,
                            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax_spec.legend()
                ax_spec.grid(True, linestyle='--', alpha=0.6)
                
                spec_path = os.path.join(spectrum_dir, f'fc{fc_layer_index}_spectrum_epoch{epoch+1}.png')
                fig_spec.savefig(spec_path)
                plt.close(fig_spec)
                
                fc_layer_index += 1
        
        # 테스트 정확도 계산
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} 완료. Test Accuracy: {acc:.2f}%")

print("4. 모델 훈련 및 스펙트럼 분석 완료")
model_path = os.path.join(base_dir, "resnet18_finetuned.pth")
torch.save(model.state_dict(), model_path)
print(f"훈련된 모델을 '{model_path}'에 저장했습니다.")

# ------------------------------
# 5. SVD 재구성 및 평가 (마지막에서 두 번째 FC 레이어)
# ------------------------------
print("\n5. SVD 재구성 및 평가 시작")

# 평가 함수
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 분석 대상 레이어: 마지막에서 두 번째 FC 레이어 (model.fc의 2번 인덱스)
target_layer = model.fc[2]
original_weights = target_layer.weight.data.clone()
W = original_weights.cpu().numpy()

# SVD 수행
print("마지막에서 두 번째 FC 레이어 가중치에 대해 SVD를 수행합니다...")
U, S, Vh = np.linalg.svd(W, full_matrices=False)
print(f"SVD 완료. 가중치 행렬 크기: {W.shape}, 특이값 개수: {len(S)}")

# 랭크(k) 개수별로 재구성 및 정확도 측정
accuracies = []
max_rank = len(S)
ranks = list(range(1, max_rank + 1))

pbar_svd = tqdm(ranks, desc="SVD 재구성 및 평가 중")
for k in pbar_svd:
    S_k = np.diag(S[:k])
    W_recon = np.dot(U[:, :k], np.dot(S_k, Vh[:k, :]))
    W_recon_tensor = torch.from_numpy(W_recon).float().to(device)
    target_layer.weight.data = W_recon_tensor

    acc = evaluate_model(model, testloader, device)
    accuracies.append(acc)
    pbar_svd.set_postfix({'Rank': k, 'Accuracy': f'{acc:.2f}%'})

target_layer.weight.data = original_weights.to(device)
print("5. SVD 재구성 및 평가 완료")


# ------------------------------
# 6. 결과 시각화
# ------------------------------
print("\n6. 재구성 결과 시각화")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(ranks, accuracies, marker='o', linestyle='-', color='b', markersize=4)
ax.set_title('Model Accuracy vs. Number of Singular Values for Reconstruction (FC2)', fontsize=16)
ax.set_xlabel('Number of Singular Values (Rank k)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)

full_rank_accuracy = evaluate_model(model, testloader, device)
ax.axhline(y=full_rank_accuracy, color='r', linestyle='--', label=f'Full Rank Accuracy ({full_rank_accuracy:.2f}%)')
ax.legend()
ax.grid(True)
plt.tight_layout()

recon_fig_path = os.path.join(base_dir, "svd_reconstruction_accuracy.png")
plt.savefig(recon_fig_path)
print(f"재구성 결과 그래프를 '{recon_fig_path}' 파일로 저장했습니다.")
plt.show()