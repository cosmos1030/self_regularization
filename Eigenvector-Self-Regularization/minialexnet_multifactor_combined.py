import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import shutil

# ------------------------------
# 1. 데이터 및 모델 준비
# ------------------------------

# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

batch_size = 64

# CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define miniAlexNet
class miniAlexNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 256, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        # classifier 구성: fc1, fc2, fc3  
        # 인덱스: fc1 -> 0, fc2 -> 2, fc3 -> 4 (ReLU층이 끼어 있음)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 384),   # fc1
            nn.ReLU(),
            nn.Linear(384, 192),           # fc2
            nn.ReLU(),
            nn.Linear(192, num_classes)    # fc3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Model initialization
model = miniAlexNet()

# Set seed for reproducibility
seed = 100
torch.manual_seed(seed)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

epochs = 100
accuracy = []

# 실험 결과 및 파일 저장 디렉토리
base_dir = os.path.join("runs", f"alex_seed{seed}_batch{batch_size}_sgd_lr{learning_rate}_epochs{epochs}_recon_expt")
if os.path.exists(base_dir):
    print(f"Directory '{base_dir}' already exists. Removing it...")
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

# 결과 저장용 하위 디렉토리들
eigenvector_dir = os.path.join(base_dir, 'eigenvectors')
bias_dir = os.path.join(base_dir, 'biases')
spectrum_dir = os.path.join(base_dir, 'spectrum')
os.makedirs(eigenvector_dir, exist_ok=True)
os.makedirs(bias_dir, exist_ok=True)
os.makedirs(spectrum_dir, exist_ok=True)

# ------------------------------
# 2. 장치 설정 및 테스트 함수
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizing {str(device)}")
model.to(device)

def test_model_accuracy(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ------------------------------
# 3. 재구성(experiment) 헬퍼 함수들
# ------------------------------

# fc layer의 인덱스 매핑 (classifier 내에서)
fc_indices = {
    "fc1": 0,
    "fc2": 2,
    "fc3": 4
}

def apply_rank_k_to_layers(model, saved_weights, target_indices, k):
    """
    model.classifier 내 target_indices에 해당하는 fc layer들에 대해
    저장된 가중치(saved_weights)에서 k singular components만 사용하여 재구성합니다.
    """
    for idx in target_indices:
        W = saved_weights[idx].numpy()  # (out_features, in_features)
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        r = min(k, len(S))
        W_recon = np.zeros_like(W)
        for i in range(r):
            W_recon += S[i] * np.outer(U[:, i], Vh[i, :])
        model.classifier[idx].weight.data = torch.tensor(W_recon, dtype=torch.float32).to(model.classifier[idx].weight.device)

def restore_weights(model, saved_weights, target_indices):
    """저장된 saved_weights를 이용해 대상 fc layer들을 복원합니다."""
    for idx in target_indices:
        model.classifier[idx].weight.data = saved_weights[idx].to(model.classifier[idx].weight.device)

# ------------------------------
# 4. 재구성 실험 설정
# ------------------------------

# 실험할 시점 에폭들 (학습 도중에 재구성 실험 진행)
recon_epochs = [25, 50, 75, 100]
# 실험 상황: 각 상황마다 target layer만 재구성하고 나머지는 고정
# 'all': fc1, fc2, fc3 모두 재구성
# 'fc1': fc1만 재구성
# 'fc2': fc2만 재구성
# 'fc3': fc3만 재구성
scenarios = {
    "all": [fc_indices["fc1"], fc_indices["fc2"], fc_indices["fc3"]],
    "fc1": [fc_indices["fc1"]],
    "fc2": [fc_indices["fc2"]],
    "fc3": [fc_indices["fc3"]]
}
max_rank = 500  # k의 최대값

# ------------------------------
# 5. Training Loop 및 재구성 실험
# ------------------------------

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0.0
    total = 0.0

    train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True)
    for i, data in enumerate(train_pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        train_pbar.set_postfix(loss=f"{avg_loss:.4f}")

    # --- Save some eigenvector and bias info (옵션) ---
    model.eval()
    with torch.no_grad():
        layer_index = 1
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight.detach().cpu().numpy().reshape(layer.out_channels, -1)
                U, S, Vh = np.linalg.svd(weight)
                leading_eigenvector = Vh[0, :]
                with open(os.path.join(eigenvector_dir, f'conv{layer_index}.csv'), 'a') as f:
                    np.savetxt(f, leading_eigenvector.reshape(1, -1))
                layer_index += 1

        layer_index = 1
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().numpy()
                bias = layer.bias.detach().cpu().numpy()
                U, S, Vh = np.linalg.svd(weight)
                leading_eigenvector = Vh[0, :]
                with open(os.path.join(eigenvector_dir, f'fc{layer_index}.csv'), 'a') as f:
                    np.savetxt(f, leading_eigenvector.reshape(1, -1))
                with open(os.path.join(bias_dir, f'fc{layer_index}_bias.csv'), 'a') as f:
                    np.savetxt(f, bias.reshape(1, -1))
                layer_index += 1

        test_pbar = tqdm(testloader, desc=f"Epoch {epoch+1}/{epochs} [Testing]")
        for testdata in test_pbar:
            images, testlabels = testdata
            images = images.to(device)
            testlabels = F.one_hot(testlabels, num_classes=10).float().to(device)
            predictions = model(images)
            _, predicted_labels = torch.max(predictions.data, 1)
            total += testlabels.size(0)
            correct += (predicted_labels == testlabels.argmax(dim=1)).sum().item()

    acc = correct / total
    accuracy.append(acc)
    print(f'Epoch {epoch+1} finished, test accuracy: {acc:.4f}')
    if total_loss < 0.0001:
        break

    # 재구성 실험 진행할 에폭인 경우
    if (epoch+1) in recon_epochs:
        print(f"→ Performing reconstruction experiments at epoch {epoch+1}")
        # 각 시나리오 별 실험 수행
        for scenario_name, target_indices in scenarios.items():
            print(f"   - Scenario: {scenario_name}")
            # 각 target layer에 대해 원본 가중치 복사 (딕셔너리: key=fc layer index, value = 가중치)
            saved_weights = {}
            for idx in target_indices:
                saved_weights[idx] = model.classifier[idx].weight.data.clone().cpu()
            
            recon_acc_list = []  # raw reconstruction accuracy, k=1 ~ max_rank
            # k를 변화시키며 실험
            for k in tqdm(range(1, max_rank + 1), desc=f"Reconstruction ({scenario_name})", leave=False):
                apply_rank_k_to_layers(model, saved_weights, target_indices, k)
                acc_recon = test_model_accuracy(model, testloader, device)
                recon_acc_list.append(acc_recon)
                restore_weights(model, saved_weights, target_indices)
            recon_acc_list = np.array(recon_acc_list)
            # 정규화: 최대 k (max_rank)일 때의 정확도를 1로 설정
            baseline = recon_acc_list[-1]
            if baseline > 0:
                norm_recon_acc = recon_acc_list / baseline
            else:
                norm_recon_acc = recon_acc_list

            # CSV 파일 저장 (파일명에 시나리오와 에폭 포함)
            np.savetxt(os.path.join(base_dir, f"recon_{scenario_name}_acc_epoch{epoch+1}.csv"), recon_acc_list)
            np.savetxt(os.path.join(base_dir, f"recon_{scenario_name}_acc_norm_epoch{epoch+1}.csv"), norm_recon_acc)

# 학습 정확도 곡선 저장
acc_array = np.array(accuracy)
np.savetxt(os.path.join(base_dir, "accuracy.csv"), acc_array)
fig, ax = plt.subplots()
ax.plot(acc_array)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Training Accuracy Across Epochs')
fig.savefig(os.path.join(base_dir, 'accuracy.png'))
plt.close(fig)

print('Finished training')

# ------------------------------
# 6. 그래프 그리기: 각 시나리오별로 overlay (정규화/원시)
# ------------------------------

for scenario_name in scenarios.keys():
    # 정규화된 그래프 overlay (여러 recon epoch curve를 하나에 overlay)
    fig, ax = plt.subplots(figsize=(10, 6))
    for epoch_val in recon_epochs:
        csv_file = os.path.join(base_dir, f"recon_{scenario_name}_acc_norm_epoch{epoch_val}.csv")
        if os.path.exists(csv_file):
            norm_acc = np.loadtxt(csv_file)
            ax.plot(range(1, len(norm_acc) + 1), norm_acc, label=f"Epoch {epoch_val}")
    ax.set_xlabel("Number of Singular Components (Rank k)")
    ax.set_ylabel("Normalized Test Accuracy")
    ax.set_title(f"Normalized Reconstruction Accuracy vs Rank ({scenario_name})")
    ax.legend()
    ax.grid(True)
    norm_fig_path = os.path.join(base_dir, f"recon_{scenario_name}_acc_norm_overlay.png")
    plt.savefig(norm_fig_path)
    plt.close(fig)
    print(f"Saved normalized overlay graph for scenario '{scenario_name}' at: {norm_fig_path}")

    # 원시(unnormalized) 그래프 overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for epoch_val in recon_epochs:
        csv_file = os.path.join(base_dir, f"recon_{scenario_name}_acc_epoch{epoch_val}.csv")
        if os.path.exists(csv_file):
            raw_acc = np.loadtxt(csv_file)
            ax.plot(range(1, len(raw_acc) + 1), raw_acc, label=f"Epoch {epoch_val}")
    ax.set_xlabel("Number of Singular Components (Rank k)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Reconstruction Accuracy vs Rank ({scenario_name})")
    ax.legend()
    ax.grid(True)
    raw_fig_path = os.path.join(base_dir, f"recon_{scenario_name}_acc_overlay.png")
    plt.savefig(raw_fig_path)
    plt.close(fig)
    print(f"Saved raw overlay graph for scenario '{scenario_name}' at: {raw_fig_path}")
