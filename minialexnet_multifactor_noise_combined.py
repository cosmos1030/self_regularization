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
# 1. Data Preparation + Label Noise
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
batch_size = 64

# -- (1) Load CIFAR-10 --
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# -- (2) Apply Noisy Labels (Load or Generate) --
noise_ratio = 0.3  # 노이즈로 바꿀 라벨 비율(예: 0.3 -> 30%)
noise_seed = 5
save_path = f"./data/noisy_labels{int(noise_ratio * 100)}_seed{noise_seed}.npy"

if os.path.exists(save_path):
    print(f"Loading noisy labels from {save_path}")
    noisy_targets = np.load(save_path)
    trainset.targets = noisy_targets.tolist()
else:
    print(f"No noisy label file found. Generating new noisy labels... (save to {save_path})")
    np.random.seed(noise_seed)
    targets = np.array(trainset.targets)
    noisy_targets = targets.copy()
    n_noisy = int(noise_ratio * len(targets))
    noisy_indices = np.random.choice(len(targets), n_noisy, replace=False)
    num_classes = 10

    for idx in noisy_indices:
        noisy_targets[idx] = np.random.randint(num_classes)
    # Save .npy so that we can reuse next time
    np.save(save_path, noisy_targets)
    trainset.targets = noisy_targets.tolist()

# -- (3) Data Loaders --
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ------------------------------
# 2. Model Definition
# ------------------------------
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
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 384),  # fc1
            nn.ReLU(),
            nn.Linear(384, 192),         # fc2
            nn.ReLU(),
            nn.Linear(192, num_classes)  # fc3
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = miniAlexNet()

# ------------------------------
# 3. Training Setup
# ------------------------------
seed = 100
torch.manual_seed(seed)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
epochs = 100
accuracy = []

# base_dir = os.path.join("runs", f"alex_seed{seed}_batch{batch_size}_sgd_lr{learning_rate}_epochs{epochs}_recon_expt")'
base_dir = os.path.join("runs", f"alex_seed{seed}_batch{batch_size}_sgd_lr{learning_rate}_epochs{epochs}_noise{noise_ratio*100}_noiseseed{noise_seed}_recon_expt")
if os.path.exists(base_dir):
    print(f"Directory '{base_dir}' already exists. Removing it...")
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

eigenvector_dir = os.path.join(base_dir, 'eigenvectors')
bias_dir = os.path.join(base_dir, 'biases')
spectrum_dir = os.path.join(base_dir, 'spectrum')
os.makedirs(eigenvector_dir, exist_ok=True)
os.makedirs(bias_dir, exist_ok=True)
os.makedirs(spectrum_dir, exist_ok=True)

device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
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
# 4. Reconstruction Helper Functions
# ------------------------------
fc_indices = {
    "fc1": 0,
    "fc2": 2,
    "fc3": 4
}
all_fc_indices = [fc_indices["fc1"], fc_indices["fc2"], fc_indices["fc3"]]

def apply_reconstruction_gpu(weight_cpu: torch.Tensor, k: int, device):
    """SVD를 이용하여 상위 k개 특이값만을 사용해 weight를 재구성"""
    weight = weight_cpu.to(device)
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    r = min(k, S.shape[0])
    W_recon = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
    return W_recon

# ------------------------------
# 5. Compute max_rank of each FC layer for reconstruction
# ------------------------------
max_rank_dict = {}
for name, idx in fc_indices.items():
    weight_shape = model.classifier[idx].weight.shape
    max_rank_dict[name] = min(weight_shape[0], weight_shape[1])
    print(f"{name} maximum rank: {max_rank_dict[name]}")

# 재구성 실험을 진행할 Epoch
recon_epochs = [25, 50, 75, 100]

# ------------------------------
# 6. Training Loop + Reconstruction Experiment
# ------------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

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

    # Epoch 단위 결과 저장(특잇값, 이젠벡터 등)
    model.eval()
    with torch.no_grad():
        # Conv layer
        conv_layer_index = 1
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight.detach().cpu().numpy().reshape(layer.out_channels, -1)
                U, S, Vh = np.linalg.svd(weight)
                leading_eigenvector = Vh[0, :]
                with open(os.path.join(eigenvector_dir, f'conv{conv_layer_index}.csv'), 'a') as f:
                    np.savetxt(f, leading_eigenvector.reshape(1, -1))
                conv_layer_index += 1

        # FC layer: eigenvector, bias, spectrum
        fc_layer_index = 1
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().numpy()
                bias = layer.bias.detach().cpu().numpy()
                U, S, Vh = np.linalg.svd(weight, full_matrices=False)
                leading_eigenvector = Vh[0, :]
                with open(os.path.join(eigenvector_dir, f'fc{fc_layer_index}.csv'), 'a') as f:
                    np.savetxt(f, leading_eigenvector.reshape(1, -1))
                with open(os.path.join(bias_dir, f'fc{fc_layer_index}_bias.csv'), 'a') as f:
                    np.savetxt(f, bias.reshape(1, -1))

                # MP theory bound, spike count 측정
                n, p = weight.shape
                spec = np.square(S) / n
                gamma = p / n
                sigma2 = np.var(weight)
                lambda_plus = sigma2 * (1 + np.sqrt(gamma))**2
                spike_count = np.sum(spec > lambda_plus)

                fig_spec, ax_spec = plt.subplots(figsize=(8, 5))
                ax_spec.hist(spec, bins=300, density=True, alpha=0.5, label='Empirical Spectrum')
                ax_spec.axvline(x=lambda_plus, color='red', linestyle='-', lw=2, label='MP Upper Bound')
                ax_spec.set_title(f"FC{fc_layer_index} Spectrum at Epoch {epoch+1}")
                ax_spec.set_xlabel("Normalized Squared Singular Value")
                ax_spec.set_ylabel("Density")
                ax_spec.text(0.05, 0.95, f"Spike Count: {spike_count}", transform=ax_spec.transAxes,
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax_spec.legend()
                spec_path = os.path.join(spectrum_dir, f'fc{fc_layer_index}_spectrum_epoch{epoch+1}.png')
                fig_spec.savefig(spec_path)
                plt.close(fig_spec)

                fc_layer_index += 1

        # Test accuracy
        correct = 0
        total = 0
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

    # ------------------------------
    # Reconstruction Experiment (특정 epoch일 때)
    # ------------------------------
    if (epoch+1) in recon_epochs:
        print(f"Performing reconstruction experiments at epoch {epoch+1}")
        saved_weights_all = {
            idx: model.classifier[idx].weight.data.clone().cpu() 
            for idx in all_fc_indices
        }
        scenario_results = {}

        # 개별 fc layer만 수정 (fc1, fc2, fc3)
        for scenario_name, fc_idx in fc_indices.items():
            max_rank_current = max_rank_dict[scenario_name]
            scenario_results[scenario_name] = []

            for k in tqdm(range(1, max_rank_current + 1), desc=f"Reconstructing {scenario_name}", leave=False):
                # 해당 fc layer만 k-rank로 재구성
                W_recon = apply_reconstruction_gpu(saved_weights_all[fc_idx], k, device)
                model.classifier[fc_idx].weight.data = W_recon
                # 나머지 FC 레이어는 원본 그대로 복원
                for other_idx in all_fc_indices:
                    if other_idx != fc_idx:
                        model.classifier[other_idx].weight.data = saved_weights_all[other_idx].to(device)

                # 재구성 후 정확도 측정
                acc_recon = test_model_accuracy(model, testloader, device)
                scenario_results[scenario_name].append(acc_recon)

            # 결과 저장
            norm_recon_acc = np.array(scenario_results[scenario_name])
            baseline = norm_recon_acc[-1]  # max rank 시 결과
            if baseline > 0:
                norm_recon_acc = norm_recon_acc / baseline

            np.savetxt(os.path.join(base_dir, f"recon_{scenario_name}_acc_epoch{epoch+1}.csv"), scenario_results[scenario_name])
            np.savetxt(os.path.join(base_dir, f"recon_{scenario_name}_acc_norm_epoch{epoch+1}.csv"), norm_recon_acc)
            print(f"Reconstruction results saved for scenario '{scenario_name}' at epoch {epoch+1}")

# ------------------------------
# 7. Save Training Accuracy Curve
# ------------------------------
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
# 8. Overlay Graphs for Reconstruction Results
# ------------------------------
for scenario_name in fc_indices.keys():
    # (1) 정규화된 재구성 정확도
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

    # (2) 원본 정확도(unnormalized)
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
