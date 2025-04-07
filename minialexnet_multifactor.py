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
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes)
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
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


epochs = 100
accuracy = []

base_dir = os.path.join("runs", f"alex_seed{seed}_batch{batch_size}_sgd_lr{learning_rate}_epochs{epochs}_factor")

# --- NEW CODE: Delete existing base_dir if it exists ---
if os.path.exists(base_dir):
    print(f"Directory '{base_dir}' already exists. Removing it...")
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

# Directories for saving eigenvectors, biases, spectrum, etc.
eigenvector_dir = os.path.join(base_dir, 'eigenvectors')
bias_dir = os.path.join(base_dir, 'biases')
spectrum_dir = os.path.join(base_dir, 'spectrum')

os.makedirs(eigenvector_dir, exist_ok=True)
os.makedirs(bias_dir, exist_ok=True)
os.makedirs(spectrum_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizing {str(device)}")
model.to(device)

# --- Helper function: test accuracy ---
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

# --- Helper functions for linear weight reconstruction experiment ---
def get_linear_weights(model):
    """Save a copy of all Linear layer weights from model.classifier."""
    saved_weights = []
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            saved_weights.append(layer.weight.data.clone().cpu())
    return saved_weights

def apply_rank_k_reconstruction(model, saved_weights, k):
    """Apply rank-k reconstruction to each Linear layer using saved weights."""
    with torch.no_grad():
        idx = 0
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                W = saved_weights[idx].numpy()  # shape: (out_features, in_features)
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                r = min(k, len(S))
                W_recon = np.zeros_like(W)
                for i in range(r):
                    W_recon += S[i] * np.outer(U[:, i], Vh[i, :])
                layer.weight.data = torch.tensor(W_recon, dtype=torch.float32).to(layer.weight.device)
                idx += 1

def restore_linear_weights(model, saved_weights):
    """Restore the original Linear layer weights from saved copy."""
    with torch.no_grad():
        idx = 0
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data = saved_weights[idx].to(layer.weight.device)
                idx += 1

# --- Training loop ---
# reconstruction 실험을 할 에폭들을 지정합니다.
# 0 에폭(학습 전), 25, 50, 75, 그리고 100번째 에폭 (실제 표기상 100으로 표시)
recon_epochs = [25, 50, 75,100]

for epoch in range(epochs):
    correct = 0.0
    total = 0.0
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

    # --- Save eigenvector and bias info (기존 코드 그대로) ---
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

    # --- Reconstruction experiment for Linear layers ---
    # 실제 에폭 번호(출력 시 1~100로 표시)를 위해 epoch+1과 recon_epochs를 비교합니다.
    if (epoch+1) in recon_epochs:
        print(f"→ Performing reconstruction experiment at epoch {epoch+1}")
        # Save current Linear weights
        saved_linear_weights = get_linear_weights(model)
        max_rank = 500
        recon_acc_list = []
        # Loop over rank k from 1 to max_rank
        for k in tqdm(range(1, max_rank + 1), desc=f"Reconstruction (Epoch {epoch+1})", leave=False):
            apply_rank_k_reconstruction(model, saved_linear_weights, k)
            acc_recon = test_model_accuracy(model, testloader, device)
            recon_acc_list.append(acc_recon)
            # Restore original weights before next k
            restore_linear_weights(model, saved_linear_weights)
        recon_acc_list = np.array(recon_acc_list)
        # Normalize: k=1000 reconstruction accuracy를 1로 설정
        baseline = recon_acc_list[-1]
        if baseline > 0:
            norm_recon_acc = recon_acc_list / baseline
        else:
            norm_recon_acc = recon_acc_list
        # Save both raw and normalized reconstruction accuracy
        np.savetxt(os.path.join(base_dir, f"recon_acc_epoch{epoch+1}.csv"), recon_acc_list)
        np.savetxt(os.path.join(base_dir, f"recon_acc_norm_epoch{epoch+1}.csv"), norm_recon_acc)

# Save training accuracy curve
acc_array = np.array(accuracy)
np.savetxt(os.path.join(base_dir, "accuracy.csv"), acc_array)
fig, ax = plt.subplots()
ax.plot(acc_array)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracies across epochs')
fig.savefig(os.path.join(base_dir, 'accuracy.png'))
plt.close(fig)

print('Finished training')

# --- Bias PCA analysis (기존 코드 그대로) ---
from sklearn.decomposition import PCA

pca_results = {}

for file in sorted(os.listdir(bias_dir)):
    file_path = os.path.join(bias_dir, file)
    bias_data = np.loadtxt(file_path, delimiter=' ')
    if len(bias_data.shape) == 1:
        bias_data = bias_data.reshape(1, -1)
    pca = PCA(n_components=1)
    pca.fit(bias_data)
    explained_variance_ratio = pca.explained_variance_ratio_[0]
    principal_components = np.abs(pca.transform(bias_data)).flatten()
    pca_results[file] = (explained_variance_ratio, principal_components)

fig, ax = plt.subplots(figsize=(10, 5))
for layer, (variance_ratio, pc_values) in pca_results.items():
    layer_name = layer.split('.')[0]
    ax.plot(pc_values, label=f"{layer_name} (Var: {variance_ratio:.2f})")
ax.set_xlabel('Epochs')
ax.set_ylabel('PC1 Magnitude')
ax.set_title('Bias PCA First Principal Component Magnitude Over Epochs')
ax.legend()
plt.grid()
plt.savefig(os.path.join(base_dir, 'bias_pca_analysis.png'))
plt.show()

# --- Overlay normalized reconstruction accuracy curves ---
fig, ax = plt.subplots(figsize=(10, 6))
for ep in recon_epochs:
    # ep here is actual epoch number (e.g. 0, 25, 50, 75, 100)
    # 파일은 저장 시에 epoch+1로 저장되었음 (예: "recon_acc_norm_epoch1.csv" for epoch 0)
    recon_norm_csv = os.path.join(base_dir, f"recon_acc_norm_epoch{ep}.csv")
    if os.path.exists(recon_norm_csv):
        norm_acc = np.loadtxt(recon_norm_csv)
        ax.plot(range(1, len(norm_acc) + 1), norm_acc, label=f"Epoch {ep}")
ax.set_xlabel("Number of Singular Components (Rank k)")
ax.set_ylabel("Normalized Test Accuracy")
ax.set_title("Normalized Linear Layer Reconstruction Accuracy vs Rank")
ax.legend()
ax.grid(True)
plt.savefig(os.path.join(base_dir, "recon_acc_norm_overlay.png"))
plt.show()