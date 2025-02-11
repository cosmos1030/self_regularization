import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Data transformation
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

batch_size = 8

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
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 100

accuracy = []

base_dir = os.path.join("runs", f"alex_seed{seed}_batch{batch_size}_sgd_lr{learning_rate}_epochs{epochs}")

os.makedirs(base_dir, exist_ok=True)

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

# Training loop
for epoch in range(epochs):
    correct = 0.0
    total = 0.0

    model.train()
    total_loss = 0.0

    train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
    for i, data in enumerate(train_pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        layer_index = 1
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight.detach().cpu().numpy().reshape(layer.out_channels, -1)
                u, s, vh = np.linalg.svd(weight)
                leading_eigenvector = vh[0, :]
                with open(os.path.join(eigenvector_dir, f'conv{layer_index}.csv'), 'a') as f:
                    np.savetxt(f, leading_eigenvector.reshape(1, -1))
                layer_index += 1
        
        layer_index = 1
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().numpy()
                u, s, vh = np.linalg.svd(weight)
                leading_eigenvector = vh[0, :]
                with open(os.path.join(eigenvector_dir, f'fc{layer_index}.csv'), 'a') as f:
                    np.savetxt(f, leading_eigenvector.reshape(1, -1))
                layer_index += 1
        
        test_pbar = tqdm(testloader, desc=f"Epoch {epoch+1}/{epochs} [Testing]")
        for testdata in test_pbar: #accuracy on test set
            images, testlabels = testdata
            images = images.to(device)
            testlabels = functional.one_hot(testlabels,num_classes=10).float().to(device)
            predictions = model(images)
            _, predicted_labels = torch.max(predictions.data, 1)
            total += testlabels.size(0)
            correct += (predicted_labels == testlabels.argmax(dim=1)).sum().item()

        acc = correct / total
        accuracy.append(acc)
    
    print(f'epoch {epoch + 1} finished, test accuracy: {acc:.4f}')
    if total_loss < 0.0001:
        break

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
