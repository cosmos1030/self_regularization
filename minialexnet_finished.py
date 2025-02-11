import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
import matplotlib.pyplot as plt
import os

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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Set up directories for saving outputs
i = 4
base_dir = None
while True:
    if not os.path.isdir(f'run{i}'):
        os.mkdir(f"run{i}")
        base_dir = f"run{i}"
        break
    else:
        i += 1

eigenvector_dir = os.path.join(base_dir, 'eigenvectors')
bias_dir = os.path.join(base_dir, 'biases')
spectrum_dir = os.path.join(base_dir, 'spectrum')

os.mkdir(eigenvector_dir)
os.mkdir(bias_dir)
os.mkdir(spectrum_dir)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizing {str(device)}")
model.to(device)

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(trainloader, 0):
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
    
    print(f'epoch {epoch + 1} finished')
    if total_loss < 0.0001:
        break

print('Finished training')
