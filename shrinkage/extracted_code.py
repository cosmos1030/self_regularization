import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as functional
import matplotlib.pyplot as plt
from tqdm import tqdm
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def test(model, testloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0], data[1]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()

    acc = correct / total

    return acc

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
seed = 726
torch.manual_seed(seed)
model = miniAlexNet()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
epochs = 100
accuracy = []

for epoch in range(epochs):
    correct = 0.0
    total = 0.0

    model.train()
    total_loss = 0.0

    train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True)
    for i, data in enumerate(train_pbar):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1) 
        train_pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    model.eval()
    with torch.no_grad():
        test_pbar = tqdm(testloader, desc=f"Epoch {epoch+1}/{epochs} [Testing]")
        for testdata in test_pbar:
            images, testlabels = testdata
            testlabels = functional.one_hot(testlabels,num_classes=10).float()
            predictions = model(images)
            _, predicted_labels = torch.max(predictions.data, 1)
            total += testlabels.size(0)
            correct += (predicted_labels == testlabels.argmax(dim=1)).sum().item()

        acc = correct / total
        accuracy.append(acc)
    
    print(f'epoch {epoch + 1} finished, test accuracy: {acc:.4f}')
    if total_loss < 0.0001:
        break

print('Finished training')

acc_array = np.array(accuracy)
fig, ax = plt.subplots()
ax.plot(acc_array)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracies across epochs')
plt.show()
model.eval()

w1 = model.classifier[0].weight.detach()
w2 = model.classifier[2].weight.detach()
w3 = model.classifier[4].weight.detach()

u1, s1, vh1 = torch.linalg.svd(w1, full_matrices=False)
u2, s2, vh2 = torch.linalg.svd(w2, full_matrices=False)
u3, s3, vh3 = torch.linalg.svd(w3, full_matrices=False)

n1, p1 = w1.size()
n2, p2 = w2.size()
n3, p3 = w3.size()

print(n1,p1,n2,p2,n3,p3)
k1 = min(n1,p1)
k2 = min(n2,p2)
fig, ax = plt.subplots(3,2, figsize=(12,10))
ax[0,0].hist(s1**2 / n1, density=True, bins=300)
ax[0,1].hist(s1**2 / p1, density=True, bins=300)
ax[1,0].hist(s2**2 / n2, density=True, bins=300)
ax[1,1].hist(s2**2 / p2, density=True, bins=300)
ax[2,0].hist(s3**2 / n3, density=True, bins=100)
ax[2,1].hist(s3**2 / p3, density=True, bins=100)
plt.show()
model_lr = miniAlexNet()
model_reg_travg = miniAlexNet()

model_lr.load_state_dict(model.state_dict())
model_reg_travg.load_state_dict(model.state_dict())
r1 = 9
r2 = 9

model_lr.classifier[0].weight = nn.Parameter(u1[:,0:r1] @ torch.diag(s1[0:r1]) @ vh1[0:r1,:])
model_lr.classifier[2].weight = nn.Parameter(u2[:,0:r2] @ torch.diag(s2[0:r2]) @ vh2[0:r2,:])

acc_lr = test(model_lr, testloader)
acc_orig = test(model, testloader)

print(acc_lr, acc_lr / acc_orig)
print(s1[0:r1].sum() / s1.sum())
print(s2[0:r2].sum() / s2.sum())
u1_reg_travg, spec_n1, _ = torch.linalg.svd(u1[:,0:r1] @ torch.diag(s1[0:r1]**2) @ u1[:,0:r1].T + 
                                      torch.trace(w1 @ w1.T - u1[:,0:r1] @ torch.diag(s1[0:r1]**2) @ u1[:,0:r1].T) * 
                                      torch.eye(n1) / n1)
u2_reg_travg, spec_n2, _ = torch.linalg.svd(u2[:,0:r2] @ torch.diag(s2[0:r2]**2) @ u2[:,0:r2].T + 
                                      torch.trace(w2 @ w2.T - u2[:,0:r2] @ torch.diag(s2[0:r2]**2) @ u2[:,0:r2].T) * 
                                      torch.eye(n2) / n2)
_, spec_p1, vh1_reg_travg = torch.linalg.svd(vh1[0:r1,:].T @ torch.diag(s1[0:r1]**2) @ vh1[0:r1,:] + 
                                       torch.trace(w1.T @ w1 - vh1[0:r1,:].T @ torch.diag(s1[0:r1]**2) @ vh1[0:r1,:]) * 
                                       torch.eye(p1) / p1)
_, spec_p2, vh2_reg_travg = torch.linalg.svd(vh2[0:r2,:].T @ torch.diag(s2[0:r2]**2) @ vh2[0:r2,:] + 
                                       torch.trace(w2.T @ w2 - vh2[0:r2,:].T @ torch.diag(s2[0:r2]**2) @ vh2[0:r2,:]) * 
                                       torch.eye(p2) / p2)
fig_reg, ax_reg = plt.subplots(2,2, figsize=(12,7))
ax_reg[0,0].hist(spec_p1[0:k1]**2 / n1, density=True, bins=300)
ax_reg[0,1].hist(spec_n1[0:k1]**2 / p1, density=True, bins=300)
ax_reg[1,0].hist(spec_p2[0:k2]**2 / n2, density=True, bins=300)
ax_reg[1,1].hist(spec_n2[0:k2]**2 / p2, density=True, bins=300)
plt.show()
model_reg_travg.classifier[0].weight = nn.Parameter(u1_reg_travg[:,0:k1] @ torch.diag(s1[0:k1]) @ vh1_reg_travg[0:k1,:])
model_reg_travg.classifier[2].weight = nn.Parameter(u2_reg_travg[:,0:k2] @ torch.diag(s2[0:k2]) @ vh2_reg_travg[0:k2,:])

acc_reg_travg = test(model_reg_travg, testloader)

print(acc_reg_travg, acc_reg_travg / acc_orig)
model_reg_diag = miniAlexNet()
model_reg_diag.load_state_dict(model.state_dict())
u1_reg_diag, spec_diag_n1, _ = torch.linalg.svd(u1[:,0:r1] @ torch.diag(s1[0:r1]**2) @ u1[:,0:r1].T + 
                                      torch.diag(torch.diag(w1 @ w1.T - u1[:,0:r1] @ torch.diag(s1[0:r1]**2) @ u1[:,0:r1].T)))
u2_reg_diag, spec_diag_n2, _ = torch.linalg.svd(u2[:,0:r2] @ torch.diag(s2[0:r2]**2) @ u2[:,0:r2].T + 
                                      torch.diag(torch.diag(w2 @ w2.T - u2[:,0:r2] @ torch.diag(s2[0:r2]**2) @ u2[:,0:r2].T)))
_, spec_diag_p1, vh1_reg_diag = torch.linalg.svd(vh1[0:r1,:].T @ torch.diag(s1[0:r1]**2) @ vh1[0:r1,:] + 
                                       torch.diag(torch.diag(w1.T @ w1 - vh1[0:r1,:].T @ torch.diag(s1[0:r1]**2) @ vh1[0:r1,:])))
_, spec_diag_p2, vh2_reg_diag = torch.linalg.svd(vh2[0:r2,:].T @ torch.diag(s2[0:r2]**2) @ vh2[0:r2,:] + 
                                       torch.diag(torch.diag(w2.T @ w2 - vh2[0:r2,:].T @ torch.diag(s2[0:r2]**2) @ vh2[0:r2,:])))
fig_reg_diag, ax_reg_diag = plt.subplots(2,2, figsize=(12,7))
ax_reg_diag[0,0].hist(spec_diag_p1[0:k1]**2 / n1, density=True, bins=300)
ax_reg_diag[0,1].hist(spec_diag_n1[0:k1]**2 / p1, density=True, bins=300)
ax_reg_diag[1,0].hist(spec_diag_p2[0:k2]**2 / n2, density=True, bins=300)
ax_reg_diag[1,1].hist(spec_diag_n2[0:k2]**2 / p2, density=True, bins=300)
plt.show()
model_reg_diag.classifier[0].weight = nn.Parameter(u1_reg_diag[:,0:k1] @ torch.diag(s1[0:k1]) @ vh1_reg_diag[0:k1,:])
model_reg_diag.classifier[2].weight = nn.Parameter(u2_reg_diag[:,0:k2] @ torch.diag(s2[0:k2]) @ vh2_reg_diag[0:k2,:])

acc_reg_diag = test(model_reg_diag, testloader)

print(acc_reg_diag, acc_reg_diag / acc_orig)
