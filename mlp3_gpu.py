import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# GPU 혹은 CPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 전처리(transform)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)  # hyperparameters
        self.fc2 = nn.Linear(1024, 256)         # hyperparameters
        self.fc3 = nn.Linear(256, 10)          # hyperparameters
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# 모델 생성 및 디바이스 할당
model = MLP3().to(device)

seed = 0  # seed number
torch.manual_seed(seed)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

accuracy = []

for epoch in range(100):
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # GPU로 데이터 이동
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # one-hot 인코딩
        labels_onehot = F.one_hot(labels, num_classes=10).float()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels_onehot)
        loss.backward()
        optimizer.step()

        # 배치 단위로 loss 누적
        total_loss += (batch_size * loss.item() / len(trainloader.dataset))
    
    # 평가 모드로 전환
    model.eval()

    with torch.no_grad():
        # weight를 CPU로 이동하여 SVD 계산
        w1 = model.fc1.weight.detach().cpu().numpy()
        n1, p1 = np.shape(w1)
        u1, s1, vh1 = np.linalg.svd(w1)
        spec1 = np.square(s1) / n1  # spectrum of fc1
        
        fig1, ax1 = plt.subplots()
        ax1.hist(spec1, bins=300, density=True)
        fig1.savefig(f'fc1_{epoch + 1}.png')
        plt.close(fig1)

        w2 = model.fc2.weight.detach().cpu().numpy()
        n2, p2 = np.shape(w2)
        u2, s2, vh2 = np.linalg.svd(w2)
        spec2 = np.square(s2) / n2  # spectrum of fc2

        fig2, ax2 = plt.subplots()
        ax2.hist(spec2, bins=300, density=True)
        fig2.savefig(f'fc2_{epoch + 1}.png')
        plt.close(fig2)

        w3 = model.fc3.weight.detach().cpu().numpy()
        n3, p3 = np.shape(w3)
        u3, s3, vh3 = np.linalg.svd(w3)
        spec3 = np.square(s3) / n3  # spectrum of fc3

        fig3, ax3 = plt.subplots()
        ax3.hist(spec3, bins=300, density=True)
        fig3.savefig(f'fc3_{epoch + 1}.png')
        plt.close(fig3)

        # leading eigenvector (정확히 말하면 svd의 V 행렬에서 첫 번째 행)
        v1 = vh1[0, :]
        v2 = vh2[0, :]
        v3 = vh3[0, :]
        
        # 각 파일에 이어붙여서 저장
        with open("fc1.csv", 'a') as f:
            np.savetxt(f, v1.reshape(1, -1))
            f.write('\n')
        with open("fc2.csv", 'a') as g:
            np.savetxt(g, v2.reshape(1, -1))
            g.write('\n')
        with open("fc3.csv", 'a') as h:
            np.savetxt(h, v3.reshape(1, -1))
            h.write('\n')

        # bias 역시 CPU로 이동하여 저장
        b1 = model.fc1.bias.detach().cpu().numpy()
        b2 = model.fc2.bias.detach().cpu().numpy()
        b3 = model.fc3.bias.detach().cpu().numpy()

        with open("fc1_bias.csv", 'a') as fb:
            np.savetxt(fb, b1.reshape(1, -1))
            fb.write('\n')
        with open("fc2_bias.csv", 'a') as gb:
            np.savetxt(gb, b2.reshape(1, -1))
            gb.write('\n')
        with open("fc3_bias.csv", 'a') as hb:
            np.savetxt(hb, b3.reshape(1, -1))
            hb.write('\n')
        
        # 테스트 정확도 측정
        for testdata in testloader:
            images, testlabels = testdata
            # GPU로 데이터 이동
            images = images.to(device)
            testlabels = testlabels.to(device)
            
            testlabels_onehot = F.one_hot(testlabels, num_classes=10).float()
            predictions = model(images)
            _, predicted_labels = torch.max(predictions.data, 1)
            total += testlabels_onehot.size(0)
            correct += (predicted_labels == testlabels).sum().item()

        acc = correct / total
        accuracy.append(acc)

    print(f'epoch {epoch + 1} finished, loss: {total_loss:.6f}, accuracy: {acc:.4f}')
    
    # loss가 매우 작아지면 학습 중단
    if total_loss < 0.0001:
        break

# 정확도 기록 저장
acc_array = np.array(accuracy)
np.savetxt("accuracy.csv", acc_array)

# 정확도 플롯
fig, ax = plt.subplots()
ax.plot(acc_array)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracies across epochs')
fig.savefig('accuracy.png')
plt.close(fig)

print('finished training')
