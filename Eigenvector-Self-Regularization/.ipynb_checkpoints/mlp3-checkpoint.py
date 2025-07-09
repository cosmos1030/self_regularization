import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

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
        self.fc1 = nn.Linear(3 * 32 * 32, 1024) #hyperparameters
        self.fc2 = nn.Linear(1024, 256) #hyperparameters
        self.fc3 = nn.Linear(256, 10) #hyperparameters
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.softmax(self.fc3(x), dim=1)
        return x
    
model = MLP3()

seed = 0 #seed number
torch.manual_seed(seed)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

accuracy = []

for epoch in range(100):
    total_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        labels = functional.one_hot(labels,num_classes=10).float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += (batch_size * loss / 50000).item()


    model.eval()

    with torch.no_grad():
        w1 = model.fc1.weight.detach().numpy()
        n1, p1 = np.shape(w1)
        u1, s1, vh1 = np.linalg.svd(w1)
        spec1 = np.square(s1)/n1 #spectrum of fc1
        
        fig1, ax1 = plt.subplots()
        ax1.hist(spec1, bins=300, density=True)
        fig1.savefig(f'fc1_{epoch + 1}.png')
        plt.close(fig1)

        w2 = model.fc2.weight.detach().numpy()
        n2, p2 = np.shape(w2)
        u2, s2, vh2 = np.linalg.svd(w2)
        spec2 = np.square(s2)/n2 #spectrum of fc2

        fig2, ax2 = plt.subplots()
        ax2.hist(spec2, bins=300, density=True)
        fig2.savefig(f'fc2_{epoch + 1}.png')
        plt.close(fig2)

        w3 = model.fc3.weight.detach().numpy()
        n3, p3 = np.shape(w3)
        u3, s3, vh3 = np.linalg.svd(w3)
        spec3 = np.square(s3)/n3 #spectrum of fc3

        fig3, ax3 = plt.subplots()
        ax3.hist(spec3, bins=300, density=True)
        fig3.savefig(f'fc3_{epoch + 1}.png')
        plt.close(fig3)

        v1 = vh1[0,:] #leading eigenvector of fc1
        v2 = vh2[0,:] #leading eigenvector of fc2
        v3 = vh3[0,:] #leading eigenvector of fc3
        with open("fc1.csv",'a') as f:
            np.savetxt(f, v1.reshape(1,-1))
            f.write('\n')
        with open("fc2.csv",'a') as g:
            np.savetxt(g, v2.reshape(1,-1))
            g.write('\n')
        with open("fc3.csv",'a') as h:
            np.savetxt(h, v3.reshape(1,-1))
            h.write('\n')

        b1 = model.fc1.bias.detach().numpy()
        b2 = model.fc2.bias.detach().numpy()
        b3 = model.fc3.bias.detach().numpy()

        with open("fc1_bias.csv",'a') as fb:
            np.savetxt(fb, b1.reshape(1,-1))
            fb.write('\n')
        with open("fc2_bias.csv",'a') as gb:
            np.savetxt(gb, b2.reshape(1,-1))
            gb.write('\n')
        with open("fc3_bias.csv",'a') as hb:
            np.savetxt(hb, b3.reshape(1,-1))
            hb.write('\n')
        
        for testdata in testloader: #accuracy on test set
            images, testlabels = testdata
            testlabels = functional.one_hot(testlabels,num_classes=10).float()
            predictions = model(images)
            _, predicted_labels = torch.max(predictions.data, 1)
            total += testlabels.size(0)
            correct += (predicted_labels == testlabels).sum().item()

        acc = correct / total
        accuracy.append(acc)


    model.train()

    print(f'epoch {epoch + 1} finished')

    if total_loss < 0.0001:
        break

acc_array = np.array(accuracy)
np.savetxt("accuracy.csv", acc_array)
fig, ax = plt.subplots()
ax.plot(acc_array)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracies across epochs')
fig.savefig('accuracy.png')
plt.close(fig)

print('finished training')
