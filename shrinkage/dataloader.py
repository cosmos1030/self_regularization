import torchvision, torchvision.transforms as T
import torch
import numpy as np
from collections import defaultdict

def generate_label_noise(noise_ratio, trainset, num_classes, seed_):
    # ---- (1) 라벨 노이즈 ----
    if noise_ratio > 0:
        np.random.seed(seed_)
        target = np.array(trainset.targets)
        n_noisy   = int(noise_ratio*len(target))
        idx_noisy = np.random.choice(len(target), n_noisy, replace=False)
        
        target[idx_noisy] = np.random.randint(num_classes, size=n_noisy)
        trainset.targets = target.tolist()
        return trainset
    else:
        return trainset

# ---- 클래스-균형 서브샘플 ----
def subsample_by_class(trainset, ratio: float, seed_: int):
    if ratio >= 1.0: 
        return trainset
    rng  = np.random.default_rng(seed_)
    cls_indices = defaultdict(list)
    for idx, lab in enumerate(trainset.targets): cls_indices[lab].append(idx)
    keep = []
    for lab, idxs in cls_indices.items():
        m = max(1, int(len(idxs)*ratio))
        keep.extend(rng.choice(idxs, m, replace=False))
    keep.sort()
    trainset.data    = trainset.data[keep]
    trainset.targets = [trainset.targets[i] for i in keep]
    return trainset

def get_dataloader(dataset, batch_size, noise_ratio, subsample_ratio, seed):
    # ------------------------------------------------------------
    # 2. CIFAR-10 (+Noise +Subsample)
    # ------------------------------------------------------------
    if dataset == 'CIFAR10':
        num_classes = 10
        num_input_channel = 3
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247 ,0.243 ,0.261 ))
        ])
        trainset = torchvision.datasets.CIFAR10("./data", True , tfm, download=True)
        testset  = torchvision.datasets.CIFAR10("./data", False, tfm, download=True)

    elif dataset == 'CIFAR100':
        num_classes = 100
        num_input_channel = 3
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        trainset = torchvision.datasets.CIFAR100("./data", True , tfm, download=True)
        testset  = torchvision.datasets.CIFAR100("./data", False, tfm, download=True)

    elif dataset == 'MNIST':
        num_classes = 10
        num_input_channel = 1
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST("./data", True , tfm, download=True)
        testset  = torchvision.datasets.MNIST("./data", False, tfm, download=True)
    trainset = generate_label_noise(noise_ratio, trainset, num_classes, seed)
    trainset = subsample_by_class(trainset, subsample_ratio, seed)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = torch.utils.data.DataLoader(testset , batch_size=batch_size, shuffle=False)
    return num_input_channel, num_classes, trainloader, testloader
