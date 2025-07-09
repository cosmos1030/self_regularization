########################################################################
# 0. 라이브러리
########################################################################
import torch, torchvision, torchvision.transforms as T
import torch.nn as nn, torch.optim as optim
import numpy as np, matplotlib.pyplot as plt, os, shutil
from tqdm import tqdm

########################################################################
# 1. 하이퍼·경로 (폴더명 기존 유지)  +  Noise 하이퍼
########################################################################
batch_size    = 64
seed          = 100
learning_rate = 0.0001
epochs        = 100
device        = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
recon_epochs  = [25, 50, 75, 100]
optimizer_type= 'sgd'

# ----------  label-noise 설정 ----------
noise_ratio = 0.2          # 0.0 → clean, 0.3 → 30 % noise
noise_seed  = 5
# ---------------------------------------

base_dir = (f"runs/alex_seed{seed}_batch{batch_size}_{optimizer_type}"
            f"_lr{learning_rate}_epochs{epochs}"
            f"_noise{int(noise_ratio*100)}_seed{noise_seed}_jse")
if os.path.exists(base_dir): shutil.rmtree(base_dir)
os.makedirs(base_dir)
eigenvector_dir=os.path.join(base_dir,'eigenvectors'); os.makedirs(eigenvector_dir)
bias_dir       =os.path.join(base_dir,'biases');      os.makedirs(bias_dir)
spectrum_dir   =os.path.join(base_dir,'spectrum');    os.makedirs(spectrum_dir)

########################################################################
# 2. CIFAR-10  (+ 라벨 노이즈 적용)
########################################################################
tfm = T.Compose([T.ToTensor(),
                 T.Normalize((0.4914,0.4822,0.4465),
                             (0.247 ,0.243 ,0.261 ))])

trainset = torchvision.datasets.CIFAR10("./data", True , tfm, download=True)
testset  = torchvision.datasets.CIFAR10("./data", False, tfm, download=True)

# ----------  라벨 노이즈 (uniform random)  ----------
if noise_ratio > 0:
    np.random.seed(noise_seed)
    targets = np.array(trainset.targets)
    n_noisy   = int(noise_ratio * len(targets))
    idx_noisy = np.random.choice(len(targets), n_noisy, replace=False)
    rand_labels = np.random.randint(10, size=n_noisy)
    targets[idx_noisy] = rand_labels
    trainset.targets   = targets.tolist()
# ---------------------------------------------------

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset , batch_size=batch_size, shuffle=False)

########################################################################
# 3. miniAlexNet
########################################################################
class miniAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,96,5,3,2), nn.ReLU(),
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(96,256,5,3,2), nn.ReLU(),
            nn.MaxPool2d(3,1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4,384), nn.ReLU(),  # fc1 idx0
            nn.Linear(384,192)    , nn.ReLU(),  # fc2 idx2
            nn.Linear(192,10))                 # fc3 idx4
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        return self.classifier(x)

torch.manual_seed(seed)
model = miniAlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = (optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
             if optimizer_type=='sgd'
             else optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999)))

########################################################################
# 4. Accuracy util
########################################################################
def accuracy(model,loader):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
    return correct/total

########################################################################
# 5.  SVD·JS helper  (top-k, JS-vec, JS-full)
########################################################################
def topk_recon(W_cpu, k):
    W=W_cpu.to(device)
    U,S,Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:,:k] @ torch.diag(S[:k]) @ Vh[:k]

def js_vec(W_cpu, k):
    W=W_cpu.to(device)
    U,S,Vh = torch.linalg.svd(W, full_matrices=False)
    U_k,S_k,Vh_k = U[:,:k],S[:k],Vh[:k]
    n  = W.size(0)
    H  = U_k * (S_k/np.sqrt(n)).reshape(1,-1)
    p  = H.size(0); e = torch.ones(p,1,device=device)
    M  = (e@(e.T@H))/p; R = H-M
    nu2= (R**2).sum()/(R.numel()-k)
    C  = torch.eye(k,device=device) - nu2*torch.linalg.inv(R.T@R)
    Hj = H@C + M@(torch.eye(k,device=device)-C)
    Uj,_,_ = torch.linalg.svd(Hj, full_matrices=False)
    return Uj[:,:k] @ torch.diag(S_k) @ Vh_k

def js_full(W_cpu, k):
    W=W_cpu.to(device)
    U,S,Vh = torch.linalg.svd(W, full_matrices=False)
    U_k,S_k,Vh_k = U[:,:k],S[:k],Vh[:k]
    n  = W.size(0)
    H  = U_k * (S_k/np.sqrt(n)).reshape(1,-1)
    p  = H.size(0); e = torch.ones(p,1,device=device)
    M  = (e@(e.T@H))/p; R = H-M
    nu2= (R**2).sum()/(R.numel()-k)
    C  = torch.eye(k,device=device) - nu2*torch.linalg.inv(R.T@R)
    Hj = H@C + M@(torch.eye(k,device=device)-C)
    sigma_js = torch.sqrt(torch.clamp(S_k**2 - n*nu2, 0.0))
    Uj,_,_   = torch.linalg.svd(Hj, full_matrices=False)
    return Uj[:,:k] @ torch.diag(sigma_js) @ Vh_k

########################################################################
# 6. FC info
########################################################################
fc_indices = {"fc1":0,"fc2":2,"fc3":4}
max_rank   = {n: min(model.classifier[i].weight.shape)
              for n,i in fc_indices.items()}

########################################################################
# 7. Train + Spectrum + Reconstruction
########################################################################
train_acc, test_acc = [],[]

for ep in range(1, epochs+1):
    # ----- Train -----
    model.train()
    for x,y in tqdm(trainloader, desc=f"[Train] {ep}/{epochs}", leave=False):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(); loss = criterion(model(x),y)
        loss.backward(); optimizer.step()

    train_acc.append(accuracy(model, trainloader))
    test_acc .append(accuracy(model, testloader))

    # ----- Spectrum 저장 -----
    for idx,(name,fc_idx) in enumerate(fc_indices.items(), start=1):
        W = model.classifier[fc_idx].weight.detach().cpu().numpy()
        U,S,_ = np.linalg.svd(W, full_matrices=False)
        n,p   = W.shape
        spec  = (S**2)/n
        mp    = np.var(W)*(1+np.sqrt(p/n))**2

        # === spike 개수 & residual noise variance(ν²) 계산 ===
        spikes   = spec[spec > mp]          # MP 경계 밖
        bulk     = spec[spec <= mp]         # MP 안쪽
        spike_cnt= len(spikes)
        nu2_est  = bulk.sum() / max(len(bulk), 1)   # Tr(N)/(n_+−q) 근사
        # ================================================

        plt.figure(figsize=(6,4))
        plt.hist(spec, bins=300, density=True, alpha=.6)
        plt.axvline(mp, color='r', label='MP+')
        plt.title(f"FC{idx} Spectrum Ep{ep}")
        plt.text(0.05,0.95,
                 f"Spikes : {spike_cnt}\nν²      : {nu2_est:.4f}",
                 transform=plt.gca().transAxes,
                 ha='left', va='top',
                 bbox=dict(boxstyle='round',facecolor='w',alpha=.8))
        plt.legend()
        plt.savefig(f"{spectrum_dir}/fc{idx}_spectrum_epoch{ep}.png"); plt.close()

    # ----- Reconstruction -----
    if ep in recon_epochs:
        saved = {i: model.classifier[i].weight.data.clone().cpu()
                 for i in fc_indices.values()}

        for name, idx in fc_indices.items():
            raw, vec, ful = [], [], []
            r_max = max_rank[name]

            for k in tqdm(range(1, r_max+1), desc=f"[Recon {name}] Ep{ep}", leave=False):

                # baseline (top-k)
                model.classifier[idx].weight.data = topk_recon(saved[idx], k).to(device)
                acc_raw = accuracy(model, testloader)
                raw.append(acc_raw)

                # full-rank에서는 shrink 건너뛰기
                if k == r_max:
                    vec.append(acc_raw)
                    ful.append(acc_raw)
                else:
                    # JS-vec
                    model.classifier[idx].weight.data = js_vec(saved[idx], k).to(device)
                    vec.append(accuracy(model, testloader))
                    # JS-full
                    model.classifier[idx].weight.data = js_full(saved[idx], k).to(device)
                    ful.append(accuracy(model, testloader))

                # 다른 FC 레이어 복구
                for j in fc_indices.values():
                    if j != idx:
                        model.classifier[j].weight.data = saved[j].to(device)

            # 결과 저장
            np.savetxt(f"{base_dir}/recon_{name}_acc_epoch{ep}.csv"      , raw)
            np.savetxt(f"{base_dir}/recon_{name}_vec_epoch{ep}.csv"      , vec)
            np.savetxt(f"{base_dir}/recon_{name}_full_epoch{ep}.csv"     , ful)
            np.savetxt(f"{base_dir}/recon_{name}_acc_norm_epoch{ep}.csv" , np.array(raw)/raw[-1])
            np.savetxt(f"{base_dir}/recon_{name}_vec_norm_epoch{ep}.csv" , np.array(vec)/vec[-1])
            np.savetxt(f"{base_dir}/recon_{name}_full_norm_epoch{ep}.csv", np.array(ful)/ful[-1])

########################################################################
# 8. Overlay Graphs
########################################################################
for name in fc_indices.keys():
    # Normalized
    fig, ax = plt.subplots(figsize=(10,6))
    for ep in recon_epochs:
        for tag,ls in [("acc_norm","-"),("vec_norm","--"),("full_norm",":")]:
            f = f"{base_dir}/recon_{name}_{tag}_epoch{ep}.csv"
            if os.path.exists(f):
                y = np.loadtxt(f)
                ax.plot(range(1,len(y)+1), y, ls, label=f"Ep{ep}-{tag.replace('_norm','')}")
    ax.set_xlabel("Rank k"); ax.set_ylabel("Normalized Test Acc")
    ax.set_title(f"Normalized Reconstruction Accuracy vs Rank ({name})")
    ax.legend(); ax.grid(True)
    plt.savefig(f"{base_dir}/recon_{name}_acc_norm_overlay.png"); plt.close()

    # Raw
    fig, ax = plt.subplots(figsize=(10,6))
    for ep in recon_epochs:
        for tag,ls in [("acc","-"),("vec","--"),("full",":")]:
            f = f"{base_dir}/recon_{name}_{tag}_epoch{ep}.csv"
            if os.path.exists(f):
                y = np.loadtxt(f)
                ax.plot(range(1,len(y)+1), y, ls, label=f"Ep{ep}-{tag}")
    ax.set_xlabel("Rank k"); ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Reconstruction Accuracy vs Rank ({name})")
    ax.legend(); ax.grid(True)
    plt.savefig(f"{base_dir}/recon_{name}_acc_overlay.png"); plt.close()

########################################################################
# 9. Acc curve 저장
########################################################################
np.savetxt(f"{base_dir}/train_accuracy.csv",np.array(train_acc))
np.savetxt(f"{base_dir}/test_accuracy.csv" ,np.array(test_acc))
plt.figure(); plt.plot(train_acc,label='train'); plt.plot(test_acc,label='test')
plt.legend(); plt.title("Accuracy")
plt.savefig(f"{base_dir}/train_test_accuracy.png"); plt.close()
