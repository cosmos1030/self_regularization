#!/usr/bin/env python
# ------------------------------------------------------------
# 0. 라이브러리
# ------------------------------------------------------------
import torch, torchvision, torchvision.transforms as T
import torch.nn as nn, torch.optim as optim
import numpy as np, matplotlib.pyplot as plt, os, shutil
from tqdm import tqdm
from collections import defaultdict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# ------------------------------------------------------------
# 1. 하이퍼·경로 (+Noise·Subsample 설정)
# ------------------------------------------------------------
batch_size    = 64
seed          = 100
learning_rate = 0.0001
epochs        = 100
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recon_epochs  = []
optimizer_type= 'sgd'

# ---------- label-noise ----------
noise_ratio = 0       # 0.0→clean, 0.3→30 % noise
noise_seed  = 5
# ---------- subsample ----------
train_percent = 1     # 1.0→전량, 0.3→30 %만 사용
# ------------------------------------------------------------

base_dir = (f"runs/alex_seed{seed}_batch{batch_size}_{optimizer_type}"
            f"_lr{learning_rate}_epochs{epochs}"
            f"_noise{int(noise_ratio*100)}_seed{noise_seed}_sub{int(train_percent*100)}_jse_eigen_final")
if os.path.exists(base_dir): shutil.rmtree(base_dir)
os.makedirs(base_dir)
eigenvector_dir=os.path.join(base_dir,'eigenvectors'); os.makedirs(eigenvector_dir)
bias_dir       =os.path.join(base_dir,'biases');      os.makedirs(bias_dir)
spectrum_dir   =os.path.join(base_dir,'spectrum');    os.makedirs(spectrum_dir)

# ------------------------------------------------------------
# 2. CIFAR-10 (+Noise +Subsample)
# ------------------------------------------------------------
tfm = T.Compose([T.ToTensor(),
                 T.Normalize((0.4914,0.4822,0.4465),
                             (0.247 ,0.243 ,0.261 ))])

trainset = torchvision.datasets.CIFAR10("./data", True , tfm, download=True)
testset  = torchvision.datasets.CIFAR10("./data", False, tfm, download=True)

# ---- (1) 라벨 노이즈 ----
if noise_ratio > 0:
    np.random.seed(noise_seed)
    tgt = np.array(trainset.targets)
    n_noisy   = int(noise_ratio*len(tgt))
    idx_noisy = np.random.choice(len(tgt), n_noisy, replace=False)
    tgt[idx_noisy] = np.random.randint(10, size=n_noisy)
    trainset.targets = tgt.tolist()

# ---- (2) 클래스-균형 서브샘플 ----
def subsample_by_class(dataset, percent: float, seed_: int):
    if percent >= 1.0: return dataset
    rng  = np.random.default_rng(seed_)
    cls_indices = defaultdict(list)
    for idx, lab in enumerate(dataset.targets): cls_indices[lab].append(idx)
    keep = []
    for lab, idxs in cls_indices.items():
        m = max(1, int(len(idxs)*percent))
        keep.extend(rng.choice(idxs, m, replace=False))
    keep.sort()
    dataset.data    = dataset.data[keep]
    dataset.targets = [dataset.targets[i] for i in keep]
    return dataset

trainset = subsample_by_class(trainset, train_percent, seed)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset , batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------
# 3. miniAlexNet
# ------------------------------------------------------------
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
        x=self.features(x)
        x=torch.flatten(x,1)
        return self.classifier(x)

torch.manual_seed(seed)
model = miniAlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = (optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
             if optimizer_type=='sgd'
             else optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999)))

# ------------------------------------------------------------
# 4. Accuracy util
# ------------------------------------------------------------
def accuracy(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            c += (model(x).argmax(1)==y).sum().item(); t += y.size(0)
    return c/t

# ------------------------------------------------------------
# 5. SVD·JS helper (top-k, JS-vec, JS-full)
# ------------------------------------------------------------
def topk_recon(W_cpu,k):
    W=W_cpu.to(device)
    U,S,Vh=torch.linalg.svd(W,full_matrices=False)
    return U[:,:k] @ torch.diag(S[:k]) @ Vh[:k]

def js_vec(W_cpu,k):
    W=W_cpu.to(device)
    U,S,Vh=torch.linalg.svd(W,full_matrices=False)
    U_k,S_k,Vh_k=U[:,:k],S[:k],Vh[:k]
    n=W.size(0); H=U_k*(S_k/np.sqrt(n)).reshape(1,-1)
    p=H.size(0); e=torch.ones(p,1,device=device)
    M=(e@(e.T@H))/p; R=H-M
    nu2=(R**2).sum()/(R.numel()-k)
    C=torch.eye(k,device=device)-nu2*torch.linalg.inv(R.T@R)
    Hj=H@C+M@(torch.eye(k,device=device)-C)
    Uj,_,_=torch.linalg.svd(Hj,full_matrices=False)
    return Uj[:,:k] @ torch.diag(S_k) @ Vh_k

def js_full(W_cpu,k):
    W=W_cpu.to(device)
    U,S,Vh=torch.linalg.svd(W,full_matrices=False)
    U_k,S_k,Vh_k=U[:,:k],S[:k],Vh[:k]
    n=W.size(0); H=U_k*(S_k/np.sqrt(n)).reshape(1,-1)
    p=H.size(0); e=torch.ones(p,1,device=device)
    M=(e@(e.T@H))/p; R=H-M
    nu2=(R**2).sum()/(R.numel()-k)
    C=torch.eye(k,device=device)-nu2*torch.linalg.inv(R.T@R)
    Hj=H@C+M@(torch.eye(k,device=device)-C)
    sigma_js=torch.sqrt(torch.clamp(S_k**2 - n*nu2,0))
    Uj,_,_=torch.linalg.svd(Hj,full_matrices=False)
    return Uj[:,:k] @ torch.diag(sigma_js) @ Vh_k

# ------------------------------------------------------------
# 6. FC info
# ------------------------------------------------------------
fc_indices={"fc1":0,"fc2":2,"fc3":4}
max_rank  ={n:min(model.classifier[i].weight.shape) for n,i in fc_indices.items()}

# ------------------------------------------------------------
# 7. Train + Spectrum + Reconstruction
# ------------------------------------------------------------
train_acc, test_acc = [],[]

for ep in range(1,epochs+1):
    # ----- Train -----
    model.train()
    for x,y in tqdm(trainloader, desc=f"[Train] {ep}/{epochs}", leave=False):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(); loss = criterion(model(x),y)
        loss.backward(); optimizer.step()

    train_acc.append(accuracy(model,trainloader))
    test_acc .append(accuracy(model,testloader))

    # ----- Spectrum -----
    for idx,(name,fc_idx) in enumerate(fc_indices.items(), start=1):
        W=model.classifier[fc_idx].weight.detach().cpu().numpy()
        U,S,_ = np.linalg.svd(W,full_matrices=False)
        n,p   = W.shape
        spec  = (S**2)/n
        mp    = np.var(W)*(1+np.sqrt(p/n))**2
        spikes = spec[spec>mp]; bulk=spec[spec<=mp]
        spike_cnt=len(spikes); nu2_est = bulk.sum()/max(len(bulk),1)

        plt.figure(figsize=(6,4))
        plt.hist(spec,bins=300,density=True,alpha=.6)
        plt.axvline(mp,color='r',label='MP+')
        plt.text(0.05,0.95,
                 f"Spikes: {spike_cnt}\nν²: {nu2_est:.4f}",
                 transform=plt.gca().transAxes,
                 va='top',ha='left',
                 bbox=dict(boxstyle='round',facecolor='w',alpha=.8))
        plt.title(f"FC{idx} Spectrum Ep{ep}"); plt.legend()
        plt.savefig(f"{spectrum_dir}/fc{idx}_spectrum_epoch{ep}.png"); plt.close()

    # ──────────────────────────────────────────────────────────
    # ###   EV / BIAS DUMP   ##################################
    # ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        # conv layers
        conv_idx = 1
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                W = layer.weight.detach().cpu().numpy().reshape(layer.out_channels, -1)
                _,_,Vh = np.linalg.svd(W)
                lead_ev = Vh[0,:]
                with open(os.path.join(eigenvector_dir,f"conv{conv_idx}.csv"),"a") as f:
                    np.savetxt(f, lead_ev.reshape(1,-1))
                conv_idx += 1

        # fc layers (eigenvector + bias)
        fc_idx_dump = 1
        Vh_list = []
        bias_list = []
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                _,_,Vh = np.linalg.svd(W, full_matrices=False)
                Vh_list.append(Vh)
                bias_list.append(b)
                lead_ev = Vh[0,:]
                sub_lead_ev = Vh[1, :]
                last_ev = Vh[-1,:]
                
                with open(os.path.join(eigenvector_dir,f"fc{fc_idx_dump}_leading.csv"),"a") as f:
                    np.savetxt(f, lead_ev.reshape(1,-1))
                with open(os.path.join(eigenvector_dir,f"fc{fc_idx_dump}_subleading.csv"),"a") as f:
                    np.savetxt(f, sub_lead_ev.reshape(1,-1))
                with open(os.path.join(eigenvector_dir,f"fc{fc_idx_dump}_last.csv"),"a") as f:
                    np.savetxt(f, last_ev.reshape(1,-1))
                # with open(os.path.join(bias_dir,f"fc{fc_idx_dump}_bias.csv"),"a") as f:
                    # np.savetxt(f, b.reshape(1,-1))
                # np.savez_compressed(
                #     os.path.join(eigenvector_dir, f"fc{fc_idx_dump}_epoch{ep:03d}.npz"),
                #     Vh=Vh, bias=b
                # )
                fc_idx_dump += 1
        np.savez_compressed(
            os.path.join(eigenvector_dir, f"epoch{str(ep)}.npz"),
            fc1_Vh=Vh_list[0], fc1_bias=bias_list[0],
            fc2_Vh=Vh_list[1], fc2_bias=bias_list[1],
            fc3_Vh=Vh_list[2], fc3_bias=bias_list[2],
        )

    # ──────────────────────────────────────────────────────────

    # ----- Reconstruction -----
    if ep in recon_epochs:
        saved={i:model.classifier[i].weight.data.cpu().clone() for i in fc_indices.values()}
        for name,idx in fc_indices.items():
            raw,vec,ful=[],[],[]
            r_max=max_rank[name]
            for k in tqdm(range(1,r_max+1),desc=f"[Recon {name}] Ep{ep}",leave=False):
                model.classifier[idx].weight.data = topk_recon(saved[idx],k).to(device)
                acc_raw = accuracy(model,testloader); raw.append(acc_raw)

                if k==r_max:         # full-rank → shrink skip
                    vec.append(acc_raw); ful.append(acc_raw)
                else:
                    model.classifier[idx].weight.data = js_vec(saved[idx],k).to(device)
                    vec.append(accuracy(model,testloader))
                    model.classifier[idx].weight.data = js_full(saved[idx],k).to(device)
                    ful.append(accuracy(model,testloader))

                for j in fc_indices.values():
                    if j!=idx: model.classifier[j].weight.data = saved[j].to(device)

            np.savetxt(f"{base_dir}/recon_{name}_acc_epoch{ep}.csv"      ,raw)
            np.savetxt(f"{base_dir}/recon_{name}_vec_epoch{ep}.csv"      ,vec)
            np.savetxt(f"{base_dir}/recon_{name}_full_epoch{ep}.csv"     ,ful)
            np.savetxt(f"{base_dir}/recon_{name}_acc_norm_epoch{ep}.csv" ,np.array(raw)/raw[-1])
            np.savetxt(f"{base_dir}/recon_{name}_vec_norm_epoch{ep}.csv" ,np.array(vec)/vec[-1])
            np.savetxt(f"{base_dir}/recon_{name}_full_norm_epoch{ep}.csv",np.array(ful)/ful[-1])

# ------------------------------------------------------------
# 8. Overlay Graphs (그대로)
# ------------------------------------------------------------
for name in fc_indices.keys():
    fig,ax=plt.subplots(figsize=(10,6))
    for ep in recon_epochs:
        for tag,ls in [("acc_norm","-"),("vec_norm","--"),("full_norm",":")]:
            f=f"{base_dir}/recon_{name}_{tag}_epoch{ep}.csv"
            if os.path.exists(f):
                y=np.loadtxt(f); ax.plot(range(1,len(y)+1),y,ls,label=f"Ep{ep}-{tag.replace('_norm','')}")
    ax.set_xlabel("Rank k"); ax.set_ylabel("Normalized Test Acc")
    ax.set_title(f"Normalized Reconstruction Accuracy vs Rank ({name})")
    ax.legend(); ax.grid(True)
    plt.savefig(f"{base_dir}/recon_{name}_acc_norm_overlay.png"); plt.close()

    fig,ax=plt.subplots(figsize=(10,6))
    for ep in recon_epochs:
        for tag,ls in [("acc","-"),("vec","--"),("full",":")]:
            f=f"{base_dir}/recon_{name}_{tag}_epoch{ep}.csv"
            if os.path.exists(f):
                y=np.loadtxt(f); ax.plot(range(1,len(y)+1),y,ls,label=f"Ep{ep}-{tag}")
    ax.set_xlabel("Rank k"); ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Reconstruction Accuracy vs Rank ({name})")
    ax.legend(); ax.grid(True)
    plt.savefig(f"{base_dir}/recon_{name}_acc_overlay.png"); plt.close()

# ------------------------------------------------------------
# 9. Accuracy curve 저장
# ------------------------------------------------------------
np.savetxt(f"{base_dir}/train_accuracy.csv",np.array(train_acc))
np.savetxt(f"{base_dir}/test_accuracy.csv" ,np.array(test_acc))
plt.figure(); plt.plot(train_acc,label='train'); plt.plot(test_acc,label='test')
plt.legend(); plt.title("Accuracy"); plt.savefig(f"{base_dir}/train_test_accuracy.png"); plt.close()
