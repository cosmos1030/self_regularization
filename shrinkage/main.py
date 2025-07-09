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
import argparse

from dataloader import get_dataloader
from models import miniAlexNet
from normalization import generate_topk_recons_batched, generate_js_recons_batched, generate_travg_recons_batched, generate_diag_recons_batched, generate_shift_rank_k_recons_batched

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'MNIST'], default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
parser.add_argument('--noise_ratio', type=float, default=0, help='0.0→clean, 0.3→30 % noise')
parser.add_argument('--noise_seed', type=int, default=0)
parser.add_argument('--subsample_ratio', type=float, default=1)
parser.add_argument('--recon_epochs', type=str, default='1', help='comma-separated epochs to run reconstruction, e.g. 10,50,100')
parser.add_argument('--debugging', type=int, default=1, help='Just for checking if the code is right')
args = parser.parse_args()

# ------------------------------------------------------------
# 1. Hyperparameters + Noise·Subsample settings
# ------------------------------------------------------------
batch_size    = args.batch_size
seed          = args.seed
learning_rate = args.learning_rate
epochs        = args.epochs
device        = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
optimizer_type= args.optimizer
dataset = args.dataset

# ---------- label-noise ----------
noise_ratio = args.noise_ratio       
# ---------- subsample ----------
subsample_ratio = args.subsample_ratio     # 1.0 → use all, 0.3→30 %
# ------------------------------------------------------------

if args.recon_epochs.strip() == '':
    recon_epochs = []
else:
    recon_epochs = list(map(int, args.recon_epochs.strip().split(',')))

base_dir = (f"runs/alex/{dataset}/sd{seed}_bh{batch_size}_ot{optimizer_type}"
            f"_lr{learning_rate}_ep{epochs}"
            f"_nr{noise_ratio}_sr{subsample_ratio}")
if args.debugging ==1:
    base_dir = "debugging"

if os.path.exists(base_dir): 
    shutil.rmtree(base_dir)
os.makedirs(base_dir)
eigenvector_dir=os.path.join(base_dir,'eigenvectors')
os.makedirs(eigenvector_dir)
bias_dir       =os.path.join(base_dir,'biases')
os.makedirs(bias_dir)
spectrum_dir   =os.path.join(base_dir,'spectrum')
os.makedirs(spectrum_dir)

num_input_channel, num_classes, trainloader, testloader = get_dataloader(dataset, batch_size, noise_ratio, subsample_ratio, seed)

torch.manual_seed(seed)
model = miniAlexNet(num_classes, num_input_channel).to(device)
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
            c += (model(x).argmax(1)==y).sum().item()
            t += y.size(0)
    return c/t


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
        optimizer.zero_grad()
        loss = criterion(model(x),y)
        loss.backward()
        optimizer.step()

    train_acc.append(accuracy(model,trainloader))
    test_acc .append(accuracy(model,testloader))

    # ----- Spectrum -----
    for idx,(name,fc_idx) in enumerate(fc_indices.items(), start=1):
        W=model.classifier[fc_idx].weight.detach().cpu().numpy()
        U,S,_ = np.linalg.svd(W,full_matrices=False)
        n,p   = W.shape
        spec  = (S**2)/n
        
        sigma2 = np.var(W)
        gamma = p/n
        lambda_plus = sigma2 * (1+ np.sqrt(gamma))**2
        lambda_minus = sigma2 * (1 - np.sqrt(gamma))**2
        
        x_mp = np.linspace(lambda_minus, lambda_plus, 1000)
        x_mp = x_mp[x_mp>0]
        pdf_mp = (1/(2*np.pi * sigma2*gamma*x_mp)) * np.sqrt((lambda_plus-x_mp)* (x_mp-lambda_minus))
        
        mp_upper_bound = lambda_plus
        spikes = spec[spec> mp_upper_bound]
        bulk = spec[spec <= mp_upper_bound]
        spike_cnt = len(spikes)
        nu2_est = bulk.sum()/ max(len(bulk),1)

        plt.figure(figsize=(6,4))
        plt.hist(spec,bins=300,density=True,alpha=.6, label='Empirical')
        plt.plot(x_mp, pdf_mp, color='r', linewidth=2, label='MP Distribution')
        plt.text(0.05,0.95,
                 f"Spikes: {spike_cnt}\nν²: {nu2_est:.4f}",
                 transform=plt.gca().transAxes,
                 va='top',ha='left',
                 bbox=dict(boxstyle='round',facecolor='w',alpha=.8))
        plt.title(f"FC{idx} Spectrum Ep{ep}")
        plt.legend()
        plt.savefig(f"{spectrum_dir}/fc{idx}_spectrum_epoch{ep}.png")
        plt.close()

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
        model.eval()
        
        original_weights = {i: model.classifier[i].weight.data.clone() for i in fc_indices.values()}
        original_biases = {i: model.classifier[i].bias.data.clone() for i in fc_indices.values()}

        for name, recon_idx in fc_indices.items():
            print(f"\n[Recon Parallel] Ep{ep}, Layer: {name}")

            # 1. 모델 분리 (기존과 동일)
            head_layers, tail_layers = [], []
            for i, layer in enumerate(model.classifier):
                if i < recon_idx: head_layers.append(layer)
                elif i > recon_idx: tail_layers.append(layer)
            
            model_head = nn.Sequential(*head_layers).to(device)
            model_tail = nn.Sequential(*tail_layers).to(device)

            # 2. SVD 및 모든 재구성 가중치 스택 생성 (기존과 동일)
            W_orig = original_weights[recon_idx].to(device)
            U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
            r_max = max_rank[name]
            U, S, Vh = U[:,:r_max], S[:r_max], Vh[:r_max,:]

            # ------------------------------------------------------------
            # [수정된 부분] tqdm으로 재구성 행렬 생성 추적
            # ------------------------------------------------------------
            method_definitions = {
                'raw': lambda: generate_topk_recons_batched(U, S, Vh),
                'vec': lambda: generate_js_recons_batched(U, S, Vh, full_mode=False),
                'full': lambda: generate_js_recons_batched(U, S, Vh, full_mode=True),
                'travg': lambda: generate_travg_recons_batched(W_orig, U, S, Vh),
                'diag': lambda: generate_diag_recons_batched(W_orig, U, S, Vh),
                'srk': lambda: generate_shift_rank_k_recons_batched(W_orig, U, S, Vh)
            }
            
            method_stacks = {}
            # tqdm으로 각 메서드의 가중치 스택 생성 과정을 감쌉니다.
            pbar_generate = tqdm(method_definitions.items(), desc="  Generating Stacks", total=len(method_definitions), leave=False)
            with torch.no_grad():
                for method_name, func in pbar_generate:
                    pbar_generate.set_postfix_str(f"Method: {method_name}")
                    method_stacks[method_name] = func()            
            bias = original_biases[recon_idx]

            # ------------------------------------------------------------
            # [수정된 부분 1] head_output 캐싱
            # ------------------------------------------------------------
            # 가장 오래 걸리는 head 부분을 미리 한 번만 계산하여 저장합니다.
            head_outputs_list = []
            labels_list = []
            print("  - Caching head outputs...")
            with torch.no_grad():
                for x, y in tqdm(testloader, desc="  Caching", leave=False):
                    x, y = x.to(device), y.to(device)
                    features = model.features(x)
                    head_output = model_head(torch.flatten(features, 1))
                    head_outputs_list.append(head_output)
                    labels_list.append(y)
            
            # ------------------------------------------------------------
            # [수정된 부분 2] tqdm으로 메서드 루프 실행
            # ------------------------------------------------------------
            # 캐싱된 결과를 사용해 각 메서드의 정확도를 빠르게 계산합니다.
            results = defaultdict(lambda: torch.zeros(r_max, device=device))
            total_samples = 0

            # tqdm으로 재구성 메서드들을 감싸서 진행률을 표시합니다.
            pbar_methods = tqdm(method_stacks.items(), desc=f"  Evaluating {name}", total=len(method_stacks))
            for method_name, W_stack in pbar_methods:
                pbar_methods.set_postfix_str(f"Method: {method_name}")
                
                with torch.no_grad():
                    # 캐싱된 데이터를 순회하며 정확도 계산
                    for head_output, labels in zip(head_outputs_list, labels_list):
                        recon_output = torch.einsum('bi,koi->kbo', head_output, W_stack) + bias
                        
                        k_dim, b_dim, o_dim = recon_output.shape
                        recon_output_flat = recon_output.reshape(k_dim * b_dim, o_dim)
                        tail_output_flat = model_tail(recon_output_flat)
                        tail_output = tail_output_flat.reshape(k_dim, b_dim, -1)
                        
                        preds = tail_output.argmax(dim=-1)
                        results[method_name] += (preds == labels.unsqueeze(0)).sum(dim=1)
                
                if total_samples == 0:
                    total_samples = sum(len(y) for y in labels_list)
            
            # ------------------------------------------------------------
            # [수정된 부분 3] 결과 저장 로직
            # ------------------------------------------------------------
            # 계산된 결과를 파일로 저장합니다.
            for method_name, correct_tensor in results.items():
                acc_array = (correct_tensor / total_samples).cpu().numpy()
                file_tag = {
                    'raw': 'acc', 'vec': 'vec', 'full': 'full',
                    'travg': 'travg', 'diag': 'diag', 'srk': 'srk'
                }[method_name]
                
                np.savetxt(f"{base_dir}/recon_{name}_{file_tag}_epoch{ep}.csv", acc_array)
                np.savetxt(f"{base_dir}/recon_{name}_{file_tag}_norm_epoch{ep}.csv", acc_array / acc_array[-1])

# ------------------------------------------------------------
# 7-1. save final model
# ------------------------------------------------------------
print("\nTraining finished. Saving the final model...")
# 저장 경로 설정 (결과가 저장되는 base_dir 안에 저장됩니다)
model_save_path = os.path.join(base_dir, 'final_model.pth')

# 모델의 state_dict 저장
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ------------------------------------------------------------
# 8. Overlay Graphs (그대로)
# ------------------------------------------------------------
for name in fc_indices.keys():
    fig,ax=plt.subplots(figsize=(10,6))
    for ep in recon_epochs:
        for tag,ls in [("acc_norm","-"), ("vec_norm","--"), ("full_norm",":"), 
               ("travg_norm", "-."), ("diag_norm", (0, (3, 1, 1, 1))),
               ("snr_norm", "m-"), 
               ("srk_norm", "c--")]:
            f=f"{base_dir}/recon_{name}_{tag}_epoch{ep}.csv"
            if os.path.exists(f):
                y=np.loadtxt(f)
                ax.plot(range(1,len(y)+1),y,ls,label=f"Ep{ep}-{tag.replace('_norm','')}")
    ax.set_xlabel("Rank k"); ax.set_ylabel("Normalized Test Acc")
    ax.set_title(f"Normalized Reconstruction Accuracy vs Rank ({name})")
    ax.legend(); ax.grid(True)
    plt.savefig(f"{base_dir}/recon_{name}_acc_norm_overlay.png"); plt.close()

    fig,ax=plt.subplots(figsize=(10,6))
    for ep in recon_epochs:
        for tag,ls in [("acc","-"), ("vec","--"), ("full",":"),
               ("travg", "-."), ("diag", (0, (3, 1, 1, 1))),
               ("snr", "m-"),
               ("srk", "c--")]:
            f=f"{base_dir}/recon_{name}_{tag}_epoch{ep}.csv"
            if os.path.exists(f):
                y=np.loadtxt(f); ax.plot(range(1,len(y)+1),y,ls,label=f"Ep{ep}-{tag}")
    ax.set_xlabel("Rank k"); ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Reconstruction Accuracy vs Rank ({name})")
    ax.legend(); ax.grid(True)
    plt.savefig(f"{base_dir}/recon_{name}_acc_overlay.png")
    plt.close()

# ------------------------------------------------------------
# 9. Accuracy curve 저장
# ------------------------------------------------------------
np.savetxt(f"{base_dir}/train_accuracy.csv",np.array(train_acc))
np.savetxt(f"{base_dir}/test_accuracy.csv" ,np.array(test_acc))
plt.figure()
plt.plot(train_acc,label='train')
plt.plot(test_acc,label='test')
plt.legend()
plt.title("Accuracy")
plt.savefig(f"{base_dir}/train_test_accuracy.png")
plt.close()
