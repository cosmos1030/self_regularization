#!/usr/bin/env python
# ------------------------------------------------------------
# [최종 통합본] 독립적인 모델 재구성 및 평가 스크립트
# - Top-k, JS-vec, JS-full, L+D Regularization(Method 2) 4가지 방법 모두 평가
# ------------------------------------------------------------
import torch, torchvision, torchvision.transforms as T
import torch.nn as nn
import numpy as np, matplotlib.pyplot as plt, os, shutil
from tqdm import tqdm
import argparse

# 훈련 스크립트와 동일한 dataloader.py가 필요합니다.
from dataloader import get_dataloader

# ------------------------------------------------------------
# 1. 모델 클래스 정의
# ------------------------------------------------------------
class miniAlexNet(nn.Module):
    def __init__(self, num_classes, num_input_channel):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channel,96,5,3,2),
            nn.ReLU(),
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(96,256,5,3,2),
            nn.ReLU(),
            nn.MaxPool2d(3,1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4,384),
            nn.ReLU(),
            nn.Linear(384,192),
            nn.ReLU(),
            nn.Linear(192,num_classes))
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)
        return self.classifier(x)

# ------------------------------------------------------------
# 2. SVD·JS 및 정규화 헬퍼 함수
# ------------------------------------------------------------
def generate_topk_recons_batched(U, S, Vh):
    """모든 k (1 to r_max)에 대한 top-k 재구성 행렬들을 일괄 생성"""
    r_max = S.shape[0]
    recons = []
    for k in range(1, r_max + 1):
        W_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        recons.append(W_k)
    return torch.stack(recons)

def generate_js_recons_batched(U, S, Vh, full_mode):
    """모든 k (1 to r_max)에 대한 JS-vec 또는 JS-full 재구성 행렬들을 일괄 생성"""
    r_max = S.shape[0]
    recons = []
    n = torch.tensor(U.shape[0], dtype=S.dtype, device=S.device)
    sqrt_n = torch.sqrt(n)

    for k in range(1, r_max + 1):
        if k == r_max:
            W_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            recons.append(W_k)
            continue

        U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
        H = U_k * (S_k / sqrt_n).reshape(1, -1)

        p = H.shape[0]
        e = torch.ones(p, 1, dtype=S.dtype, device=S.device)
        M = (e @ (e.T @ H)) / p
        R = H - M

        if R.numel() - k <= 0:
            nu2 = 0
        else:
            nu2 = (R ** 2).sum() / (R.numel() - k)

        try:
            C = torch.eye(k, dtype=S.dtype, device=S.device) - nu2 * torch.linalg.inv(R.T @ R)
        except torch.linalg.LinAlgError:
            C = torch.eye(k, dtype=S.dtype, device=S.device)

        Hj = H @ C + M @ (torch.eye(k, dtype=S.dtype, device=S.device) - C)
        Uj, _, _ = torch.linalg.svd(Hj, full_matrices=False)

        if full_mode:
            sigma_js = torch.sqrt(torch.clamp(S_k ** 2 - n * nu2, min=0))
            W_k = Uj[:, :k] @ torch.diag(sigma_js) @ Vh_k
        else: # vec_mode
            W_k = Uj[:, :k] @ torch.diag(S_k) @ Vh_k
        recons.append(W_k)

    return torch.stack(recons)

def get_regularized_eigenvectors(C, W_for_mp):
    """L+D 모델을 기반으로 정규화된 고유벡터를 계산하는 내부 헬퍼 함수"""
    device = C.device
    p, n = W_for_mp.shape
    W_np = W_for_mp.detach().cpu().numpy()

    eigvals, eigvecs = torch.linalg.eigh(C)
    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    mp_plus = (np.var(W_np) * (1 + np.sqrt(n / p))**2) if p >= n else \
              (np.var(W_np) * (1 + np.sqrt(p / n))**2)
    mp_threshold = mp_plus * n 
    
    k_spike = (eigvals > mp_threshold).sum().item()

    if k_spike == 0:
        return eigvecs

    spiked_eigvals = eigvals[:k_spike]
    spiked_eigvecs = eigvecs[:, :k_spike]
    bulk_eigvals = eigvals[k_spike:]

    L = spiked_eigvecs @ torch.diag(spiked_eigvals) @ spiked_eigvecs.T
    nu2 = torch.mean(bulk_eigvals)
    D = nu2 * torch.eye(C.shape[0], device=device)

    C_reg = L + D
    
    reg_eigvals, reg_eigvecs = torch.linalg.eigh(C_reg)
    sorted_indices_reg = torch.argsort(reg_eigvals, descending=True)
    reg_eigvecs = reg_eigvecs[:, sorted_indices_reg]
    
    return reg_eigvecs

def generate_reg_method2_batched(U, S, Vh):
    """[최적화 및 버그 수정 버전] L+D 정규화 방법으로 모든 k에 대한 재구성 행렬을 일괄 생성"""
    device = U.device
    r_max = S.shape[0]
    
    W = U @ torch.diag(S) @ Vh
    
    V_new = get_regularized_eigenvectors(W @ W.T, W)
    U_new = get_regularized_eigenvectors(W.T @ W, W.T)
    
    # U_new가 정제된 V (right singular vectors) 이므로, Vh_new는 U_new.T 입니다.
    Vh_new = U_new.T
    
    recons = []
    for k in range(1, r_max + 1):
        # V_new가 정제된 U (left singular vectors) 입니다.
        U_k_new = V_new[:, :k]
        S_k = torch.diag(S[:k])
        
        # Vh_new에서 k개의 행을 슬라이싱합니다.
        Vh_k_new = Vh_new[:k, :]
        
        # 수정된 부분: U_new[:k, :].T  ->  U_new.T[:k, :] 와 동일한 로직
        W_k = U_k_new @ S_k @ Vh_k_new
        recons.append(W_k)
        
    return torch.stack(recons)


# ------------------------------------------------------------
# 3. 메인 실행 블록
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="독립적인 모델 재구성 평가 스크립트 (4가지 방법 통합)")
    parser.add_argument('--model_path', type=str, required=True, help='평가할 .pth 모델 파일의 경로')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'MNIST'], required=True, help='모델이 훈련된 데이터셋')
    parser.add_argument('--output_dir', type=str, default='recon_results_all', help='결과(그래프, CSV)를 저장할 디렉토리')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # --- 설정 ---
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if os.path.exists(args.output_dir):
        print(f"Warning: Output directory '{args.output_dir}' already exists. Overwriting...")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    print("="*50)
    print(f"모델 로드: {args.model_path}")
    print(f"데이터셋: {args.dataset}")
    print(f"결과 저장 위치: {args.output_dir}")
    print("="*50)

    # --- 데이터 로더 준비 ---
    num_input_channel, num_classes, _, testloader = get_dataloader(
        args.dataset, args.batch_size, noise_ratio=0, subsample_ratio=1, seed=0
    )

    # --- 모델 로드 ---
    model = miniAlexNet(num_classes, num_input_channel).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- 재구성 및 평가 ---
    fc_indices={"fc1":0,"fc2":2,"fc3":4}
    max_rank  ={n:min(model.classifier[i].weight.shape) for n,i in fc_indices.items()}

    original_weights = {i: model.classifier[i].weight.data.clone() for i in fc_indices.values()}
    original_biases = {i: model.classifier[i].bias.data.clone() for i in fc_indices.values()}

    for name, recon_idx in fc_indices.items():
        print(f"\n[Recon Parallel] Layer: {name}")

        head_layers = [layer for i, layer in enumerate(model.classifier) if i < recon_idx]
        tail_layers = [layer for i, layer in enumerate(model.classifier) if i > recon_idx]
        model_head = nn.Sequential(*head_layers).to(device)
        model_tail = nn.Sequential(*tail_layers).to(device)

        W_orig = original_weights[recon_idx].to(device)
        U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)

        r_max = max_rank[name]
        U, S, Vh = U[:,:r_max], S[:r_max], Vh[:r_max,:]

        with torch.no_grad():
            print("Generating reconstructed matrices for all methods...")
            W_raw_stack = generate_topk_recons_batched(U, S, Vh)
            W_vec_stack = generate_js_recons_batched(U, S, Vh, full_mode=False)
            W_full_stack = generate_js_recons_batched(U, S, Vh, full_mode=True)
            W_reg_stack = generate_reg_method2_batched(U, S, Vh)

        bias = original_biases[recon_idx]

        correct_raw = torch.zeros(r_max, device=device)
        correct_vec = torch.zeros(r_max, device=device)
        correct_full = torch.zeros(r_max, device=device)
        correct_reg = torch.zeros(r_max, device=device)
        total = 0

        with torch.no_grad():
            for x, y in tqdm(testloader, desc=f"Parallel Eval {name}", leave=False):
                x, y = x.to(device), y.to(device)
                features = model.features(x)
                head_output = model_head(torch.flatten(features, 1))

                def get_batch_acc(W_stack):
                    recon_output = torch.einsum('bi,koi->kbo', head_output, W_stack) + bias
                    k_dim, b_dim, o_dim = recon_output.shape
                    tail_output_flat = model_tail(recon_output.reshape(k_dim * b_dim, o_dim))
                    tail_output = tail_output_flat.reshape(k_dim, b_dim, -1)
                    preds = tail_output.argmax(dim=-1)
                    return (preds == y.unsqueeze(0)).sum(dim=1)

                correct_raw  += get_batch_acc(W_raw_stack)
                correct_vec  += get_batch_acc(W_vec_stack)
                correct_full += get_batch_acc(W_full_stack)
                correct_reg  += get_batch_acc(W_reg_stack)
                total += y.size(0)

        raw = (correct_raw / total).cpu().numpy()
        vec = (correct_vec / total).cpu().numpy()
        ful = (correct_full / total).cpu().numpy()
        reg = (correct_reg / total).cpu().numpy()

        # 결과 CSV 파일로 저장
        np.savetxt(f"{args.output_dir}/recon_{name}_acc.csv", raw)
        np.savetxt(f"{args.output_dir}/recon_{name}_vec.csv", vec)
        np.savetxt(f"{args.output_dir}/recon_{name}_full.csv", ful)
        np.savetxt(f"{args.output_dir}/recon_{name}_reg.csv", reg)
        # 정규화된 결과 저장
        np.savetxt(f"{args.output_dir}/recon_{name}_acc_norm.csv" ,np.array(raw)/raw[-1])
        np.savetxt(f"{args.output_dir}/recon_{name}_vec_norm.csv" ,np.array(vec)/vec[-1])
        np.savetxt(f"{args.output_dir}/recon_{name}_full_norm.csv",np.array(ful)/ful[-1])
        np.savetxt(f"{args.output_dir}/recon_{name}_reg_norm.csv", np.array(reg)/reg[-1])

    # --- 그래프 생성 ---
    print("\nGenerating overlay graphs...")
    for name in fc_indices.keys():
        # 정규화된 정확도 그래프
        fig,ax=plt.subplots(figsize=(12,7))
        plot_data = [
            ("acc_norm", "-", "Top-k"),
            ("vec_norm", "--", "JS-vec"),
            ("full_norm", ":", "JS-full"),
            ("reg_norm", "-.", "L+D Reg")
        ]
        for tag, ls, label_name in plot_data:
            f = f"{args.output_dir}/recon_{name}_{tag}.csv"
            if os.path.exists(f):
                y = np.loadtxt(f)
                ax.plot(range(1,len(y)+1), y, ls=ls, label=label_name)
        ax.set_xlabel("Rank k"); ax.set_ylabel("Normalized Test Accuracy")
        ax.set_title(f"Normalized Reconstruction Accuracy vs Rank ({name})")
        ax.legend(); ax.grid(True)
        plt.savefig(f"{args.output_dir}/recon_{name}_acc_norm_overlay.png"); plt.close()

        # 원본 정확도 그래프
        fig,ax=plt.subplots(figsize=(12,7))
        plot_data_abs = [
            ("acc", "-", "Top-k"),
            ("vec", "--", "JS-vec"),
            ("full", ":", "JS-full"),
            ("reg", "-.", "L+D Reg")
        ]
        for tag, ls, label_name in plot_data_abs:
            f = f"{args.output_dir}/recon_{name}_{tag}.csv"
            if os.path.exists(f):
                y = np.loadtxt(f)
                ax.plot(range(1,len(y)+1), y, ls=ls, label=label_name)
        ax.set_xlabel("Rank k"); ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Reconstruction Accuracy vs Rank ({name})")
        ax.legend(); ax.grid(True)
        plt.savefig(f"{args.output_dir}/recon_{name}_acc_overlay.png")
        plt.close()

    print(f"\nAll tasks finished. Results are in '{args.output_dir}'.")