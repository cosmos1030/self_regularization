#!/usr/bin/env python
# ------------------------------------------------------------
# [논문 방법론 적용] 가중치 행렬 필터링 평가 스크립트
# - 방법 1: 특잇값 제거 (Removing)
# - 방법 2: 특잇값 이동 및 제거 (Shifting + Removing)
# ------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import argparse
from scipy.optimize import curve_fit

# 훈련 스크립트와 동일한 dataloader.py가 필요합니다.
from dataloader import get_dataloader

# ------------------------------------------------------------
# 1. 모델 클래스 정의 (기존과 동일)
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
# 2. 논문 기반 필터링 헬퍼 함수
# ------------------------------------------------------------

def fit_marchenko_pastur(S, Q, num_bins=50):
    """
    특잇값(S)에 Marchenko-Pastur 분포를 피팅하여 노이즈 표준편차(sigma)와
    벌크 상단 경계(v_max)를 추정합니다. [cite: 46, 182]
    """
    S_np = S.detach().cpu().numpy()
    
    # MP 확률 밀도 함수 (PDF)
    def mp_pdf(v, sigma, C):
        v_min = sigma * (1 - np.sqrt(Q))**2
        v_max = sigma * (1 + np.sqrt(Q))**2
        pdf = np.zeros_like(v)
        mask = (v >= v_min) & (v <= v_max)
        pdf[mask] = (C / (2 * np.pi * sigma * v[mask])) * np.sqrt((v_max - v[mask]) * (v[mask] - v_min))
        return pdf

    # 특잇값 스펙트럼의 히스토그램 생성 (논문의 가우시안 브로드닝 근사)
    hist, bin_edges = np.histogram(S_np, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        # 곡선 피팅으로 sigma 추정
        popt, _ = curve_fit(mp_pdf, bin_centers, hist, p0=[np.median(S_np), 1.0])
        sigma_fit = popt[0]
    except RuntimeError:
        # 피팅 실패 시, 이론적인 값으로 대체
        sigma_fit = np.median(S_np) / (1 - Q) if Q < 1 else np.median(S_np)

    v_max_mp = sigma_fit * (1 + np.sqrt(Q))**2
    return sigma_fit, v_max_mp

def shift_singular_values(S, sigma, q):
    """
    논문의 식 (3)에 따라 큰 특잇값들을 이동(shift)시킵니다. [cite: 80]
    """
    S_scaled = S / sigma
    S_shifted_scaled_sq = (S_scaled**2 - q - 1 + torch.sqrt(torch.clamp((S_scaled**2 - q - 1)**2 - 4*q, min=1e-8))) / 2
    S_shifted = torch.sqrt(torch.clamp(S_shifted_scaled_sq, min=0)) * sigma
    return S_shifted

def reconstruct_matrix(U, S, Vh, num_to_keep):
    """
    주어진 SVD 구성요소와 유지할 특잇값의 수(num_to_keep)를 사용하여 행렬을 재구성합니다.
    가장 큰 특잇값부터 num_to_keep개 만큼 유지합니다.
    """
    if num_to_keep == 0:
        return torch.zeros_like(U @ torch.diag(S) @ Vh)
    
    # 가장 큰 특잇값부터 순서대로 유지
    U_k = U[:, :num_to_keep]
    S_k = S[:num_to_keep]
    Vh_k = Vh[:num_to_keep, :]
    
    return U_k @ torch.diag(S_k) @ Vh_k

# ------------------------------------------------------------
# 3. 메인 실행 블록
# ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="논문 기반 가중치 필터링 평가 스크립트")
    parser.add_argument('--model_path', type=str, required=True, help='평가할 .pth 모델 파일의 경로')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'MNIST'], required=True, help='모델이 훈련된 데이터셋')
    parser.add_argument('--output_dir', type=str, default='paper_filtering_results', help='결과(그래프, CSV)를 저장할 디렉토리')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=101, help='제거 비율을 테스트할 단계 수 (0% to 100%)')
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

    # --- 원본 모델 정확도 측정 ---
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    original_accuracy = correct / total
    print(f"Original Model Test Accuracy: {original_accuracy:.4f}")

    # --- 필터링 및 평가 ---
    fc_indices={"fc1":0,"fc2":2,"fc3":4}
    
    for name, recon_idx in fc_indices.items():
        print(f"\n[Filtering Evaluation] Layer: {name}")

        W_orig = model.classifier[recon_idx].weight.data.clone()
        U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
        r_max = S.shape[0]
        m, n = W_orig.shape
        q = min(m, n) / max(m, n)

        # 1. Marchenko-Pastur 피팅 및 특잇값 이동
        sigma_mp, v_max_mp = fit_marchenko_pastur(S, q)
        print(f"  - MP fit: sigma={sigma_mp:.4f}, v_max_threshold={v_max_mp:.4f}")
        
        S_shifted = S.clone()
        shift_mask = S > v_max_mp
        num_shifted = shift_mask.sum().item()
        if num_shifted > 0:
            S_shifted[shift_mask] = shift_singular_values(S[shift_mask], sigma_mp, q)
            print(f"  - Shifted {num_shifted} singular values larger than v_max.")
        else:
            print("  - No singular values were large enough to be shifted.")

        # 2. 제거 비율에 따른 정확도 평가
        accuracies_removed_only = []
        accuracies_shifted_removed = []
        removal_percentages = np.linspace(0, 100, args.num_steps)

        pbar = tqdm(removal_percentages, desc=f"Evaluating {name}")
        for percent_removed in pbar:
            num_to_remove = int(r_max * percent_removed / 100)
            num_to_keep = r_max - num_to_remove

            # 방법 1: 제거만 적용
            W_removed = reconstruct_matrix(U, S, Vh, num_to_keep)
            # 방법 2: 이동 후 제거 적용
            W_shifted = reconstruct_matrix(U, S_shifted, Vh, num_to_keep)

            # 정확도 계산
            correct_removed, correct_shifted, total = 0, 0, 0
            with torch.no_grad():
                model.classifier[recon_idx].weight.data = W_removed
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_removed += (predicted == y).sum().item()
                
                model.classifier[recon_idx].weight.data = W_shifted
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_shifted += (predicted == y).sum().item()
                    total += y.size(0)

            accuracies_removed_only.append(correct_removed / total)
            accuracies_shifted_removed.append(correct_shifted / total)
        
        # 가중치 원상 복구
        model.classifier[recon_idx].weight.data = W_orig

        # 결과 CSV로 저장
        results_df = np.array([removal_percentages, accuracies_removed_only, accuracies_shifted_removed]).T
        np.savetxt(f"{args.output_dir}/{name}_filtering_acc.csv", results_df, 
                   header="percent_removed,acc_removed_only,acc_shifted_removed", delimiter=',', comments='')

        # 그래프 생성
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(removal_percentages, accuracies_removed_only, 'b-', label='Removal Only')
        ax.plot(removal_percentages, accuracies_shifted_removed, 'g-', label='Shift + Removal')
        ax.axhline(y=original_accuracy, color='r', linestyle='--', label=f'Original Accuracy ({original_accuracy:.3f})')
        
        ax.set_xlabel("Percentage of Singular Values Removed (%)", fontsize=12)
        ax.set_ylabel("Test Accuracy", fontsize=12)
        ax.set_title(f"Test Accuracy vs. SVs Removed ({name})", fontsize=14)
        ax.legend()
        plt.savefig(f"{args.output_dir}/{name}_filtering_comparison.png")
        plt.close()

    print(f"\nAll tasks finished. Results are in '{args.output_dir}'.")