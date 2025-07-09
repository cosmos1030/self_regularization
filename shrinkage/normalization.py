import torch

# ------------------------------------------------------------
# 5. SVD·JS helper (top-k, JS-vec, JS-full)
# ------------------------------------------------------------
# def topk_recon(W_cpu,k):
#     W=W_cpu.to(device)
#     U,S,Vh=torch.linalg.svd(W,full_matrices=False)
#     return U[:,:k] @ torch.diag(S[:k]) @ Vh[:k]

def topk_recon(U, S, Vh, k):
    return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]

def js_vec(U, S, Vh, k):
    U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
    dtype, device = S_k.dtype, S_k.device
    sqrt_n = torch.sqrt(torch.tensor(U.shape[0], dtype=dtype, device=device))
    H = U_k * (S_k / sqrt_n).reshape(1, -1)

    p = H.shape[0]
    e = torch.ones(p, 1, dtype=dtype, device=device)
    M = (e @ (e.T @ H)) / p
    R = H - M
    nu2 = (R ** 2).sum() / (R.numel() - k)

    C = torch.eye(k, dtype=dtype, device=device) - nu2 * torch.linalg.inv(R.T @ R)
    Hj = H @ C + M @ (torch.eye(k, dtype=dtype, device=device) - C)

    Uj, _, _ = torch.linalg.svd(Hj, full_matrices=False)
    return Uj[:, :k] @ torch.diag(S_k) @ Vh_k


def js_full(U, S, Vh, k):
    U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
    dtype, device = S_k.dtype, S_k.device
    n = torch.tensor(U.shape[0], dtype=dtype, device=device)
    sqrt_n = torch.sqrt(n)
    H = U_k * (S_k / sqrt_n).reshape(1, -1)

    p = H.shape[0]
    e = torch.ones(p, 1, dtype=dtype, device=device)
    M = (e @ (e.T @ H)) / p
    R = H - M
    nu2 = (R ** 2).sum() / (R.numel() - k)

    C = torch.eye(k, dtype=dtype, device=device) - nu2 * torch.linalg.inv(R.T @ R)
    Hj = H @ C + M @ (torch.eye(k, dtype=dtype, device=device) - C)

    sigma_js = torch.sqrt(torch.clamp(S_k ** 2 - n * nu2, min=0))
    Uj, _, _ = torch.linalg.svd(Hj, full_matrices=False)
    return Uj[:, :k] @ torch.diag(sigma_js) @ Vh_k

def travg_recon(W, U, S, Vh, k):
    """Trace-Averaging Regularization 함수"""
    n, p = W.shape
    m = min(n, p) # 최대 랭크
    
    # 1. 새로운 왼쪽 특이벡터(U_reg) 계산
    # Spike (상위 k개 정보) + Averaged Noise (나머지 정보의 평균 에너지)
    U_spike_cov = U[:, :k] @ torch.diag(S[:k]**2) @ U[:, :k].T
    noise_cov_trace = torch.trace(W @ W.T - U_spike_cov)
    U_reg_cov = U_spike_cov + (noise_cov_trace / n) * torch.eye(n, device=W.device)
    U_reg, _, _ = torch.linalg.svd(U_reg_cov)

    # 2. 새로운 오른쪽 특이벡터(Vh_reg) 계산
    Vh_spike_cov = Vh[:k, :].T @ torch.diag(S[:k]**2) @ Vh[:k, :]
    noise_cov_trace_p = torch.trace(W.T @ W - Vh_spike_cov)
    Vh_reg_cov = Vh_spike_cov + (noise_cov_trace_p / p) * torch.eye(p, device=W.device)
    _, _, Vh_reg = torch.linalg.svd(Vh_reg_cov)
    
    # 3. 새로운 특이벡터들과 원본 특이값으로 가중치 재구성
    # 원본 코드에서는 전체 랭크(m)를 사용했으므로 동일하게 구현
    return U_reg[:, :m] @ torch.diag(S[:m]) @ Vh_reg[:m, :]

def diag_recon(W, U, S, Vh, k):
    """Diagonal Regularization 함수"""
    n, p = W.shape
    m = min(n, p) # 최대 랭크

    # 1. 새로운 왼쪽 특이벡터(U_reg) 계산
    # Spike (상위 k개 정보) + Diagonal Noise (나머지 정보의 대각성분)
    U_spike_cov = U[:, :k] @ torch.diag(S[:k]**2) @ U[:, :k].T
    U_noise_cov_diag = torch.diag(torch.diag(W @ W.T - U_spike_cov))
    U_reg_cov = U_spike_cov + U_noise_cov_diag
    U_reg, _, _ = torch.linalg.svd(U_reg_cov)
    
    # 2. 새로운 오른쪽 특이벡터(Vh_reg) 계산
    Vh_spike_cov = Vh[:k, :].T @ torch.diag(S[:k]**2) @ Vh[:k, :]
    Vh_noise_cov_diag = torch.diag(torch.diag(W.T @ W - Vh_spike_cov))
    Vh_reg_cov = Vh_spike_cov + Vh_noise_cov_diag
    _, _, Vh_reg = torch.linalg.svd(Vh_reg_cov)

    # 3. 새로운 특이벡터들과 원본 특이값으로 가중치 재구성
    return U_reg[:, :m] @ torch.diag(S[:m]) @ Vh_reg[:m, :]

# 위치: generate_shift_remove_recons_batched 함수 바로 아래

def shift_singular_values(S, sigma, q):
    """
    논문의 식 (3)에 따라 큰 특잇값들을 이동(shift)시킵니다.
    """
    if sigma < 1e-6: return S # sigma가 0에 가까우면 이동시키지 않음
    S_scaled = S / sigma
    S_shifted_scaled_sq = (S_scaled**2 - q - 1 + torch.sqrt(torch.clamp((S_scaled**2 - q - 1)**2 - 4*q, min=1e-8))) / 2
    S_shifted = torch.sqrt(torch.clamp(S_shifted_scaled_sq, min=0)) * sigma
    return S_shifted



# ------------------------------------------------------------
# 5. SVD·JS helper ... (기존 함수들 아래에 추가)
# ------------------------------------------------------------

def generate_topk_recons_batched(U, S, Vh):
    """모든 k (1 to r_max)에 대한 top-k 재구성 행렬들을 일괄 생성"""
    r_max = S.shape[0]
    # W_k = sum_{i=1 to k} s_i * u_i @ vh_i.T
    # 이는 U와 Vh의 외적(outer product)의 누적 합으로 계산 가능
    s_diag = torch.diag(S)
    recons = torch.cumsum(U @ s_diag @ Vh, dim=0) # 이 부분은 잘못된 계산입니다.
    
    # 올바른 계산: 각 rank-1 행렬을 만들고 누적합
    # u_i * s_i @ vh_i
    rank_1_matrices = U * S.unsqueeze(0) @ Vh
    # 위 계산은 차원이 맞지 않습니다. 아래처럼 수정합니다.
    # U: [out, r], S: [r], Vh: [r, in]
    # (U * S)는 [out, r]
    # (U * S) @ Vh 는 [out, in] 행렬 하나만 나옵니다.
    
    # 올바른 방법 2:
    recons = []
    for k in range(1, r_max + 1):
        W_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        recons.append(W_k)
    return torch.stack(recons)

def generate_js_recons_batched(U, S, Vh, full_mode):
    """모든 k (1 to r_max)에 대한 JS-vec 또는 JS-full 재구성 행렬들을 일괄 생성 (수정된 버전)"""
    r_max = S.shape[0]
    recons = []
    n = torch.tensor(U.shape[0], dtype=S.dtype, device=S.device)
    sqrt_n = torch.sqrt(n)

    for k in range(1, r_max + 1):
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # 여기가 핵심 수정 사항입니다.
        # k가 최대 랭크이면, shrinkage 계산 없이 원본 행렬을 복원하고 루프를 계속합니다.
        if k == r_max:
            W_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            recons.append(W_k)
            continue
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

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

# ... generate_js_recons_batched 함수 바로 아래에 추가 ...

def generate_travg_recons_batched(W, U, S, Vh):
    """모든 k에 대해 Trace-Averaging 재구성 행렬들을 일괄 생성"""
    r_max = S.shape[0]
    recons = []
    for k in range(1, r_max + 1):
        # k가 최대 랭크이면 원본 행렬 복원
        if k == r_max:
            W_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        else:
            W_k = travg_recon(W, U, S, Vh, k)
        recons.append(W_k)
    return torch.stack(recons)

def generate_diag_recons_batched(W, U, S, Vh):
    """모든 k에 대해 Diagonal 재구성 행렬들을 일괄 생성"""
    r_max = S.shape[0]
    recons = []
    for k in range(1, r_max + 1):
        # k가 최대 랭크이면 원본 행렬 복원
        if k == r_max:
            W_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        else:
            W_k = diag_recon(W, U, S, Vh, k)
        recons.append(W_k)
    return torch.stack(recons)
# 위치: generate_diag_recons_batched 함수 바로 아래

def generate_shift_rank_k_recons_batched(W, U, S, Vh):
    """
    각 재구성 랭크 k를 신호/노이즈 구분 기준으로 동적으로 사용하여
    'Shift + Removal'을 수행하는, 일관성 있는 재구성 함수
    """
    r_max = S.shape[0]
    m, n = W.shape
    q = min(m, n) / max(m, n)
    recons = []

    # 모든 k (rank)에 대해 루프를 실행
    for k in range(1, r_max + 1):
        # k 자체가 신호/노이즈 구분 기준이 됨
        S_signal = S[:k]
        S_noise = S[k:]

        # 노이즈 부분이 없거나 매우 작으면(1개 이하) sigma 추정이 불가하므로 이동 없음
        if len(S_noise) < 2:
            S_signal_shifted = S_signal
        else:
            # 노이즈 부분에서 sigma 추정
            sigma_sq_est = torch.mean(S_noise**2)
            sigma_est = torch.sqrt(sigma_sq_est)
            
            # 추정된 sigma로 신호 부분(S_signal)을 이동
            S_signal_shifted = shift_singular_values(S_signal, sigma_est, q)

        # 이동된 신호(S_signal_shifted)를 사용해 k-rank 행렬 재구성
        W_k = U[:, :k] @ torch.diag(S_signal_shifted) @ Vh[:k, :]
        recons.append(W_k)
        
    return torch.stack(recons)