import os
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# geoopt & geomstats 관련 모듈
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import Stiefel
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

# ---------------------------
# 모델 및 데이터 설정
# ---------------------------
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

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

# ---------------------------
# 훈련 함수 (시드별)
# ---------------------------
def run_training(seed, epochs=100, batch_size=64, learning_rate=0.0001, optimizer_type="adam"):
    # 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainloader, testloader = get_data_loaders(batch_size)
    model = miniAlexNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == "adam":
        optimizer_model = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer_model = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # base_dir (훈련 결과와 eigenvector 파일이 저장될 경로)
    base_dir = os.path.join("multiseed_runs", f"alex_seed{seed}_batch{batch_size}_{optimizer_type.lower()}_lr{learning_rate}_epochs{epochs}")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    eigen_dir = os.path.join(base_dir, "eigenvectors")
    os.makedirs(eigen_dir, exist_ok=True)
    
    # 훈련 루프 (에폭마다 fc 레이어의 leading eigenvector 저장)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in tqdm(trainloader, desc=f"Seed {seed} Epoch {epoch+1}/{epochs} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_model.step()
        # 에폭 종료 후 – 모델의 classifier 내 각 Linear layer에 대해 leading eigenvector 저장
        layer_index = 1
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().numpy()
                u, s, vh = np.linalg.svd(weight)
                leading_eigenvector = vh[0, :]
                file_path = os.path.join(eigen_dir, f"fc{layer_index}.csv")
                with open(file_path, "a") as f:
                    np.savetxt(f, leading_eigenvector.reshape(1, -1))
                layer_index += 1
    
    # 훈련 후 최종 정확도 측정 (전체 데이터셋에 대해)
    model.eval()
    total_train, correct_train = 0, 0
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
    train_acc = correct_train / total_train

    total_test, correct_test = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_acc = correct_test / total_test

    return train_acc, test_acc, base_dir

# ---------------------------
# GPCA 분석 함수 (피팅 및 RSS 측정)
# ---------------------------
def gpca_analysis(eigen_csv, gpca_dim=1, max_iter=200000, tolerance=1e-5, patience=100, device=torch.device("cpu"), start_epoch=1, end_epoch=None):
    data = np.loadtxt(eigen_csv)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if end_epoch is None:
        end_epoch = data.shape[0]
    data = data[start_epoch-1:end_epoch]
    l, p = data.shape
    space = Hypersphere(dim=p-1)
    data_tensor = torch.from_numpy(data).float().to(device)
    data_aug = data_tensor.clone()

    def sphere_dist(x, y):
        nx = torch.norm(x)
        ny = torch.norm(y)
        inner = torch.dot(x, y) / (nx * ny + 1e-14)
        inner_clamped = torch.clamp(inner, -1.0, 1.0)
        return torch.arccos(inner_clamped)

    for _ in range(3):
        for i in range(l - 1):
            if sphere_dist(data_aug[i+1], data_aug[i]) > sphere_dist(-data_aug[i+1], data_aug[i]):
                data_aug[i+1] = -data_aug[i+1]

    def exp_map(v, p_):
        norm_v = torch.norm(v)
        if norm_v < 1e-14:
            return p_
        return torch.cos(norm_v) * p_ + torch.sin(norm_v) * (v / norm_v)

    def proj_to_geodesic(X, tangent, base):
        norm_t = torch.norm(tangent)
        if norm_t < 1e-14:
            return base.unsqueeze(0).repeat(X.shape[0], 1)
        dot_xt = torch.sum(X * tangent, dim=1)
        dot_xb = torch.sum(X * base, dim=1)
        factor = torch.atan2(dot_xt, dot_xb) / (norm_t + 1e-14)
        cos_f = torch.cos(factor).unsqueeze(1)
        sin_f = torch.sin(factor).unsqueeze(1)
        unit_t = tangent / norm_t
        proj = cos_f * base + sin_f * unit_t
        proj_corrected = []
        for i in range(X.shape[0]):
            if sphere_dist(X[i], proj[i]) <= sphere_dist(X[i], -proj[i]):
                proj_corrected.append(proj[i])
            else:
                proj_corrected.append(-proj[i])
        return torch.stack(proj_corrected)

    def proj_to_2sphere(X, t1, t2, base):
        p1 = base
        p2 = exp_map(t1, base)
        p3 = exp_map(t2, base)
        A = torch.stack([p1, p2, p3], dim=1)
        pinv = torch.linalg.pinv(A.T @ A)
        proj_mat = A @ pinv @ A.T
        Xp = (proj_mat @ X.T).T
        row_norm = torch.norm(Xp, dim=1, keepdim=True)
        row_norm = torch.clamp(row_norm, 1e-14)
        return Xp / row_norm

    stiefel_cols = 2 if gpca_dim == 1 else 3
    manifold = Stiefel()
    init_mat = torch.randn(p, stiefel_cols, device=device)
    while torch.linalg.matrix_rank(init_mat) < stiefel_cols:
        init_mat = torch.randn(p, stiefel_cols, device=device)
    Q_, _ = torch.linalg.qr(init_mat)
    stiefel_param = ManifoldParameter(Q_, manifold=manifold)

    learning_rate_opt = 1e-3
    optimizer_opt = geoopt.optim.RiemannianSGD([stiefel_param], lr=learning_rate_opt, momentum=0.9, weight_decay=1e-4)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_opt, T_0=5000, T_mult=2, eta_min=1e-4)

    count_pat = 0
    prev_loss = float('inf')

    def loss_1d(X, M):
        b = M[:, 0]
        t = M[:, 1]
        Xp = proj_to_geodesic(X, t, b)
        dsq = torch.stack([sphere_dist(X[i], Xp[i])**2 for i in range(X.shape[0])]).sum()
        return 0.5 * dsq

    def loss_2d(X, M):
        b = M[:, 0]
        t1 = M[:, 1]
        t2 = M[:, 2]
        Xp = proj_to_2sphere(X, t1, t2, b)
        dsq = torch.stack([sphere_dist(X[i], Xp[i])**2 for i in range(X.shape[0])]).sum()
        return 0.5 * dsq

    loss_func = (lambda: loss_1d(data_aug, stiefel_param)) if gpca_dim == 1 else (lambda: loss_2d(data_aug, stiefel_param))

    # 최적화 루프에 tqdm progress bar 적용
    from tqdm import tqdm
    with tqdm(range(max_iter), desc=f"GPCA-{gpca_dim}D Fitting", leave=False) as pbar:
        for step in pbar:
            optimizer_opt.zero_grad()
            val = loss_func()
            val.backward()
            optimizer_opt.step()
            scheduler_opt.step()
            with torch.no_grad():
                Qtemp, _ = torch.linalg.qr(stiefel_param)
                stiefel_param.copy_(Qtemp)
            now_loss = val.item()
            pbar.set_postfix(loss=f"{now_loss:.6f}")
            if abs(prev_loss - now_loss) < tolerance:
                count_pat += 1
                if count_pat >= patience:
                    pbar.write(f"Converged at step {step}, loss={now_loss:.6f}")
                    break
            else:
                count_pat = 0
            prev_loss = now_loss

    final_M = stiefel_param.detach().clone()
    if gpca_dim == 1:
        base_vec = final_M[:, 0]
        tangent = final_M[:, 1]
        sphere_vec = proj_to_geodesic(data_aug, tangent, base_vec)
    else:
        base_vec = final_M[:, 0]
        t1 = final_M[:, 1]
        t2 = final_M[:, 2]
        sphere_vec = proj_to_2sphere(data_aug, t1, t2, base_vec)

    sphere_vec_np = sphere_vec.cpu().numpy()
    data_np = data_aug.cpu().numpy()
    rss = np.sum(space.metric.squared_dist(data_np, sphere_vec_np))

    # (1D GPCA인 경우 추가 계산)
    if gpca_dim == 1:
        Xlog = np.array([space.metric.log(sphere_vec_np[i], base_vec.cpu().numpy()) for i in range(len(sphere_vec_np))])
        tan_np = tangent.cpu().numpy()
        norm_t = np.linalg.norm(tan_np) + 1e-14
        unit_tan = tan_np / norm_t
        ratio = Xlog / unit_tan
        geodesic_param = ratio[:, 0]
        geodesic_param = (geodesic_param + np.pi) % (2 * np.pi) - np.pi

        def function_to_minimize(t_arr, param_arr):
            cc = np.cos(t_arr).reshape(-1, 1) @ np.cos(param_arr).reshape(1, -1)
            ss = np.sin(t_arr).reshape(-1, 1) @ np.sin(param_arr).reshape(1, -1)
            distmat = np.square(np.arccos(np.clip(cc + ss, -1.0, 1.0)))
            return np.sum(distmat, axis=1)

        def geodesic_mean(param_arr, basep, uten):
            N = param_arr.size
            t_star = np.sum(param_arr) / N
            k = np.arange(N)
            t_candidates = t_star + 2 * np.pi * (k / N)
            arr = function_to_minimize(t_candidates, param_arr)
            idx_min = np.argmin(arr)
            t_opt = t_candidates[idx_min]
            return space.metric.exp(t_opt * uten, basep)

        gm = geodesic_mean(geodesic_param, base_vec.cpu().numpy(), unit_tan)
        var_geod = np.sum([space.metric.squared_dist(sphere_vec_np[i], gm) for i in range(len(sphere_vec_np))])
        mixed_var = rss + var_geod
        fit_score = 1 - (rss / mixed_var) if mixed_var > 1e-14 else 0.0
    else:
        sphere_2d = Hypersphere(dim=2)
        fm = FrechetMean(space=sphere_2d)
        fm.fit(sphere_vec_np)
        m_ = fm.estimate_
        var_sphere = np.sum(sphere_2d.metric.squared_dist(sphere_vec_np, m_))
        mixed_var = rss + var_sphere
        fit_score = 1 - (rss / mixed_var) if mixed_var > 1e-14 else 0.0

    return fit_score, rss


# ---------------------------
# CSV 업데이트 함수 (seed별 결과 기록)
# ---------------------------
def update_summary_csv(output_file, row, seed):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df = df[df['seed'] != seed]
    else:
        df = pd.DataFrame(columns=list(row.keys()))
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(output_file, index=False)

# ---------------------------
# 메인 루프: 시드별 실험 자동화
# ---------------------------
def main():
    
    total_seeds = 30
    batch_size = 64
    optimizer_type = 'adam'
    learning_rate = 0.0001
    epochs = 100
    
    # setting_name 자동 생성
    setting_name = f"alex_batch{batch_size}_{optimizer_type.lower()}_lr{learning_rate}_epochs{epochs}"
    output_dir = os.path.join("multiseed_output_test", setting_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # fc1.csv 파일을 기준으로 진행된 seed 확인 (이미 처리한 seed가 있다면 건너뜁니다.)
    fc1_file = os.path.join(output_dir, "fc1.csv")
    start_seed = 1
    if os.path.exists(fc1_file):
        df_temp = pd.read_csv(fc1_file)
        if not df_temp.empty:
            start_seed = int(df_temp['seed'].max()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for seed in range(start_seed, total_seeds + 1):
        print(f"\n===== Running experiment for seed {seed} =====")
        train_acc, test_acc, base_dir = run_training(seed, epochs=epochs, batch_size=batch_size,
                                                      learning_rate=learning_rate, optimizer_type=optimizer_type)
        print(f"Seed {seed}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
        # fc 레이어별 결과 (fc1, fc2, fc3)
        for layer_index in range(1, 4):
            eigen_csv = os.path.join(base_dir, "eigenvectors", f"fc{layer_index}.csv")
            print(f"GPCA for fc{layer_index} ...")
            # 전체 에폭 (1-100)
            S1_fit, S1_RSS = gpca_analysis(eigen_csv, gpca_dim=1, device=device, start_epoch=1)
            S2_fit, S2_RSS = gpca_analysis(eigen_csv, gpca_dim=2, device=device, start_epoch=1)
            row = {
                "seed": seed,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "S1_fit": S1_fit,
                "S1_RSS": S1_RSS,
                "S2_fit": S2_fit,
                "S2_RSS": S2_RSS
            }
            # 추가 에폭 범위: 2-100, 3-100, 4-100, 5-100
            for start in range(2, 6):
                s1_fit_subset, s1_rss_subset = gpca_analysis(eigen_csv, gpca_dim=1, device=device, start_epoch=start)
                s2_fit_subset, s2_rss_subset = gpca_analysis(eigen_csv, gpca_dim=2, device=device, start_epoch=start)
                row[f"S1_fit_{start}_100"] = s1_fit_subset
                row[f"S1_RSS_{start}_100"] = s1_rss_subset
                row[f"S2_fit_{start}_100"] = s2_fit_subset
                row[f"S2_RSS_{start}_100"] = s2_rss_subset
            
            output_file = os.path.join(output_dir, f"fc{layer_index}.csv")
            update_summary_csv(output_file, row, seed)
            print(f"Updated {output_file} with seed {seed} results.")

if __name__ == "__main__":
    main()
