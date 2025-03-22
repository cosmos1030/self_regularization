import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch
import torch
import torch.optim as optim

# Geomstats
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.backend as gs

# Geoopt (Riemannian optimization library)
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import Stiefel

torch.manual_seed(42)  # 항상 동일한 초기값 사용
np.random.seed(42)

# ------------------------------------------------------------------------------
# 1) Argument Parsing: multiple dirs/layers + start_epoch, end_epoch
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Project eigenvectors to hypersphere and visualize")
parser.add_argument('--dirs', type=str, nargs='+', required=True, help="List of base directories (e.g., run1 run2)")
parser.add_argument('--layers', type=str, nargs='+', required=True, help="List of layers (e.g., fc1 fc2 fc3)")
parser.add_argument('--start-epoch', type=int, default=0, help="Starting epoch (row index in CSV) to process")
parser.add_argument('--end-epoch', type=int, default=100, help="Ending epoch (row index in CSV) to process")
args = parser.parse_args()

# ------------------------------------------------------------------------------
# 2) Iterate Over Each Directory and Layer
# ------------------------------------------------------------------------------
for base_dir in args.dirs:
    for layer in args.layers:
        print(f"\n===== Processing Directory: {base_dir}, Layer: {layer} =====")

        # -------------------------------------------
        # Config: 파일 경로
        # -------------------------------------------
        csv_file = os.path.join(base_dir, f'eigenvectors/{layer}.csv')
        epoch_range_str = f"{args.start_epoch}_{args.end_epoch}"
        output_csv = os.path.join(base_dir, f"{layer}_sphere_epoch{epoch_range_str}.csv")
        output_fig = os.path.join(base_dir, f"{layer}_sphere_epoch{epoch_range_str}.png")
        output_heatmap_fig = os.path.join(base_dir, f"{layer}_sphere_heatmap_epoch{epoch_range_str}.png")
        output_residual_heatmap_fig = os.path.join(base_dir, f"{layer}_sphere_residual_heatmap_epoch{epoch_range_str}.png")

        if not os.path.exists(csv_file):
            print(f"❌ CSV file {csv_file} does not exist. Skipping this layer...")
            continue

        print(f"CSV file: {csv_file}")

        # -------------------------------------------
        # 1) CSV에서 고유벡터 로드 (Selecting a range of rows)
        # -------------------------------------------
        df = pd.read_csv(csv_file, sep='\s+', header=None)

        # Safely handle the case where end_epoch could exceed the data length
        end_idx = min(args.end_epoch, len(df))
        df = df.iloc[args.start_epoch-1:end_idx, :]

        eigv = df.to_numpy()  # shape (l, p)
        l, p = eigv.shape
        print("Data shape:", eigv.shape)

        # -------------------------------------------
        # ✅ CUDA 설정
        # -------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # PyTorch tensor 변환 후 GPU 이동
        data = torch.from_numpy(eigv).float().to(device)

        # -------------------------------------------
        # 2) 방향(부호) 정규화 (Same as your code)
        # -------------------------------------------
        def sphere_dist(x, y):
            nx = torch.norm(x)
            ny = torch.norm(y)
            inner = torch.dot(x, y) / (nx * ny + 1e-14)
            inner_clamped = torch.clamp(inner, -1.0, 1.0)
            return torch.arccos(inner_clamped)

        data_aug = data.clone()
        for _ in range(3):
            for i in range(l - 1):
                dist_pos = sphere_dist(data_aug[i+1], data_aug[i])
                dist_neg = sphere_dist(-data_aug[i+1], data_aug[i])
                if dist_neg < dist_pos:
                    data_aug[i+1] = -data_aug[i+1]

        # -------------------------------------------
        # 3) 기하 연산 함수 정의 (Same as your code)
        # -------------------------------------------
        def exp_map(v, p):
            norm_v = torch.norm(v)
            if norm_v < 1e-14:
                return p
            return torch.cos(norm_v)*p + torch.sin(norm_v)*(v / norm_v)

        def dist_sphere(A, B):
            norm_a = torch.norm(A, dim=1)
            norm_b = torch.norm(B, dim=1)
            inner_prod = torch.sum(A * B, dim=1)
            cos_angle = inner_prod / (norm_a * norm_b + 1e-14)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            return torch.arccos(cos_angle)

        def to_tangent(v, p):
            coef = torch.dot(v, p) / (torch.dot(p, p) + 1e-14)
            return v - coef * p

        def proj_to_2sphere(X, tangent_1, tangent_2, base_point):
            p1 = base_point
            p2 = exp_map(tangent_1, base_point)
            p3 = exp_map(tangent_2, base_point)

            A = torch.stack([p1, p2, p3], dim=1)  # shape: (p, 3)
            pinv = torch.linalg.pinv(A.T @ A)    # (3, 3)
            proj_mat = A @ pinv @ A.T            # (p, p)
            X_proj = (proj_mat @ X.T).T          # (l, p)

            row_norms = torch.norm(X_proj, dim=1, keepdim=True)
            row_norms = torch.clamp(row_norms, min=1e-14)
            return X_proj / row_norms

        # -------------------------------------------
        # 4) 리만 최적화 설정 (Stiefel Manifold)
        # -------------------------------------------
        manifold = Stiefel()
        init_mat = torch.randn(p, 3, device=device)
        while torch.linalg.matrix_rank(init_mat) < 3:
            init_mat = torch.randn(p, 3, device=device)
        Q, _ = torch.linalg.qr(init_mat)
        stiefel_param = ManifoldParameter(Q, manifold=manifold)

        learning_rate = 1e-3
        optimizer = geoopt.optim.RiemannianSGD([stiefel_param], lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-4)

        max_iter = 1000000
        tolerance = 1e-20
        patience = 50
        count = 0
        prev_loss = float('inf')

        # -------------------------------------------
        # 5) Loss 함수 정의 & 학습 루프 (Same as your code)
        # -------------------------------------------
        def loss_fn(X, M):
            base_point = M[:, 0]
            t1 = M[:, 1]
            t2 = M[:, 2]

            # X를 (base_point, t1, t2)가 이루는 스피어에 투영
            X_proj = proj_to_2sphere(X, t1, t2, base_point)
            dist_sq = dist_sphere(X, X_proj) ** 2

            return 0.5 * torch.sum(dist_sq)

        print("Start Riemannian optimization...")
        with tqdm(range(max_iter), desc="Training", unit="step") as pbar:
            for step in pbar:
                optimizer.zero_grad()
                loss_value = loss_fn(data_aug, stiefel_param)
                loss_value.backward()
                optimizer.step()
                scheduler.step()

                # 직교성 유지
                with torch.no_grad():
                    Q, _ = torch.linalg.qr(stiefel_param)
                    stiefel_param.copy_(Q)

                current_loss = loss_value.item()
                pbar.set_postfix(loss=f"{current_loss:.6f}")

                # 간단한 수렴 기준
                if abs(prev_loss - current_loss) < tolerance:
                    count += 1
                    if count >= patience:
                        print(f"✅ Converged at step {step} with loss={current_loss:.6f}")
                        break
                prev_loss = current_loss

        print("Final Loss:", prev_loss)

        # -------------------------------------------
        # 6) 최종 파라미터로 2-Sphere 투영 & 후처리
        # -------------------------------------------
        with torch.no_grad():
            final_M = stiefel_param.detach().clone()
            base_point = final_M[:, 0]
            t1 = final_M[:, 1]
            t2 = final_M[:, 2]

            sphere_vec = proj_to_2sphere(data_aug, t1, t2, base_point)
            sphere_vec_np = sphere_vec.cpu().numpy()

        # -------------------------------------------
        # 7) RSS 계산 (Geomstats)
        # -------------------------------------------
        space_full = Hypersphere(dim=p-1)
        data_np = data_aug.cpu().numpy()

        rss = np.sum(space_full.metric.squared_dist(data_np, sphere_vec_np))
        print("RSS:", rss)

        # -------------------------------------------
        # 8) Dimension Reduction (QR) & 저장
        # -------------------------------------------
        base_point_ = space_full.projection(base_point.cpu().numpy())
        point1_ = space_full.projection(t1.cpu().numpy())
        point2_ = space_full.projection(t2.cpu().numpy())

        basis = np.vstack((base_point_, point1_, point2_)).T
        Q_, _ = np.linalg.qr(basis)

        # (l, p)
        sphere_data = sphere_vec_np @ Q_
        sphere_data = np.array(sphere_data, dtype=np.float64)

        # 정규화
        norms = gs.linalg.norm(sphere_data, axis=1, keepdims=True)
        sphere_data /= norms

        np.savetxt(output_csv, sphere_data)
        print("Saved sphere data to:", output_csv)

        # -------------------------------------------
        # 9) Frechet Mean & 분산 계산 (Geomstats)
        # -------------------------------------------
        sphere = Hypersphere(dim=2)
        sphere_mean = FrechetMean(sphere)
        sphere_mean.fit(sphere_data)
        sphere_mean_estimate = sphere_mean.estimate_

        sphere_variance = np.sum(sphere.metric.squared_dist(sphere_data, sphere_mean_estimate))
        mixed_variance = rss + sphere_variance
        if mixed_variance > 1e-14:
            fitting_score = 1 - rss / mixed_variance
        else:
            fitting_score = 0.0
        print("Fitting Score:", fitting_score)

        # -------------------------------------------
        # 10) 시각화 (Geomstats + Matplotlib 3D)
        # -------------------------------------------
        # Normal 3D scatter
        #fig = plt.figure(figsize=(8, 8))
        #ax = visualization.plot(sphere_data, space='S2', color='black', alpha=0.7)
        #ax.set_box_aspect([1, 1, 1])
        #plt.title(f'Fitting Score: {fitting_score:.4f} RSS: {rss:.4f}\n'
        #         f'Experiment: {base_dir.split("/")[-1]} \n'
        #          f'Layer: {layer} | [Epoch {args.start_epoch} → {end_idx}]',
        #          fontsize=8)
        #fig.savefig(output_fig, dpi=300)
        #plt.show()
        #print("Figure saved to:", output_fig)

        ################################################
        # 추가 스캐터 플롯(heatmap 느낌)
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        # Generate a color gradient for visualization
        num_points = sphere_data.shape[0]
        colors = np.linspace(0, 1, num_points)

        # Create a scatter plot (Heatmap-like)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(sphere_data[:, 0], sphere_data[:, 1], sphere_data[:, 2],
                        c=colors, cmap='hot', alpha=0.7)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.2)  # Shrink and adjust aspect ratio
        cbar.set_label("Epochs (index)")

        ax.set_xlabel('X', fontsize=6)  # Adjust axis label font size
        ax.set_ylabel('Y', fontsize=6)
        ax.set_zlabel('Z', fontsize=6)
        
        # Move title to bottom
        plt.figtext(0.5, 0.05, f'Epoch Trajectory Heatmap: Change Over Training\n'
                       f'Fitting Score: {fitting_score:.4f} | RSS: {rss:.4f}\n'
                       f'Experiment: {base_dir.split("/")[-1]} \n'
                       f'Layer: {layer} | [Epoch {args.start_epoch} → {end_idx}]',
            ha="center", fontsize=8, wrap=True)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Sphere mesh
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_wireframe(x, y, z, color='skyblue', alpha=0.4, linewidth=0.5)
        fig.savefig(output_heatmap_fig, dpi=300)
        plt.show()
        print("Heatmap-like figure saved to:", output_heatmap_fig)

        ################################ residuals ##################################
        residuals = np.sqrt(space_full.metric.squared_dist(data_np, sphere_vec_np))

        # Color map by residual size
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(sphere_data[:, 0], sphere_data[:, 1], sphere_data[:, 2],
                        c=residuals, cmap='hot', alpha=0.7)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.2)  # Shrink and adjust aspect ratio
        cbar.set_label("Residual Size")

        ax.set_xlabel('X', fontsize=6)  # Adjust axis label font size
        ax.set_ylabel('Y', fontsize=6)
        ax.set_zlabel('Z', fontsize=6)
        
        plt.figtext(0.5, 0.05, f'Residual Heatmap: Point-wise Fitting Error\n'
                       f'Fitting Score: {fitting_score:.4f} | RSS: {rss:.4f}\n'
                       f'Experiment: {base_dir.split("/")[-1]} \n'
                       f'Layer: {layer} | [Epoch {args.start_epoch} → {end_idx}]',
            ha="center", fontsize=8, wrap=True)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.plot_wireframe(x, y, z, color='skyblue', alpha=0.4, linewidth=0.5)
        fig.savefig(output_residual_heatmap_fig, dpi=300)
        plt.show()
        print("Heatmap-like figure saved to:", output_residual_heatmap_fig)

        print("✅ Processing complete for this layer!\n")

print("✅ All processing complete for all directories and layers!")
