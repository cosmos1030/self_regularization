import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
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
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '9'

# 시드 고정 (재현성)
torch.manual_seed(42)
np.random.seed(42)

def main():
    # ------------------------------------------------------------------------------
    # 1) Argument Parsing
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Project eigenvectors to hypersphere (1D or 2D GPCA) and visualize")
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                        help="List of base directories (e.g., run1 run2)")
    parser.add_argument('--layers', type=str, nargs='+', required=True,
                        help="List of layers (e.g., fc1 fc2 fc3)")
    parser.add_argument('--start-epoch', type=int, default=0,
                        help="Starting epoch (row index in CSV) to process")
    parser.add_argument('--end-epoch', type=int, default=100,
                        help="Ending epoch (row index in CSV) to process")
    parser.add_argument('--gpca-dim', type=int, default=2, choices=[1,2],
                        help="GPCA dimension (1 or 2). Default=2 -> 2D GPCA")
    args = parser.parse_args()

    # ------------------------------------------------------------------------------
    # 2) Iterate Over Each Directory and Layer
    # ------------------------------------------------------------------------------
    for base_dir in args.dirs:
        for layer in args.layers:
            print(f"\n===== Processing Directory: {base_dir}, Layer: {layer} =====")

            # -------------------------------------------
            # 파일 경로 세팅
            # -------------------------------------------
            csv_file = os.path.join(base_dir, f'eigenvectors/{layer}.csv')
            epoch_range_str = f"{args.start_epoch}_{args.end_epoch}"
            # 결과 파일(공통)
            output_csv = os.path.join(base_dir, f"{layer}_sphere_epoch{epoch_range_str}.csv")
            output_heatmap_fig = os.path.join(base_dir, f"{layer}_sphere_heatmap_epoch{epoch_range_str}.png")
            output_residual_heatmap_fig = os.path.join(base_dir, f"{layer}_sphere_residual_heatmap_epoch{epoch_range_str}.png")
            # 1D GPCA 전용 그래프
            output_1d_fig = os.path.join(base_dir, f"{layer}_1Dgeodesic_epoch{epoch_range_str}.png")

            if not os.path.exists(csv_file):
                print(f"❌ CSV file {csv_file} does not exist. Skipping this layer...")
                continue

            print(f"CSV file: {csv_file}")

            # -------------------------------------------
            # 1) CSV에서 고유벡터 읽기 (선택한 구간)
            # -------------------------------------------
            df = pd.read_csv(csv_file, sep='\s+', header=None)
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

            data = torch.from_numpy(eigv).float().to(device)

            # -------------------------------------------
            # 2) 방향(부호) 정규화
            # -------------------------------------------
            space_full = Hypersphere(dim=p-1)

            def sphere_dist(x, y):
                nx = torch.norm(x)
                ny = torch.norm(y)
                inner = torch.dot(x, y) / (nx*ny + 1e-14)
                inner_clamped = torch.clamp(inner, -1.0, 1.0)
                return torch.arccos(inner_clamped)

            data_aug = data.clone()
            # 반복해서 인접벡터 부호 안정화
            for _ in range(3):
                for i in range(l - 1):
                    dist_pos = sphere_dist(data_aug[i+1], data_aug[i])
                    dist_neg = sphere_dist(-data_aug[i+1], data_aug[i])
                    if dist_neg < dist_pos:
                        data_aug[i+1] = -data_aug[i+1]

            # -------------------------------------------
            # 3) 기하학 연산 함수
            # -------------------------------------------
            def exp_map(v, p):
                norm_v = torch.norm(v)
                if norm_v < 1e-14:
                    return p
                return torch.cos(norm_v)*p + torch.sin(norm_v)*(v/norm_v)

            def dist_sphere(A, B):
                norm_a = torch.norm(A, dim=1)
                norm_b = torch.norm(B, dim=1)
                inner = torch.sum(A*B, dim=1)
                cos_ = inner / (norm_a*norm_b + 1e-14)
                cos_ = torch.clamp(cos_, -1.0, 1.0)
                return torch.arccos(cos_)

            def to_tangent(v, p):
                alpha = torch.dot(v, p) / (torch.dot(p, p)+1e-14)
                return v - alpha*p

            # (A) 1D GPCA 투영
            def proj_to_geodesic(X, tangent, base_point):
                norm_t = torch.norm(tangent)
                if norm_t < 1e-14:
                    # tangent 0이면 그냥 base_point
                    return base_point.unsqueeze(0).repeat(X.shape[0], 1)

                dot_xt = torch.sum(X*tangent, dim=1)
                dot_xb = torch.sum(X*base_point, dim=1)
                factor = torch.atan2(dot_xt, dot_xb) / (norm_t+1e-14)

                cos_f = torch.cos(factor).unsqueeze(1)
                sin_f = torch.sin(factor).unsqueeze(1)
                unit_t = tangent / norm_t

                proj = cos_f*base_point + sin_f*unit_t

                # +/- 더 가까운 쪽
                dist_p = dist_sphere(X, proj)
                dist_m = dist_sphere(X, -proj)
                mask = dist_m < dist_p
                proj[mask] = -proj[mask]
                return proj

            # (B) 2D GPCA 투영
            def proj_to_2sphere(X, t1, t2, base_point):
                p1 = base_point
                p2 = exp_map(t1, base_point)
                p3 = exp_map(t2, base_point)

                A = torch.stack([p1, p2, p3], dim=1)  # (p,3)
                pinv = torch.linalg.pinv(A.T@A)
                proj_mat = A@pinv@A.T
                Xp = (proj_mat@X.T).T

                row_norm = torch.norm(Xp, dim=1, keepdim=True)
                row_norm = torch.clamp(row_norm, 1e-14)
                return Xp/row_norm

            # ------------------------------------------------------------------------------
            # 4) Stiefel Manifold 파라미터 생성 (gpca_dim=1 or 2)
            # ------------------------------------------------------------------------------
            gpca_dim = args.gpca_dim
            if gpca_dim == 1:
                stiefel_cols = 2
            else:
                stiefel_cols = 3

            manifold = Stiefel()
            init_mat = torch.randn(p, stiefel_cols, device=device)
            while torch.linalg.matrix_rank(init_mat) < stiefel_cols:
                init_mat = torch.randn(p, stiefel_cols, device=device)
            Q_, _ = torch.linalg.qr(init_mat)
            stiefel_param = ManifoldParameter(Q_, manifold=manifold)

            # ------------------------------------------------------------------------------
            # 5) Riemannian Optimizer
            # ------------------------------------------------------------------------------
            learning_rate = 1e-3
            optimizer = geoopt.optim.RiemannianSGD([stiefel_param], lr=learning_rate,
                                                   momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5000, T_mult=2, eta_min=1e-4
            )

            max_iter = 200000
            tolerance = 1e-5
            patience = 100
            count_pat = 0
            prev_loss = float('inf')

            # Loss 함수
            def loss_1d(X, M):
                b = M[:,0]
                t = M[:,1]
                Xp = proj_to_geodesic(X, t, b)
                dsq = dist_sphere(X, Xp)**2
                return 0.5*torch.sum(dsq)

            def loss_2d(X, M):
                b = M[:,0]
                t1 = M[:,1]
                t2 = M[:,2]
                Xp = proj_to_2sphere(X, t1, t2, b)
                dsq = dist_sphere(X, Xp)**2
                return 0.5*torch.sum(dsq)

            if gpca_dim == 1:
                def total_loss():
                    return loss_1d(data_aug, stiefel_param)
            else:
                def total_loss():
                    return loss_2d(data_aug, stiefel_param)

            # ------------------------------------------------------------------------------
            # 6) Optimization loop
            # ------------------------------------------------------------------------------
            with tqdm(range(max_iter), desc=f"GPCA-{gpca_dim}D") as pbar:
                for step in pbar:
                    optimizer.zero_grad()
                    val = total_loss()
                    val.backward()
                    optimizer.step()
                    scheduler.step()

                    # 직교성 보정
                    with torch.no_grad():
                        Qtemp, _ = torch.linalg.qr(stiefel_param)
                        stiefel_param.copy_(Qtemp)

                    now_loss = val.item()
                    pbar.set_postfix(loss=f"{now_loss:.6f}")

                    if abs(prev_loss - now_loss) < tolerance:
                        count_pat += 1
                        if count_pat >= patience:
                            print(f"Converged at step {step}, loss={now_loss:.6f}")
                            break
                    else:
                        count_pat = 0
                    prev_loss = now_loss

            # ------------------------------------------------------------------------------
            # 7) 최종 파라미터로 데이터 투영, RSS 계산
            # ------------------------------------------------------------------------------
            final_M = stiefel_param.detach().clone()
            if gpca_dim == 1:
                base = final_M[:,0]
                tangent = final_M[:,1]
                sphere_vec = proj_to_geodesic(data_aug, tangent, base)
            else:
                base = final_M[:,0]
                t1 = final_M[:,1]
                t2 = final_M[:,2]
                sphere_vec = proj_to_2sphere(data_aug, t1, t2, base)

            sphere_vec_np = sphere_vec.cpu().numpy()
            data_np = data_aug.cpu().numpy()

            rss = np.sum(space_full.metric.squared_dist(data_np, sphere_vec_np))
            print("RSS:", rss)

            # ------------------------------------------------------------------------------
            # (A) 1D GPCA 시각화: 지오데식 파라미터 vs Epoch
            # ------------------------------------------------------------------------------
            if gpca_dim == 1:
                # (1) 지오데식 파라미터 계산
                base_np = base.cpu().numpy()
                tan_np = tangent.cpu().numpy()
                Xlog = space_full.metric.log(sphere_vec_np, base_np)
                norm_t = np.linalg.norm(tan_np) + 1e-14
                unit_tan = tan_np / norm_t
                ratio = Xlog / unit_tan  # elementwise
                geodesic_param = ratio[:,0]
                geodesic_param = (geodesic_param + np.pi) % (2 * np.pi) - np.pi

                # (2) 지오데식 평균 & fitting score
                # Huckemann--Ziezold 방식
                def function_to_minimize(t_arr, param_arr):
                    cc = (np.cos(t_arr).reshape(-1,1) @ np.cos(param_arr).reshape(1,-1))
                    ss = (np.sin(t_arr).reshape(-1,1) @ np.sin(param_arr).reshape(1,-1))
                    distmat = np.square(np.arccos(np.clip(cc+ss, -1.0,1.0)))
                    return np.sum(distmat, axis=1)

                def geodesic_mean(param_arr, basep, uten):
                    N = param_arr.size
                    t_star = np.sum(param_arr)/N
                    k = np.arange(N)
                    t_candidates = t_star + 2*np.pi*(k/N)
                    arr = function_to_minimize(t_candidates, param_arr)
                    idx_min = np.argmin(arr)
                    t_opt = t_candidates[idx_min]
                    return space_full.metric.exp(t_opt*uten, basep)

                gm = geodesic_mean(geodesic_param, base_np, unit_tan)
                var_geod = np.sum(space_full.metric.squared_dist(sphere_vec_np, gm))
                mixed_var = rss + var_geod
                if mixed_var > 1e-14:
                    fit_score = 1 - rss/mixed_var
                else:
                    fit_score = 0.0

                print("Fitting Score:", fit_score)

                # (3) Epoch vs. geodesic_param 그래프
                x_epochs = np.arange(1, l+1)
                plt.figure(figsize=(6,4))
                plt.plot(x_epochs, geodesic_param, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Geodesic Parameter')
                plt.title(f'1D GPCA: RSS={rss:.4f}, FitScore={fit_score:.4f}')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_1d_fig, dpi=300)
                plt.show()
                print("1D GPCA figure saved:", output_1d_fig)
                
                # Compute per-epoch residuals: residual = distance between original vector and its projection
                residuals = np.sqrt(space_full.metric.squared_dist(data_np, sphere_vec_np))
                epochs = np.arange(args.start_epoch, args.start_epoch + residuals.shape[0])
    
                plt.figure(figsize=(6,4))
                plt.plot(epochs, residuals, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Residual Magnitude')
                plt.title('Residual Magnitude vs Epoch (1D GPCA)')
                plt.grid(True)
                plt.tight_layout()
                output_1d_residual_fig = os.path.join(base_dir, f"{layer}_residual_vs_epoch_1D.png")
                plt.savefig(output_1d_residual_fig, dpi=300)
                plt.show()
                print("1D Residual vs Epoch figure saved:", output_1d_residual_fig)

            # ------------------------------------------------------------------------------
            # (B) 2D GPCA 시각화: 구 위 3D scatter + heatmap
            # ------------------------------------------------------------------------------
            else:
                # 임의로 (l,3) 임베딩해서 시각화
                base_np = base.cpu().numpy()
                t1_np = t1.cpu().numpy()
                t2_np = t2.cpu().numpy()

                base_np = space_full.projection(base_np)
                t1_np = space_full.projection(t1_np)
                t2_np = space_full.projection(t2_np)

                B_mat = np.vstack((base_np, t1_np, t2_np)).T  # (p,3)
                Qmat, _ = np.linalg.qr(B_mat)
                sphere_data = sphere_vec_np @ Qmat
                sphere_data /= (np.linalg.norm(sphere_data, axis=1, keepdims=True) + 1e-14)
                np.savetxt(output_csv, sphere_data)
                print("2D sphere data saved:", output_csv)

                # Frechet Mean for S^2
                sphere_2d = Hypersphere(dim=2)
                fm = FrechetMean(sphere_2d)
                fm.fit(sphere_data)
                m_ = fm.estimate_
                var_sphere = np.sum(sphere_2d.metric.squared_dist(sphere_data, m_))
                mixed_var = rss + var_sphere
                if mixed_var>1e-14:
                    fit_score = 1 - rss/mixed_var
                else:
                    fit_score=0.0

                print("Fitting Score:", fit_score)

                from mpl_toolkits.mplot3d import Axes3D  # noqa
                num_pts = sphere_data.shape[0]
                color_vals = np.linspace(0, 1, num_pts)

                # ---- Heatmap by epoch index ----
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(sphere_data[:,0], sphere_data[:,1], sphere_data[:,2],
                                c=color_vals, cmap='hot', alpha=0.7)
                cbar = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.2)
                cbar.set_label("Epoch (Index)")
                # 예시: 6개의 tick을 표시
                ticks = np.linspace(args.start_epoch, args.end_epoch, 6)
                # color_vals가 0~1이므로, ticks를 0~1 범위로 맞춰줍니다.
                norm_ticks = (ticks - args.start_epoch) / (args.end_epoch - args.start_epoch)
                cbar.set_ticks(norm_ticks)
                cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks])


                ax.set_xlim([-1,1])
                ax.set_ylim([-1,1])
                ax.set_zlim([-1,1])
                ax.set_box_aspect([1,1,1])

                #plt.figtext(0.5, 0.05,
                            #f'2D GPCA: RSS={rss:.4f}, Fit={fit_score:.4f}',
                            #ha="center", fontsize=8, wrap=True)
                plt.title(f'2D GPCA: RSS={rss:.4f}, FitScore={fit_score:.4f}')

                # 구 mesh
                u = np.linspace(0, 2*np.pi, 30)
                v = np.linspace(0, np.pi, 30)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones_like(u), np.cos(v))
                ax.xaxis.set_major_locator(LinearLocator(numticks=5))
                ax.yaxis.set_major_locator(LinearLocator(numticks=5))
                ax.zaxis.set_major_locator(LinearLocator(numticks=5))
                ax.plot_wireframe(x, y, z, color='skyblue', alpha=0.4, linewidth=0.5)

                fig.savefig(output_heatmap_fig, dpi=300)
                plt.show()
                print("2D GPCA heatmap figure saved:", output_heatmap_fig)

                # ---- Residual Heatmap ----
                residuals = np.sqrt(space_full.metric.squared_dist(data_np, sphere_vec_np))
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(sphere_data[:,0], sphere_data[:,1], sphere_data[:,2],
                                c=residuals, cmap='hot', alpha=0.7)
                cb = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.2)
                cb.set_label("Residual Distance")

                ax.set_xlim([-1,1])
                ax.set_ylim([-1,1])
                ax.set_zlim([-1,1])
                ax.set_box_aspect([1,1,1])

                #plt.figtext(0.5, 0.05,
                 #           f'Residual Heatmap: RSS={rss:.4f}',
                 #           ha="center", fontsize=8, wrap=True)
                plt.title(f'Residual Heatmap: RSS={rss:.4f}, FitScore={fit_score:.4f}')
                ax.plot_wireframe(x, y, z, color='skyblue', alpha=0.4, linewidth=0.5)

                fig.savefig(output_residual_heatmap_fig, dpi=300)
                plt.show()
                print("Residual Heatmap figure saved:", output_residual_heatmap_fig)
                
                # Compute per-epoch residuals for 2D projection
                residuals = np.sqrt(space_full.metric.squared_dist(data_np, sphere_vec_np))
                epochs = np.arange(args.start_epoch, args.start_epoch + residuals.shape[0])
    
                plt.figure(figsize=(6,4))
                plt.plot(epochs, residuals, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Residual Magnitude')
                plt.title('Residual Magnitude vs Epoch (2D GPCA)')
                plt.grid(True)
                plt.tight_layout()
                # Optionally, save the figure
                output_2d_residual_fig = os.path.join(base_dir, f"{layer}_residual_vs_epoch_2D.png")
                plt.savefig(output_2d_residual_fig, dpi=300)
                plt.show()
                print("2D Residual vs Epoch figure saved:", output_2d_residual_fig)

            print("✅ Processing complete for this layer!\n")
            
            

    print("✅ All processing complete for all directories and layers!")

if __name__ == "__main__":
    main()
