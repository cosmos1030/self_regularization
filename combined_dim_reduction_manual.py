import os
import argparse
import numpy as np
import pandas as pd # Not strictly needed if not reading external CSVs for this script
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
import plotly.graph_objects as go  # ✅ for interactive HTML export



# Geoopt (Riemannian optimization library)
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import Stiefel
# import os # Already imported

os.environ['CUDA_VISIBLE_DEVICES'] = '9' # Set as per your environment

# 시드 고정 (재현성)
torch.manual_seed(42)
np.random.seed(42)

def main():
    # ------------------------------------------------------------------------------
    # 1) Argument Parsing
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Track and visualize eigenvector trajectory with crossing detection.")
    parser.add_argument('--dirs',       type=str, nargs='+', required=True, help="List of run directories")
    parser.add_argument('--layers',     type=str, nargs='+', required=True, help="Which FC layers to process, e.g. fc1 fc2")
    parser.add_argument('--eig-index',  type=int, default=0, help="Initial singular vector index (row of Vh) to track (0-based)")
    parser.add_argument('--start-epoch',type=int, default=1, help="First epoch to include")
    parser.add_argument('--end-epoch',  type=int, default=100, help="Last epoch to include")
    parser.add_argument('--gpca-dim',   type=int, default=2, choices=[1,2], help="GPCA dimension (1 or 2)")
    parser.add_argument('--manual-switch', type=str, nargs='*',
                         help="Manual switch points for tracking. Format: 'epoch:new_index'. "
                              "Example: --manual-switch 35:2 60:1")
    args = parser.parse_args()
    
    manual_switch_points = {}
    if args.manual_switch:
         for switch in args.manual_switch:
             try:
                 epoch_str, index_str = switch.split(':')
                 manual_switch_points[int(epoch_str)] = int(index_str)
             except ValueError:
                 print(f"Warning: Invalid format for --manual-switch '{switch}'. Ignoring. Expected format is 'epoch:new_index'.")
     
    if manual_switch_points:
         print(f"===== Manual switch points have been set: {manual_switch_points} =====")

    # ------------------------------------------------------------------------------
    # 2) Iterate Over Each Directory and Layer
    # ------------------------------------------------------------------------------
    for base_dir in args.dirs:
        for layer in args.layers:
            # -------------------------------------------
            # 결과 저장 폴더 생성
            # -------------------------------------------
            analysis_parent_dir = os.path.join(base_dir, "gpca_tracked_trajectory_results")
            current_analysis_folder_name = f"layer_{layer}_initial_eig_{args.eig_index}_epochs_{args.start_epoch}-{args.end_epoch}_{args.manual_switch}"
            output_folder = os.path.join(analysis_parent_dir, current_analysis_folder_name)
            os.makedirs(output_folder, exist_ok=True)

            print(f"\n===== Processing Directory: {base_dir} =====")
            print(f"===== Layer: {layer}, Initial EigIndex: {args.eig_index}, Epochs: {args.start_epoch}-{args.end_epoch} =====")
            print(f"===== Output will be saved to: {output_folder} =====")

            # 파일 경로 세팅
            output_csv = os.path.join(output_folder, "tracked_sphere_projection_coords.csv")
            output_heatmap_fig = os.path.join(output_folder, "tracked_2D_projection_heatmap.png")
            output_residual_heatmap_fig = os.path.join(output_folder, "tracked_2D_residual_heatmap.png")
            output_1d_fig = os.path.join(output_folder, "tracked_1D_geodesic_params.png")
            output_1d_residual_fig = os.path.join(output_folder, "tracked_1D_residuals_vs_epoch.png")
            output_2d_residual_fig = os.path.join(output_folder, "tracked_2D_residuals_vs_epoch.png")
            crossing_info_file_path = os.path.join(output_folder, f"sv_trajectory_and_crossing_info_initial_eig{args.eig_index}.txt")

            # ----------------------------------------------------------------------
            #  apartado 1) 모든 에포크의 전체 Vh 행렬 로드 및 부호 정렬
            # ----------------------------------------------------------------------
            all_epochs_Vh_raw = []
            num_epochs_to_process = args.end_epoch - args.start_epoch + 1

            for ep_num in range(args.start_epoch, args.end_epoch + 1):
                npz_path = os.path.join(base_dir, f"eigenvectors/epoch{str(ep_num)}.npz")
                if not os.path.exists(npz_path):
                    print(f"Warning: NPZ file not found {npz_path}, will use None for epoch {ep_num}")
                    all_epochs_Vh_raw.append(None)
                    continue
                try:
                    data_content = np.load(npz_path)
                    if f"{layer}_Vh" in data_content:
                        all_epochs_Vh_raw.append(data_content[f"{layer}_Vh"])
                    else:
                        print(f"Warning: Key '{layer}_Vh' not found in {npz_path} for epoch {ep_num}, will use None.")
                        all_epochs_Vh_raw.append(None)
                except Exception as e:
                    print(f"Error loading Vh from {npz_path} for epoch {ep_num}: {e}. Will use None.")
                    all_epochs_Vh_raw.append(None)
            
            if all(vh is None for vh in all_epochs_Vh_raw):
                print(f"❌ No Vh data found for layer {layer} in the specified epoch range. Skipping this layer.")
                continue

            aligned_all_epochs_Vh = [None] * len(all_epochs_Vh_raw)
            first_valid_vh = next((vh for vh in all_epochs_Vh_raw if vh is not None), None)
            if first_valid_vh is None:
                print(f"❌ No valid Vh matrices found after loading for layer {layer}. Skipping.")
                continue
            
            num_singular_vectors = first_valid_vh.shape[0]

            for sv_j in range(num_singular_vectors):
                trajectory_sv_j = []
                epoch_indices_for_sv_j = []

                for i, vh_matrix_at_epoch_i in enumerate(all_epochs_Vh_raw):
                    if vh_matrix_at_epoch_i is not None and sv_j < vh_matrix_at_epoch_i.shape[0]:
                        trajectory_sv_j.append(vh_matrix_at_epoch_i[sv_j])
                        epoch_indices_for_sv_j.append(i)
                
                if not trajectory_sv_j:
                    continue

                trajectory_sv_j_np = np.array(trajectory_sv_j)

                for t_traj in range(1, len(trajectory_sv_j_np)):
                    if np.dot(trajectory_sv_j_np[t_traj], trajectory_sv_j_np[t_traj-1]) < 0:
                        trajectory_sv_j_np[t_traj] *= -1
                
                for i_traj, original_raw_idx in enumerate(epoch_indices_for_sv_j):
                    if aligned_all_epochs_Vh[original_raw_idx] is None and all_epochs_Vh_raw[original_raw_idx] is not None:
                        aligned_all_epochs_Vh[original_raw_idx] = np.zeros_like(all_epochs_Vh_raw[original_raw_idx])
                    
                    if aligned_all_epochs_Vh[original_raw_idx] is not None:
                         aligned_all_epochs_Vh[original_raw_idx][sv_j] = trajectory_sv_j_np[i_traj]
            
            # ----------------------------------------------------------------------
            # apartado 2) Trajectory 추적 및 해당 벡터 추출
            # ----------------------------------------------------------------------
            tracked_vectors = []
            crossing_info_details = []
            
            current_tracked_idx = args.eig_index
            
            for t in range(num_epochs_to_process):
                Vh_t = aligned_all_epochs_Vh[t]

                if Vh_t is None:
                    print(f"Warning: Missing Vh data for epoch {args.start_epoch + t}, cannot append to trajectory.")
                    tracked_vectors.append(None)
                    if t < num_epochs_to_process - 1:
                        crossing_info_details.append(
                            (args.start_epoch + t, args.start_epoch + t + 1, current_tracked_idx, current_tracked_idx, "N/A (Missing current Vh)")
                        )
                    continue

                if not (0 <= current_tracked_idx < Vh_t.shape[0]):
                    print(f"Error: current_tracked_idx {current_tracked_idx} is out of bounds for Vh_t at epoch {args.start_epoch + t}. Shape: {Vh_t.shape}. Stopping.")
                    break 
                
                vector_to_add = Vh_t[current_tracked_idx]
                tracked_vectors.append(vector_to_add)

                # ▼▼▼ [핵심 수정 부분] 자동 추적 로직 완전 삭제 ▼▼▼
                if t < num_epochs_to_process - 1:
                     epoch_t = args.start_epoch + t
                     epoch_t_plus_1 = args.start_epoch + t + 1
                     
                     best_next_idx = -1
                     log_message = "N/A"

                     # 1. 수동 전환이 지정되었는지 먼저 확인
                     if epoch_t_plus_1 in manual_switch_points:
                         best_next_idx = manual_switch_points[epoch_t_plus_1]
                         log_message = "MANUAL OVERRIDE"
                         print(f"Epoch {epoch_t_plus_1}: MANUAL SWITCH to index {best_next_idx}")

                     # 2. 수동 전환이 없으면, 현재 추적 중인 인덱스를 그대로 유지 (자동 추적 없음)
                     else:
                         best_next_idx = current_tracked_idx # 현재 인덱스를 다음 인덱스로 그대로 사용
                         log_message = "NO CHANGE"    # 로그에 표시할 문자열

                     # 결과 기록 및 인덱스 업데이트
                     crossing_info_details.append(
                         (epoch_t, epoch_t_plus_1, current_tracked_idx, best_next_idx, log_message)
                     )
                     current_tracked_idx = best_next_idx
            
            # None 값을 가진 tracked_vectors 처리
            final_tracked_vectors_for_gpca = [v for v in tracked_vectors if v is not None]
            if not final_tracked_vectors_for_gpca:
                print(f"❌ No valid vectors in the tracked trajectory for layer {layer}, initial_eig_{args.eig_index}. Skipping GPCA.")
                with open(crossing_info_file_path, 'w') as f_cross:
                    f_cross.write(f"Singular Vector Trajectory Tracking & Crossing Information\n")
                    f_cross.write(f"Layer: {layer}, Initial SV Index: {args.eig_index}\n")
                    f_cross.write(f"Epoch Range: {args.start_epoch}-{args.end_epoch}\n")
                    f_cross.write("="*70 + "\n")
                    f_cross.write("Epoch_t -> Epoch_t+1 : Old_SV_Index_at_t -> New_SV_Index_at_t+1 (Info)\n")
                    f_cross.write("="*70 + "\n")
                    for ct, nt, oi, ni, sim in crossing_info_details:
                        f_cross.write(f"{ct:03d} -> {nt:03d} : SV {oi} -> SV {ni} ({sim})\n")
                    if not final_tracked_vectors_for_gpca:
                         f_cross.write("\nNo valid vectors remained in the trajectory for GPCA analysis.\n")
                print(f"Crossing and trajectory information saved to: {crossing_info_file_path}")
                continue

            eigv = np.array(final_tracked_vectors_for_gpca)
            l, p = eigv.shape
            if l < 2 :
                print(f"❌ Only {l} valid vector(s) in the tracked trajectory for layer {layer}, initial_eig_{args.eig_index}. Skipping GPCA.")
                continue

            print(f"Loaded {l} vectors (after tracking) of dimension {p} for layer {layer}, initial_eig_{args.eig_index}")
            
            with open(crossing_info_file_path, 'w') as f_cross:
                f_cross.write(f"Singular Vector Trajectory Tracking & Crossing Information\n")
                f_cross.write(f"Layer: {layer}, Initial SV Index: {args.eig_index}\n")
                f_cross.write(f"Epoch Range: {args.start_epoch}-{args.end_epoch}\n")
                f_cross.write(f"Number of vectors in final trajectory for GPCA: {l}\n")
                f_cross.write("="*70 + "\n")
                f_cross.write("Epoch_t -> Epoch_t+1 : SV_Index_at_t -> SV_Index_at_t+1 (Info)\n")
                f_cross.write("="*70 + "\n")
                for ct, nt, oi, ni, sim_val in crossing_info_details:
                    actual_tracked_idx_at_t_plus_1 = ni
                    f_cross.write(f"{ct:03d} -> {nt:03d} : SV {oi} -> SV {actual_tracked_idx_at_t_plus_1} (Sim: {sim_val})\n")
            print(f"Crossing and trajectory information saved to: {crossing_info_file_path}")

            # -------------------------------------------
            # ✅ CUDA 설정
            # -------------------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device:", device)

            data = torch.from_numpy(eigv).float().to(device)

            # -------------------------------------------
            # apartado 2) 방향(부호) 정규화
            # -------------------------------------------
            space_full = Hypersphere(dim=p-1)

            def sphere_dist_torch(x, y):
                nx = torch.norm(x)
                ny = torch.norm(y)
                inner = torch.dot(x, y) / (nx*ny + 1e-14)
                inner_clamped = torch.clamp(inner, -1.0, 1.0)
                return torch.arccos(inner_clamped)

            data_aug = data.clone()
            if l > 1:
                for _ in range(3):
                    for i in range(l - 1):
                        dist_pos = sphere_dist_torch(data_aug[i+1], data_aug[i])
                        dist_neg = sphere_dist_torch(-data_aug[i+1], data_aug[i])
                        if dist_neg < dist_pos:
                            data_aug[i+1] *= -1
            
            # -------------------------------------------
            # apartado 3) 기하학 연산 함수
            # -------------------------------------------
            def exp_map(v_tangent, p_base):
                norm_v = torch.norm(v_tangent)
                if norm_v < 1e-14:
                    return p_base
                return torch.cos(norm_v)*p_base + torch.sin(norm_v)*(v_tangent/norm_v)

            def dist_sphere_batch(A_batch, B_batch):
                norm_a = torch.norm(A_batch, dim=1, keepdim=True)
                norm_b = torch.norm(B_batch, dim=1, keepdim=True)
                inner = torch.sum(A_batch*B_batch, dim=1)
                cos_ = inner / (norm_a.squeeze(-1)*norm_b.squeeze(-1) + 1e-14)
                cos_ = torch.clamp(cos_, -1.0, 1.0)
                return torch.arccos(cos_)

            def proj_to_geodesic(X_batch, tangent_vec, base_pt):
                norm_t = torch.norm(tangent_vec)
                if norm_t < 1e-14:
                    return base_pt.unsqueeze(0).repeat(X_batch.shape[0], 1)
                dot_xt = torch.sum(X_batch * tangent_vec.unsqueeze(0), dim=1)
                dot_xb = torch.sum(X_batch * base_pt.unsqueeze(0), dim=1)
                factor = torch.atan2(dot_xt, dot_xb) / (norm_t + 1e-14)
                cos_f = torch.cos(factor).unsqueeze(1)
                sin_f = torch.sin(factor).unsqueeze(1)
                unit_t = tangent_vec / norm_t
                proj = cos_f * base_pt.unsqueeze(0) + sin_f * unit_t.unsqueeze(0)
                return proj

            def proj_to_2sphere(X_batch, t1_vec, t2_vec, base_pt):
                p1_sphere = base_pt
                p2_sphere = exp_map(t1_vec, base_pt)
                p3_sphere = exp_map(t2_vec, base_pt)
                A_plane_basis = torch.stack([p1_sphere, p2_sphere, p3_sphere], dim=1)
                try:
                    Q_A, _ = torch.linalg.qr(A_plane_basis)
                    proj_matrix_on_plane = Q_A @ Q_A.T
                    Xp_on_plane = X_batch @ proj_matrix_on_plane
                except Exception as e:
                    print(f"Warning: QR decomposition failed: {e}. Using pseudo-inverse.")
                    pinv_A_T_A = torch.linalg.pinv(A_plane_basis.T @ A_plane_basis)
                    proj_matrix_on_plane = A_plane_basis @ pinv_A_T_A @ A_plane_basis.T
                    Xp_on_plane = (proj_matrix_on_plane @ X_batch.T).T
                row_norm = torch.norm(Xp_on_plane, dim=1, keepdim=True)
                Xp_normalized = Xp_on_plane / torch.clamp(row_norm, min=1e-14)
                return Xp_normalized

            # ------------------------------------------------------------------------------
            # apartado 4) Stiefel Manifold 파라미터 생성
            # ------------------------------------------------------------------------------
            gpca_dim = args.gpca_dim
            stiefel_cols = gpca_dim + 1

            manifold = Stiefel()
            init_rand_matrix = torch.randn(p, stiefel_cols, device=device)
            q_init, _ = torch.linalg.qr(init_rand_matrix)
            stiefel_param_init_val = q_init[:, :stiefel_cols]
            stiefel_param = ManifoldParameter(stiefel_param_init_val.clone().detach().requires_grad_(True), manifold=manifold)

            # ------------------------------------------------------------------------------
            # apartado 5) Riemannian Optimizer
            # ------------------------------------------------------------------------------
            learning_rate = 1e-3
            optimizer = geoopt.optim.RiemannianSGD([stiefel_param], lr=learning_rate, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-5)
            max_iter = 200000
            tolerance_loss_change = 1e-5
            patience_epochs = 200
            count_patience = 0
            prev_loss_val = float('inf')

            def loss_fn_1d(X_input, M_stiefel):
                b_pt = M_stiefel[:,0]
                t_vec = M_stiefel[:,1]
                Xp_proj = proj_to_geodesic(X_input, t_vec, b_pt)
                d_sq = dist_sphere_batch(X_input, Xp_proj)**2
                return 0.5 * torch.sum(d_sq)

            def loss_fn_2d(X_input, M_stiefel):
                b_pt = M_stiefel[:,0]
                t1_v = M_stiefel[:,1]
                t2_v = M_stiefel[:,2]
                Xp_proj = proj_to_2sphere(X_input, t1_v, t2_v, b_pt)
                d_sq = dist_sphere_batch(X_input, Xp_proj)**2
                return 0.5 * torch.sum(d_sq)

            current_loss_function = loss_fn_1d if gpca_dim == 1 else loss_fn_2d
            
            # ------------------------------------------------------------------------------
            # apartado 6) Optimization loop
            # ------------------------------------------------------------------------------
            print(f"Starting GPCA-{gpca_dim}D optimization for initial_eig_{args.eig_index}...")
            pbar_desc = f"GPCA-{gpca_dim}D (L:{layer}, InitEig:{args.eig_index})"
            with tqdm(range(max_iter), desc=pbar_desc) as pbar:
                for step in pbar:
                    optimizer.zero_grad()
                    loss_val = current_loss_function(data_aug, stiefel_param)
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        print(f"Error: Loss became NaN or Inf at step {step}. Stopping optimization.")
                        break
                    loss_val.backward()
                    optimizer.step()
                    scheduler.step()
                    current_loss_item = loss_val.item()
                    pbar.set_postfix(loss=f"{current_loss_item:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")
                    if abs(prev_loss_val - current_loss_item) < tolerance_loss_change:
                        count_patience += 1
                    else:
                        count_patience = 0
                    if count_patience >= patience_epochs:
                        print(f"Converged at step {step} due to stable loss. Loss = {current_loss_item:.6f}")
                        break
                    prev_loss_val = current_loss_item
            
            # ------------------------------------------------------------------------------
            # apartado 7) 최종 파라미터로 데이터 투영, RSS 계산
            # ------------------------------------------------------------------------------
            final_M_stiefel = stiefel_param.detach().clone()
            Q_final, _ = torch.linalg.qr(final_M_stiefel)
            final_M_stiefel = Q_final[:, :stiefel_cols]

            if gpca_dim == 1:
                base_final = final_M_stiefel[:,0]
                tangent_final = final_M_stiefel[:,1]
                sphere_vec_projected = proj_to_geodesic(data_aug, tangent_final, base_final)
            else: # gpca_dim == 2
                base_final = final_M_stiefel[:,0]
                t1_final = final_M_stiefel[:,1]
                t2_final = final_M_stiefel[:,2]
                sphere_vec_projected = proj_to_2sphere(data_aug, t1_final, t2_final, base_final)

            sphere_vec_projected_np = sphere_vec_projected.cpu().numpy()
            data_aug_np = data_aug.cpu().numpy()
            
            data_aug_np_normalized = data_aug_np / (np.linalg.norm(data_aug_np, axis=1, keepdims=True) + 1e-9)
            sphere_vec_projected_np_normalized = sphere_vec_projected_np / (np.linalg.norm(sphere_vec_projected_np, axis=1, keepdims=True) + 1e-9)
            
            rss_final = np.sum(space_full.metric.squared_dist(data_aug_np_normalized, sphere_vec_projected_np_normalized))
            print(f"Final RSS for tracked trajectory (initial_eig_{args.eig_index}): {rss_final:.4f}")

            # ------------------------------------------------------------------------------
            # (A) 1D GPCA 시각화
            # ------------------------------------------------------------------------------
            actual_epochs_for_plot = []
            valid_vector_indices = [i for i, v in enumerate(tracked_vectors) if v is not None]
            if not valid_vector_indices:
                print("No valid vectors to plot.")
                continue
            
            start_idx = valid_vector_indices[0]
            actual_epochs_for_plot = np.arange(args.start_epoch + start_idx, args.start_epoch + start_idx + l)

            if gpca_dim == 1:
                base_final_np = base_final.cpu().numpy()
                tangent_final_np = tangent_final.cpu().numpy()
                unit_base_final_np = base_final_np / (np.linalg.norm(base_final_np) + 1e-9)
                tangent_at_base = space_full.to_tangent(vector=tangent_final_np, base_point=unit_base_final_np)
                unit_tangent_at_base = tangent_at_base / (np.linalg.norm(tangent_at_base) + 1e-9)
                log_map_vectors = space_full.metric.log(point=sphere_vec_projected_np_normalized, base_point=unit_base_final_np)
                geodesic_parameters = np.sum(log_map_vectors * unit_tangent_at_base, axis=1)

                fm_original_data = FrechetMean(metric=space_full.metric)
                fm_original_data.fit(data_aug_np_normalized)
                total_variance = np.sum(space_full.metric.squared_dist(data_aug_np_normalized, fm_original_data.estimate_))
                
                fit_score_val = 0.0
                if total_variance > 1e-14:
                    fit_score_val = 1.0 - (rss_final / total_variance)
                elif rss_final < 1e-14:
                    fit_score_val = 1.0

                print(f"Fitting Score (1D GPCA): {fit_score_val:.4f}")

                plt.figure(figsize=(7,5))
                plt.plot(actual_epochs_for_plot, geodesic_parameters, marker='.', linestyle='-')
                plt.xlabel(f'Epoch (Actual count: {l})')
                plt.ylabel('Geodesic Parameter (Distance along Geodesic)')
                title_1d = (f'Tracked 1D GPCA: Layer {layer}, InitEig {args.eig_index}\n'
                            f'Epochs {args.start_epoch}-{args.end_epoch} (Used {l} pts)\n'
                            f'RSS={rss_final:.3f}, FitScore={fit_score_val:.3f}')
                plt.title(title_1d, fontsize=10)
                plt.grid(True); plt.tight_layout()
                plt.savefig(output_1d_fig, dpi=300); plt.close()
                print(f"1D GPCA (tracked) figure saved: {output_1d_fig}")
                
                residuals_1d_plot = np.sqrt(space_full.metric.squared_dist(data_aug_np_normalized, sphere_vec_projected_np_normalized))
                plt.figure(figsize=(7,5))
                plt.plot(actual_epochs_for_plot, residuals_1d_plot, marker='.', linestyle='-')
                plt.xlabel(f'Epoch (Actual count: {l})'); plt.ylabel('Residual Magnitude')
                plt.title(f'Residuals vs Epoch (Tracked 1D GPCA)\nLayer {layer}, InitEig {args.eig_index}', fontsize=10)
                plt.grid(True); plt.tight_layout()
                plt.savefig(output_1d_residual_fig, dpi=300); plt.close()
                print(f"1D Residual (tracked) vs Epoch figure saved: {output_1d_residual_fig}")

            # ------------------------------------------------------------------------------
            # (B) 2D GPCA 시각화
            # ------------------------------------------------------------------------------
            elif gpca_dim == 2:
                coords_for_plot_3d = sphere_vec_projected_np_normalized @ final_M_stiefel.cpu().numpy()
                coords_for_plot_3d /= (np.linalg.norm(coords_for_plot_3d, axis=1, keepdims=True) + 1e-9)

                np.savetxt(output_csv, coords_for_plot_3d, fmt='%.8f')
                print(f"2D sphere (tracked) projected data (3D coords for plot) saved: {output_csv}")

                fm_original_data_2d = FrechetMean(space=space_full)
                fm_original_data_2d.fit(data_aug_np_normalized)
                total_variance_2d = np.sum(space_full.metric.squared_dist(data_aug_np_normalized, fm_original_data_2d.estimate_))
                
                fit_score_val_2d = 0.0
                if total_variance_2d > 1e-14:
                    fit_score_val_2d = 1.0 - (rss_final / total_variance_2d)
                elif rss_final < 1e-14:
                    fit_score_val_2d = 1.0
                print(f"Fitting Score (2D GPCA): {fit_score_val_2d:.4f}")

                from mpl_toolkits.mplot3d import Axes3D
                num_plot_pts = coords_for_plot_3d.shape[0]
                color_vals_plot = np.linspace(0, 1, num_plot_pts)

                fig_2d = plt.figure(figsize=(8,8))
                ax_2d = fig_2d.add_subplot(111, projection='3d')
                sc_2d = ax_2d.scatter(coords_for_plot_3d[:,0], coords_for_plot_3d[:,1], coords_for_plot_3d[:,2],
                                      c=color_vals_plot, cmap='hot', alpha=0.8, s=40)
                cbar_2d = plt.colorbar(sc_2d, ax=ax_2d, shrink=0.6, aspect=15, pad=0.1)
                cbar_2d.set_label("Normalized Epoch Progression")
                
                num_cbar_ticks = min(6, num_plot_pts)
                if num_plot_pts > 1:
                    cbar_tick_indices = np.linspace(0, num_plot_pts - 1, num_cbar_ticks, dtype=int)
                    cbar_tick_values_norm = np.linspace(0, 1, num_cbar_ticks)
                    cbar_tick_labels = [f"{actual_epochs_for_plot[idx]}" for idx in cbar_tick_indices]
                    cbar_2d.set_ticks(cbar_tick_values_norm)
                    cbar_2d.set_ticklabels(cbar_tick_labels)
                elif num_plot_pts == 1:
                     cbar_2d.set_ticks([0.5])
                     cbar_2d.set_ticklabels([f"{actual_epochs_for_plot[0]}"])

                ax_2d.set_xlim([-1.1,1.1]); ax_2d.set_ylim([-1.1,1.1]); ax_2d.set_zlim([-1.1,1.1])
                ax_2d.set_box_aspect([1,1,1]); ax_2d.set_xlabel("PC1"); ax_2d.set_ylabel("PC2"); ax_2d.set_zlabel("PC3")
                title_2d_heatmap = (f'Tracked 2D GPCA Projection: Layer {layer}, InitEig {args.eig_index}\n'
                                    f'Epochs {args.start_epoch}-{args.end_epoch} (Used {l} pts)\n'
                                    f'RSS={rss_final:.3f}, FitScore={fit_score_val_2d:.3f}')
                ax_2d.set_title(title_2d_heatmap, fontsize=10)
                u_s, v_s = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
                x_s = np.cos(u_s)*np.sin(v_s); y_s = np.sin(u_s)*np.sin(v_s); z_s = np.cos(v_s)
                ax_2d.plot_wireframe(x_s, y_s, z_s, color='lightblue', alpha=0.5, linewidth=0.5)
                fig_2d.savefig(output_heatmap_fig, dpi=300); plt.close(fig_2d)
                print(f"2D GPCA (tracked) heatmap figure saved: {output_heatmap_fig}")

                residuals_2d_plot = np.sqrt(space_full.metric.squared_dist(data_aug_np_normalized, sphere_vec_projected_np_normalized))
                fig_res_2d = plt.figure(figsize=(8,8))
                ax_res_2d = fig_res_2d.add_subplot(111, projection='3d')
                sc_res_2d = ax_res_2d.scatter(coords_for_plot_3d[:,0], coords_for_plot_3d[:,1], coords_for_plot_3d[:,2],
                                              c=residuals_2d_plot, cmap='hot', alpha=0.8, s=40)
                cb_res_2d = plt.colorbar(sc_res_2d, ax=ax_res_2d, shrink=0.6, aspect=15, pad=0.1)
                cb_res_2d.set_label("Residual Distance")
                ax_res_2d.set_xlim([-1.1,1.1]); ax_res_2d.set_ylim([-1.1,1.1]); ax_res_2d.set_zlim([-1.1,1.1])
                ax_res_2d.set_box_aspect([1,1,1]); ax_res_2d.set_xlabel("PC1"); ax_res_2d.set_ylabel("PC2"); ax_res_2d.set_zlabel("PC3")
                title_2d_resid = (f'Residuals on Tracked 2D GPCA: Layer {layer}, InitEig {args.eig_index}\n'
                                  f'Max Residual: {np.max(residuals_2d_plot):.3f}')
                ax_res_2d.set_title(title_2d_resid, fontsize=10)
                ax_res_2d.plot_wireframe(x_s, y_s, z_s, color='lightblue', alpha=0.5, linewidth=0.5)
                fig_res_2d.savefig(output_residual_heatmap_fig, dpi=300); plt.close(fig_res_2d)
                print(f"2D Residual (tracked) heatmap figure saved: {output_residual_heatmap_fig}")
                
                plt.figure(figsize=(7,5))
                plt.plot(actual_epochs_for_plot, residuals_2d_plot, marker='.', linestyle='-')
                plt.xlabel(f'Epoch (Actual count: {l})'); plt.ylabel('Residual Magnitude')
                plt.title(f'Residuals vs Epoch (Tracked 2D GPCA)\nLayer {layer}, InitEig {args.eig_index}', fontsize=10)
                plt.grid(True); plt.tight_layout()
                plt.savefig(output_2d_residual_fig, dpi=300); 
                # ---------- Plotly-based Interactive Visualization ----------
                plotly_output_html = os.path.join(output_folder, f'tracked_2D_projection_interactive.html')
                plotly_fig = go.Figure()

                # --- 점 투영 좌표 시각화 ---
                plotly_fig.add_trace(go.Scatter3d(
                    x=coords_for_plot_3d[:,0], y=coords_for_plot_3d[:,1], z=coords_for_plot_3d[:,2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color_vals_plot,
                        colorscale='Inferno',
                        opacity=0.8,
                    ),
                    name='Tracked Projection'
                ))

                # --- 구 wireframe (semi-transparent surface) ---
                u_s, v_s = np.mgrid[0:2*np.pi:60j, 0:np.pi:60j]
                x_s = np.cos(u_s)*np.sin(v_s)
                y_s = np.sin(u_s)*np.sin(v_s)
                z_s = np.cos(v_s)

                plotly_fig.add_trace(go.Surface(
                    x=x_s, y=y_s, z=z_s,
                    opacity=0.1,
                    showscale=False,
                    colorscale=[[0, 'gray'], [1, 'gray']],
                    name='Unit Sphere',
                    hoverinfo='skip'
                ))

                # --- 레이아웃 설정 ---
                plotly_fig.update_layout(
                    title=f'Tracked 2D GPCA (Interactive) — Layer {layer}, InitEig {args.eig_index}',
                    scene=dict(
                        xaxis_title='PC1',
                        yaxis_title='PC2',
                        zaxis_title='PC3',
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(x=0.01, y=0.99),
                    coloraxis_colorbar=dict(title='Epoch')
                )

                # --- 저장 ---
                plotly_fig.write_html(plotly_output_html)
                print(f"📦 Interactive HTML plot with sphere saved: {plotly_output_html}")


                plt.close()
                print(f"2D Residual (tracked) vs Epoch figure saved: {output_2d_residual_fig}")

            print(f"✅ Processing complete for: {output_folder}\n")
            
    print("✅ All processing complete for all directories and layers!")

if __name__ == "__main__":
    main()