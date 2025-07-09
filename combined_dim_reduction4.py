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
    args = parser.parse_args()

    # ------------------------------------------------------------------------------
    # 2) Iterate Over Each Directory and Layer
    # ------------------------------------------------------------------------------
    for base_dir in args.dirs:
        for layer in args.layers:
            # -------------------------------------------
            # 결과 저장 폴더 생성
            # -------------------------------------------
            analysis_parent_dir = os.path.join(base_dir, "gpca_tracked_trajectory_results")
            current_analysis_folder_name = f"layer_{layer}_initial_eig_{args.eig_index}_epochs_{args.start_epoch}-{args.end_epoch}"
            output_folder = os.path.join(analysis_parent_dir, current_analysis_folder_name)
            os.makedirs(output_folder, exist_ok=True)

            print(f"\n===== Processing Directory: {base_dir} =====")
            print(f"===== Layer: {layer}, Initial EigIndex: {args.eig_index}, Epochs: {args.start_epoch}-{args.end_epoch} =====")
            print(f"===== Output will be saved to: {output_folder} =====")

            # 파일 경로 세팅 (폴더 기준으로 파일명은 단순화)
            output_csv = os.path.join(output_folder, "tracked_sphere_projection_coords.csv")
            output_heatmap_fig = os.path.join(output_folder, "tracked_2D_projection_heatmap.png")
            output_residual_heatmap_fig = os.path.join(output_folder, "tracked_2D_residual_heatmap.png")
            output_1d_fig = os.path.join(output_folder, "tracked_1D_geodesic_params.png")
            output_1d_residual_fig = os.path.join(output_folder, "tracked_1D_residuals_vs_epoch.png")
            output_2d_residual_fig = os.path.join(output_folder, "tracked_2D_residuals_vs_epoch.png")
            crossing_info_file_path = os.path.join(output_folder, f"sv_trajectory_and_crossing_info_initial_eig{args.eig_index}.txt")

            # ----------------------------------------------------------------------
            #  apartado 1) 모든 에포크의 전체 Vh 행렬 로드 및 부호 정렬 (시간 축 기준)
            # ----------------------------------------------------------------------
            all_epochs_Vh_raw = [] # 부호 정렬 전 Vh 행렬들
            num_epochs_to_process = args.end_epoch - args.start_epoch + 1

            for ep_num in range(args.start_epoch, args.end_epoch + 1):
                npz_path = os.path.join(base_dir, f"eigenvectors/epoch{str(ep_num)}.npz") # NPZ 파일명 규칙 확인!
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

            # 전체 Vh 행렬들에 대해 시간 축으로 부호 정렬 수행
            # 각 singular vector (행)의 궤적을 따라 부호를 맞춤
            aligned_all_epochs_Vh = [None] * len(all_epochs_Vh_raw)
            first_valid_vh = next((vh for vh in all_epochs_Vh_raw if vh is not None), None)
            if first_valid_vh is None:
                print(f"❌ No valid Vh matrices found after loading for layer {layer}. Skipping.")
                continue
            
            num_singular_vectors = first_valid_vh.shape[0]

            for sv_j in range(num_singular_vectors): # 각 singular vector 인덱스 j에 대해
                # sv_j 번째 singular vector의 시간적 궤적 (None 제외)
                trajectory_sv_j = []
                epoch_indices_for_sv_j = [] # None이 아닌 Vh의 원래 all_epochs_Vh_raw에서의 인덱스

                for i, vh_matrix_at_epoch_i in enumerate(all_epochs_Vh_raw):
                    if vh_matrix_at_epoch_i is not None and sv_j < vh_matrix_at_epoch_i.shape[0]:
                        trajectory_sv_j.append(vh_matrix_at_epoch_i[sv_j])
                        epoch_indices_for_sv_j.append(i)
                
                if not trajectory_sv_j:
                    continue # 이 singular vector 인덱스에 대한 데이터가 없음

                trajectory_sv_j_np = np.array(trajectory_sv_j)

                # 시간 축으로 부호 정렬
                for t_traj in range(1, len(trajectory_sv_j_np)):
                    if np.dot(trajectory_sv_j_np[t_traj], trajectory_sv_j_np[t_traj-1]) < 0:
                        trajectory_sv_j_np[t_traj] *= -1
                
                # 정렬된 벡터를 aligned_all_epochs_Vh에 다시 채워넣기
                for i_traj, original_raw_idx in enumerate(epoch_indices_for_sv_j):
                    if aligned_all_epochs_Vh[original_raw_idx] is None and all_epochs_Vh_raw[original_raw_idx] is not None:
                        # deepcopy가 더 안전할 수 있으나, 일단 shape이 같은 새 배열로 초기화
                        aligned_all_epochs_Vh[original_raw_idx] = np.zeros_like(all_epochs_Vh_raw[original_raw_idx])
                    
                    if aligned_all_epochs_Vh[original_raw_idx] is not None: # None이 아닐 때만 할당
                         aligned_all_epochs_Vh[original_raw_idx][sv_j] = trajectory_sv_j_np[i_traj]
            
            # ----------------------------------------------------------------------
            # apartado 2) Trajectory 추적 및 해당 벡터 추출
            # ----------------------------------------------------------------------
            tracked_vectors = [] # 최종적으로 GPCA에 사용될, 추적된 벡터들의 시계열
            crossing_info_details = [] # (epoch_t, epoch_t+1, old_idx, new_idx, similarity) 저장
            
            current_tracked_idx = args.eig_index # 추적 시작 인덱스
            max_sv_idx_overall = num_singular_vectors - 1 # 가능한 최대 SV 인덱스
            
            prev_change_vector = None

            for t in range(num_epochs_to_process): # 0부터 (len-1)까지의 인덱스 (aligned_all_epochs_Vh 기준)
                Vh_t = aligned_all_epochs_Vh[t]

                if Vh_t is None:
                    print(f"Warning: Missing Vh data for epoch {args.start_epoch + t}, cannot append to trajectory.")
                    tracked_vectors.append(None) # 또는 이전 벡터를 반복하거나, 평균 등으로 채울 수도 있음
                    # 다음 스텝에서의 current_tracked_idx는 이전 값을 유지
                    if t < num_epochs_to_process -1 : # 마지막 에폭이 아니면
                        crossing_info_details.append(
                            (args.start_epoch + t, args.start_epoch + t + 1, current_tracked_idx, current_tracked_idx, "N/A (Missing current Vh)")
                        )
                    continue

                # 현재 추적 중인 인덱스가 Vh_t의 유효 범위를 벗어나는지 확인 (이론상 발생 안해야 함)
                if not (0 <= current_tracked_idx < Vh_t.shape[0]):
                    print(f"Error: current_tracked_idx {current_tracked_idx} is out of bounds for Vh_t at epoch {args.start_epoch + t}. Shape: {Vh_t.shape}. Stopping trajectory for this layer.")
                    break 
                
                vector_to_add = Vh_t[current_tracked_idx]
                tracked_vectors.append(vector_to_add)

                # 다음 스텝(t+1)으로 넘어갈 때 current_tracked_idx를 업데이트 (Crossing 확인)
                if t < num_epochs_to_process - 1: # 마지막 epoch이 아니라면 다음 epoch을 고려
                    Vh_t_plus_1 = aligned_all_epochs_Vh[t+1]
                    if Vh_t_plus_1 is None:
                        print(f"Warning: Missing Vh data for next epoch {args.start_epoch + t + 1}. Assuming no change in tracked index.")
                        crossing_info_details.append(
                            (args.start_epoch + t, args.start_epoch + t + 1, current_tracked_idx, current_tracked_idx, "N/A (Missing next Vh)")
                        )
                        # prev_change_vector는 그대로 유지 (다음 유효 데이터에서 계산 재개)
                        continue

                    v_current_at_t = vector_to_add # Vh_t[current_tracked_idx]
                    
                    # 비교 대상 인덱스 후보
                    candidate_next_indices = []
                    if current_tracked_idx > 0: candidate_next_indices.append(current_tracked_idx - 1)
                    candidate_next_indices.append(current_tracked_idx)
                    if current_tracked_idx < max_sv_idx_overall: candidate_next_indices.append(current_tracked_idx + 1)
                    
                    best_next_idx = current_tracked_idx
                    max_similarity = -2.0 # 코사인 유사도는 -1 ~ 1 범위

                    # --- 여기가 핵심 로직 ---
                    if prev_change_vector is None:
                        # [CASE 1: 첫 번째 변화] 이전 변화량이 없으므로, 기존 방식(벡터 자체의 유사도)으로 추적
                        for next_idx_candidate in candidate_next_indices:
                            if not (0 <= next_idx_candidate < Vh_t_plus_1.shape[0]): continue
                            v_candidate_at_t_plus_1 = Vh_t_plus_1[next_idx_candidate]
                            similarity = np.dot(v_current_at_t, v_candidate_at_t_plus_1) / (np.linalg.norm(v_current_at_t) * np.linalg.norm(v_candidate_at_t_plus_1) + 1e-9)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_next_idx = next_idx_candidate
                    else:
                        # [CASE 2: 두 번째 변화부터] 변화량 벡터 간의 유사도로 추적
                        norm_prev_change = np.linalg.norm(prev_change_vector)
                        if norm_prev_change < 1e-9: # 이전 변화량이 거의 0이면 추적 의미 없음
                             # 이 경우, 가장 가까운 벡터를 선택하는 기존 방식으로 일시적 전환 가능
                            best_next_idx = current_tracked_idx # 간단하게는 인덱스 유지
                        else:
                            normalized_prev_change = prev_change_vector / norm_prev_change
                            for next_idx_candidate in candidate_next_indices:
                                if not (0 <= next_idx_candidate < Vh_t_plus_1.shape[0]): continue
                                v_candidate_at_t_plus_1 = Vh_t_plus_1[next_idx_candidate]
                                
                                # 현재 후보 변화량 벡터 계산
                                current_change_vector_cand = v_candidate_at_t_plus_1 - v_current_at_t
                                norm_current_change_cand = np.linalg.norm(current_change_vector_cand)
                                
                                if norm_current_change_cand < 1e-9:
                                    similarity = -1.0 # 변화가 없으면 유사도 낮게 처리
                                else:
                                    normalized_current_change_cand = current_change_vector_cand / norm_current_change_cand
                                    similarity = np.dot(normalized_prev_change, normalized_current_change_cand)
                                
                                if similarity > max_similarity:
                                    max_similarity = similarity
                                    best_next_idx = next_idx_candidate

                    # --- 로직 종료 ---
                    
                    # 다음 이터레이션을 위해 'prev_change_vector' 업데이트
                    v_best_next = Vh_t_plus_1[best_next_idx]
                    prev_change_vector = v_best_next - v_current_at_t

                    # 결과 기록 및 인덱스 업데이트
                    crossing_info_details.append(
                        (args.start_epoch + t, args.start_epoch + t + 1, current_tracked_idx, best_next_idx, f"{max_similarity:.4f}")
                    )
                    current_tracked_idx = best_next_idx
            
            # None 값을 가진 tracked_vectors 처리 (예: 이전 값으로 채우거나, 평균 등)
            # 여기서는 간단히 None인 항목은 제외하고 GPCA 분석 진행
            final_tracked_vectors_for_gpca = [v for v in tracked_vectors if v is not None]
            if not final_tracked_vectors_for_gpca:
                print(f"❌ No valid vectors in the tracked trajectory for layer {layer}, initial_eig_{args.eig_index}. Skipping GPCA.")
                # crossing 정보는 저장
                with open(crossing_info_file_path, 'w') as f_cross:
                    f_cross.write(f"Singular Vector Trajectory Tracking & Crossing Information\n")
                    f_cross.write(f"Layer: {layer}, Initial SV Index: {args.eig_index}\n")
                    f_cross.write(f"Epoch Range: {args.start_epoch}-{args.end_epoch}\n")
                    f_cross.write("="*70 + "\n")
                    f_cross.write("Epoch_t -> Epoch_t+1 : Old_SV_Index_at_t -> New_SV_Index_at_t+1 (Similarity)\n")
                    f_cross.write("="*70 + "\n")
                    for ct, nt, oi, ni, sim in crossing_info_details:
                        f_cross.write(f"{ct:03d} -> {nt:03d} : SV {oi} -> SV {ni} ({sim})\n")
                    if not final_tracked_vectors_for_gpca:
                         f_cross.write("\nNo valid vectors remained in the trajectory for GPCA analysis.\n")
                print(f"Crossing and trajectory information saved to: {crossing_info_file_path}")
                continue # 다음 레이어로

            eigv = np.array(final_tracked_vectors_for_gpca) # GPCA에 사용할 최종 벡터 배열
            l, p = eigv.shape # l은 실제 사용된 벡터 수
            if l < 2 : # GPCA를 의미있게 수행하려면 최소 2개의 데이터 포인트 필요
                print(f"❌ Only {l} valid vector(s) in the tracked trajectory for layer {layer}, initial_eig_{args.eig_index}. Skipping GPCA.")
                # crossing 정보는 위에서 이미 저장됨
                continue

            print(f"Loaded {l} vectors (after tracking) of dimension {p} for layer {layer}, initial_eig_{args.eig_index}")
            
            # crossing 정보 저장
            with open(crossing_info_file_path, 'w') as f_cross:
                f_cross.write(f"Singular Vector Trajectory Tracking & Crossing Information\n")
                f_cross.write(f"Layer: {layer}, Initial SV Index: {args.eig_index}\n")
                f_cross.write(f"Epoch Range: {args.start_epoch}-{args.end_epoch}\n")
                f_cross.write(f"Number of vectors in final trajectory for GPCA: {l}\n")
                f_cross.write("="*70 + "\n")
                f_cross.write("Epoch_t -> Epoch_t+1 : SV_Index_at_t -> SV_Index_at_t+1 (Similarity with t's vector)\n")
                f_cross.write("="*70 + "\n")
                for ct, nt, oi, ni, sim_val in crossing_info_details:
                    actual_tracked_idx_at_t_plus_1 = ni # 이 값이 다음 current_tracked_idx가 됨
                    f_cross.write(f"{ct:03d} -> {nt:03d} : SV {oi} -> SV {actual_tracked_idx_at_t_plus_1} (Sim: {sim_val})\n")
            print(f"Crossing and trajectory information saved to: {crossing_info_file_path}")

            # -------------------------------------------
            # ✅ CUDA 설정 (이후는 기존 GPCA 코드와 거의 동일)
            # -------------------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device:", device)

            data = torch.from_numpy(eigv).float().to(device) # eigv는 이미 추적된 벡터들

            # -------------------------------------------
            # apartado 2) 방향(부호) 정규화 (GPCA 입력 데이터에 대해)
            # -------------------------------------------
            # aligned_all_epochs_Vh에서 이미 시간축으로 정렬했지만,
            # GPCA 입력인 eigv (추적된 궤적)에 대해 한 번 더 수행하는 것이 안전할 수 있음
            # (특히 None 값 처리 등으로 인해 순서가 바뀐 경우 등)
            # 하지만 위에서 이미 시간축 정렬된 벡터들을 순서대로 가져왔으므로, 이 단계의 부호 정렬은
            # eigv 내의 인접 벡터간 부호 일관성을 다시 한번 맞추는 역할
            
            space_full = Hypersphere(dim=p-1) # p는 벡터 차원

            def sphere_dist_torch(x, y): # PyTorch용
                nx = torch.norm(x)
                ny = torch.norm(y)
                inner = torch.dot(x, y) / (nx*ny + 1e-14)
                inner_clamped = torch.clamp(inner, -1.0, 1.0)
                return torch.arccos(inner_clamped)

            data_aug = data.clone() # GPCA에 사용할 데이터 (부호 정렬 후)
            if l > 1: # 벡터가 2개 이상일 때만 의미 있음
                for _ in range(3): # 안정화를 위해 여러 번 반복
                    for i in range(l - 1):
                        # data_aug[i+1]과 data_aug[i] 비교
                        dist_pos = sphere_dist_torch(data_aug[i+1], data_aug[i])
                        dist_neg = sphere_dist_torch(-data_aug[i+1], data_aug[i])
                        if dist_neg < dist_pos:
                            data_aug[i+1] *= -1
            
            # -------------------------------------------
            # apartado 3) 기하학 연산 함수 (기존과 동일)
            # -------------------------------------------
            def exp_map(v_tangent, p_base):
                norm_v = torch.norm(v_tangent)
                if norm_v < 1e-14:
                    return p_base
                return torch.cos(norm_v)*p_base + torch.sin(norm_v)*(v_tangent/norm_v)

            def dist_sphere_batch(A_batch, B_batch): # Batchwise distance
                norm_a = torch.norm(A_batch, dim=1, keepdim=True)
                norm_b = torch.norm(B_batch, dim=1, keepdim=True)
                inner = torch.sum(A_batch*B_batch, dim=1)
                cos_ = inner / (norm_a.squeeze(-1)*norm_b.squeeze(-1) + 1e-14)
                cos_ = torch.clamp(cos_, -1.0, 1.0)
                return torch.arccos(cos_)

            # (A) 1D GPCA 투영 함수 (기존과 동일, dist_sphere_batch 사용하도록 수정 가능)
            def proj_to_geodesic(X_batch, tangent_vec, base_pt):
                norm_t = torch.norm(tangent_vec)
                if norm_t < 1e-14:
                    return base_pt.unsqueeze(0).repeat(X_batch.shape[0], 1)

                dot_xt = torch.sum(X_batch * tangent_vec.unsqueeze(0), dim=1)
                dot_xb = torch.sum(X_batch * base_pt.unsqueeze(0), dim=1)
                # factor는 각 X_batch의 점이 지오데식 상에 투영될 때의 "각도 파라미터"
                factor = torch.atan2(dot_xt, dot_xb) / (norm_t + 1e-14) # atan2 결과는 -pi ~ pi

                cos_f = torch.cos(factor).unsqueeze(1)
                sin_f = torch.sin(factor).unsqueeze(1)
                unit_t = tangent_vec / norm_t

                proj = cos_f * base_pt.unsqueeze(0) + sin_f * unit_t.unsqueeze(0)
                
                # 투영된 점이 원래 점 X_batch에 더 가까운 방향인지 확인 (antipodal ambiguity)
                # dist_p = dist_sphere_batch(X_batch, proj)
                # dist_m = dist_sphere_batch(X_batch, -proj) # -proj는 antipodal point
                # mask = dist_m < dist_p
                # proj[mask] = -proj[mask] # 더 가까운 쪽으로 플립
                return proj # Stiefel 최적화 과정에서 부호는 학습될 수 있음

            # (B) 2D GPCA 투영 함수 (기존과 동일, dist_sphere_batch 사용하도록 수정 가능)
            def proj_to_2sphere(X_batch, t1_vec, t2_vec, base_pt):
                # base_pt, t1_vec, t2_vec은 Stiefel manifold에서 온 직교하는 벡터들 (p차원)
                # 이들이 정의하는 3D 부분 공간 (선형)에 X_batch를 먼저 투영하고, 그 결과를 normalize.
                # M_sub = torch.stack([base_pt, t1_vec, t2_vec], dim=1) # (p, 3) 직교 행렬
                # X_projected_to_subspace = X_batch @ M_sub @ M_sub.T # (N, p) @ (p,3) @ (3,p) -> (N,p)
                
                # 더 정확하게는, exp_map을 사용한 기존 방식이 geodesic plane을 정의
                p1_sphere = base_pt
                p2_sphere = exp_map(t1_vec, base_pt)
                p3_sphere = exp_map(t2_vec, base_pt) # t1, t2는 base_pt에서의 탄젠트 벡터로 해석

                A_plane_basis = torch.stack([p1_sphere, p2_sphere, p3_sphere], dim=1) # (p,3)
                
                # A_plane_basis의 열들이 선형적으로 독립적이어야 함
                try:
                    # Robust projection onto column space of A_plane_basis
                    Q_A, _ = torch.linalg.qr(A_plane_basis) # Q_A is (p, k) where k <= 3
                    proj_matrix_on_plane = Q_A @ Q_A.T # (p,p)
                    Xp_on_plane = X_batch @ proj_matrix_on_plane # (N,p)
                except Exception as e: # e.g. A_plane_basis is rank deficient
                    print(f"Warning: QR decomposition failed in proj_to_2sphere due to A_plane_basis rank: {e}. Using pseudo-inverse.")
                    # Fallback to pseudo-inverse if QR fails (e.g. t1, t2, base are not well-conditioned)
                    pinv_A_T_A = torch.linalg.pinv(A_plane_basis.T @ A_plane_basis)
                    proj_matrix_on_plane = A_plane_basis @ pinv_A_T_A @ A_plane_basis.T
                    Xp_on_plane = (proj_matrix_on_plane @ X_batch.T).T


                row_norm = torch.norm(Xp_on_plane, dim=1, keepdim=True)
                Xp_normalized = Xp_on_plane / torch.clamp(row_norm, min=1e-14)
                return Xp_normalized

            # ------------------------------------------------------------------------------
            # apartado 4) Stiefel Manifold 파라미터 생성 (기존과 동일)
            # ------------------------------------------------------------------------------
            gpca_dim = args.gpca_dim
            stiefel_cols = gpca_dim + 1 # Base point + gpca_dim tangent vectors

            manifold = Stiefel() # geoopt.manifolds.Stiefel()
            # Initialize parameter on the manifold
            init_rand_matrix = torch.randn(p, stiefel_cols, device=device)
            q_init, _ = torch.linalg.qr(init_rand_matrix)
            stiefel_param_init_val = q_init[:, :stiefel_cols] # Ensure correct shape and orthonormality
            
            stiefel_param = ManifoldParameter(stiefel_param_init_val.clone().detach().requires_grad_(True), manifold=manifold)

            # ------------------------------------------------------------------------------
            # apartado 5) Riemannian Optimizer (기존과 동일)
            # ------------------------------------------------------------------------------
            learning_rate = 1e-3
            optimizer = geoopt.optim.RiemannianSGD([stiefel_param], lr=learning_rate, momentum=0.9) # weight_decay는 RAdam 등에서 더 일반적
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-5) # Slightly higher eta_min

            max_iter = 200000 # 늘릴 수 있음
            tolerance_loss_change = 1e-5 # 손실 변화량 기준
            patience_epochs = 200 # 연속적으로 손실 변화가 tolerance 미만일 경우 중단

            count_patience = 0
            prev_loss_val = float('inf')

            # Loss 함수 (dist_sphere_batch 사용)
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
            # apartado 6) Optimization loop (기존과 유사, 수렴 조건 및 로깅 개선)
            # ------------------------------------------------------------------------------
            print(f"Starting GPCA-{gpca_dim}D optimization for initial_eig_{args.eig_index}...")
            pbar_desc = f"GPCA-{gpca_dim}D (L:{layer}, InitEig:{args.eig_index})"
            with tqdm(range(max_iter), desc=pbar_desc) as pbar:
                for step in pbar:
                    optimizer.zero_grad()
                    loss_val = current_loss_function(data_aug, stiefel_param) # data_aug 사용!
                    
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        print(f"Error: Loss became NaN or Inf at step {step}. Stopping optimization.")
                        break
                    
                    loss_val.backward()
                    optimizer.step() # Riemannian step, should keep param on manifold
                    scheduler.step()

                    current_loss_item = loss_val.item()
                    pbar.set_postfix(loss=f"{current_loss_item:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

                    # 수렴 조건 체크 (상대적 변화량 또는 절대적 변화량)
                    if abs(prev_loss_val - current_loss_item) < tolerance_loss_change: #  * prev_loss_val (for relative)
                        count_patience += 1
                    else:
                        count_patience = 0 # 리셋
                    
                    if count_patience >= patience_epochs:
                        print(f"Converged at step {step} due to stable loss. Loss = {current_loss_item:.6f}")
                        break
                    
                    prev_loss_val = current_loss_item
            
            # ------------------------------------------------------------------------------
            # apartado 7) 최종 파라미터로 데이터 투영, RSS 계산 (기존과 동일)
            # ------------------------------------------------------------------------------
            final_M_stiefel = stiefel_param.detach().clone()
            # Stiefel 파라미터가 manifold 위에 있도록 QR 분해로 한 번 더 정규화 (geoopt가 이미 하지만, 안정성 위해)
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

            sphere_vec_projected_np = sphere_vec_projected.cpu().numpy() # 투영된 결과 (N, p)
            data_aug_np = data_aug.cpu().numpy() # GPCA 입력 데이터 (N, p)

            # RSS 계산 시 geomstats의 Hypersphere metric 사용
            # 입력 벡터들은 단위 구 위에 있어야 함
            data_aug_np_normalized = data_aug_np / (np.linalg.norm(data_aug_np, axis=1, keepdims=True) + 1e-9)
            sphere_vec_projected_np_normalized = sphere_vec_projected_np / (np.linalg.norm(sphere_vec_projected_np, axis=1, keepdims=True) + 1e-9)
            
            rss_final = np.sum(space_full.metric.squared_dist(data_aug_np_normalized, sphere_vec_projected_np_normalized))
            print(f"Final RSS for tracked trajectory (initial_eig_{args.eig_index}): {rss_final:.4f}")

            # ------------------------------------------------------------------------------
            # (A) 1D GPCA 시각화: 지오데식 파라미터 vs Epoch (기존과 동일한 로직, 입력 데이터만 다름)
            # ------------------------------------------------------------------------------
            actual_epochs_for_plot = np.arange(args.start_epoch, args.start_epoch + l) # l은 실제 사용된 벡터 수

            if gpca_dim == 1:
                base_final_np = base_final.cpu().numpy()
                tangent_final_np = tangent_final.cpu().numpy()

                # Geomstats를 사용하여 지오데식 파라미터 계산
                # sphere_vec_projected_np_normalized : 지오데식 위의 점들
                # base_final_np : 지오데식의 기준점
                # tangent_final_np : 지오데식의 방향 (base_final_np에서)
                
                # tangent_final_np를 base_final_np에서의 탄젠트 벡터로 만들고 정규화
                unit_base_final_np = base_final_np / (np.linalg.norm(base_final_np) + 1e-9)
                tangent_at_base = space_full.to_tangent(vector=tangent_final_np, base_point=unit_base_final_np)
                unit_tangent_at_base = tangent_at_base / (np.linalg.norm(tangent_at_base) + 1e-9)

                log_map_vectors = space_full.metric.log(point=sphere_vec_projected_np_normalized, base_point=unit_base_final_np)
                geodesic_parameters = np.sum(log_map_vectors * unit_tangent_at_base, axis=1) # 내적

                # Fitting score 계산 (RSS / Total Variance)
                fm_original_data = FrechetMean(metric=space_full.metric)
                fm_original_data.fit(data_aug_np_normalized) # 원본 데이터(부호정렬된 추적궤적)의 프레셰 평균
                total_variance = np.sum(space_full.metric.squared_dist(data_aug_np_normalized, fm_original_data.estimate_))
                
                fit_score_val = 0.0
                if total_variance > 1e-14:
                    fit_score_val = 1.0 - (rss_final / total_variance)
                elif rss_final < 1e-14: # 총 분산도 0이고 RSS도 0이면 완벽히 설명
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
            # (B) 2D GPCA 시각화 (기존과 동일한 로직, 입력 데이터만 다름)
            # ------------------------------------------------------------------------------
            elif gpca_dim == 2: # else 로 해도 무방 (gpca_dim이 1 또는 2이므로)
                # 투영된 데이터 sphere_vec_projected_np_normalized 를 3D 공간에 시각화하기 위한 좌표 변환
                # final_M_stiefel의 열벡터 (base, t1, t2)는 p차원 공간에서의 직교 기저.
                # 이 기저를 사용하여 sphere_vec_projected_np_normalized를 3D 좌표로 변환.
                coords_for_plot_3d = sphere_vec_projected_np_normalized @ final_M_stiefel.cpu().numpy() # (l, p) @ (p, 3) -> (l, 3)
                coords_for_plot_3d /= (np.linalg.norm(coords_for_plot_3d, axis=1, keepdims=True) + 1e-9) # 정규화

                np.savetxt(output_csv, coords_for_plot_3d, fmt='%.8f')
                print(f"2D sphere (tracked) projected data (3D coords for plot) saved: {output_csv}")

                # fm_original_data_2d = FrechetMean(metric=space_full.metric)
                # fm_original_data_2d.fit(data_aug_np_normalized)
                # total_variance_2d = np.sum(space_full.metric.squared_dist(data_aug_np_normalized, fm_original_data_2d.estimate_))
                fm_original_data_2d = FrechetMean(space=space_full) # 수정된 코드
                fm_original_data_2d.fit(data_aug_np_normalized) # 원본 데이터(부호정렬된 추적궤적)의 프레셰 평균
                total_variance_2d = np.sum(space_full.metric.squared_dist(data_aug_np_normalized, fm_original_data_2d.estimate_))
                
                fit_score_val_2d = 0.0
                if total_variance_2d > 1e-14:
                    fit_score_val_2d = 1.0 - (rss_final / total_variance_2d)
                elif rss_final < 1e-14:
                    fit_score_val_2d = 1.0
                print(f"Fitting Score (2D GPCA): {fit_score_val_2d:.4f}")

                from mpl_toolkits.mplot3d import Axes3D
                num_plot_pts = coords_for_plot_3d.shape[0]
                color_vals_plot = np.linspace(0, 1, num_plot_pts) # Epoch 진행에 따른 색상

                # Heatmap by epoch index
                fig_2d = plt.figure(figsize=(8,8))
                ax_2d = fig_2d.add_subplot(111, projection='3d')
                sc_2d = ax_2d.scatter(coords_for_plot_3d[:,0], coords_for_plot_3d[:,1], coords_for_plot_3d[:,2],
                                      c=color_vals_plot, cmap='viridis', alpha=0.8, s=40)
                cbar_2d = plt.colorbar(sc_2d, ax=ax_2d, shrink=0.6, aspect=15, pad=0.1)
                cbar_2d.set_label("Normalized Epoch Progression")
                
                num_cbar_ticks = min(6, num_plot_pts) # 점 개수보다 많지 않게
                if num_plot_pts > 1:
                    cbar_tick_indices = np.linspace(0, num_plot_pts - 1, num_cbar_ticks, dtype=int)
                    cbar_tick_values_norm = np.linspace(0, 1, num_cbar_ticks) # Normalized tick positions
                    # 실제 epoch 번호로 레이블링 (actual_epochs_for_plot 사용)
                    cbar_tick_labels = [f"{actual_epochs_for_plot[idx]}" for idx in cbar_tick_indices]
                    cbar_2d.set_ticks(cbar_tick_values_norm)
                    cbar_2d.set_ticklabels(cbar_tick_labels)
                elif num_plot_pts == 1: # 점이 하나일 때
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
                ax_2d.plot_wireframe(x_s, y_s, z_s, color='gray', alpha=0.3, linewidth=0.5)
                fig_2d.savefig(output_heatmap_fig, dpi=300); plt.close(fig_2d)
                print(f"2D GPCA (tracked) heatmap figure saved: {output_heatmap_fig}")

                # Residual Heatmap
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
                ax_res_2d.plot_wireframe(x_s, y_s, z_s, color='gray', alpha=0.3, linewidth=0.5)
                fig_res_2d.savefig(output_residual_heatmap_fig, dpi=300); plt.close(fig_res_2d)
                print(f"2D Residual (tracked) heatmap figure saved: {output_residual_heatmap_fig}")
                
                # Residual Magnitude vs Epoch (2D)
                plt.figure(figsize=(7,5))
                plt.plot(actual_epochs_for_plot, residuals_2d_plot, marker='.', linestyle='-')
                plt.xlabel(f'Epoch (Actual count: {l})'); plt.ylabel('Residual Magnitude')
                plt.title(f'Residuals vs Epoch (Tracked 2D GPCA)\nLayer {layer}, InitEig {args.eig_index}', fontsize=10)
                plt.grid(True); plt.tight_layout()
                plt.savefig(output_2d_residual_fig, dpi=300); plt.close()
                print(f"2D Residual (tracked) vs Epoch figure saved: {output_2d_residual_fig}")

            print(f"✅ Processing complete for: {output_folder}\n")
            
    print("✅ All processing complete for all directories and layers!")

if __name__ == "__main__":
    main()