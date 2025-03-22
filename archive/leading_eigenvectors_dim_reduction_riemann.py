import os
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

# Geoopt (리만 기하학 최적화 라이브러리)
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import Stiefel

torch.manual_seed(42)  # 항상 동일한 초기값 사용
np.random.seed(42)

# -------------------------------------------
# Config: 디렉토리/파일 관련
# -------------------------------------------
base_dir = 'runs/alex_seed100_batch64_sgd_lr0.01_epochs100'
base_dir = './'


csv_file = os.path.join(base_dir, 'eigenvectors/fc3.csv')
csv_file = os.path.join(base_dir, 'fc1.csv')
output_csv = os.path.join(base_dir, "fc1_sphere.csv")
output_fig = os.path.join(base_dir, "fc1_sphere.png")
output_heatmap_fig = os.path.join(base_dir, "fc1_sphere_heatmap.png")

# -------------------------------------------
# 1) CSV에서 고유벡터 로드
# -------------------------------------------
df = pd.read_csv(csv_file, sep='\s+', header=None)
df = df.iloc[:100,:]
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
# 2) 방향(부호) 정규화
#    (이전 코드와 동일)
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
# 3) 기하 연산 함수 정의
#    (이전 코드와 동일)
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
    """
    v를 p에 대한 접공간으로 사영
    p는 이미 단위 벡터라고 가정
    """
    coef = torch.dot(v, p) / (torch.dot(p, p) + 1e-14)
    return v - coef * p

def proj_to_2sphere(X, tangent_1, tangent_2, base_point):
    """
    base_point, tangent_1, tangent_2가 만드는 2차원 스피어로
    X (shape: [l, p])를 투영
    """
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
# Stiefel(p, 3) 은 R^{p x 3}에서 각 열 벡터가 상호 직교, 단위 노름을 갖도록 하는 다양체
#  => 첫 번째 열: base_point (스피어 상의 점)
#     두 번째/세 번째 열: base_point 접공간에 놓인 서로 직교하는 벡터

manifold = Stiefel()  # 기본 Stiefel(p,3)로 사용
# 초기값을 QR 분해로 정규직교화
init_mat = torch.randn(p, 3, device=device)
Q, _ = torch.linalg.qr(init_mat)  # (p,3)
stiefel_param = ManifoldParameter(Q, manifold=manifold)

# 옵티마이저: Riemannian AdamW
learning_rate = 1e-3
optimizer = geoopt.optim.RiemannianAdam([stiefel_param], lr=learning_rate)

# 스케줄러
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-4)

max_iter = 1000000
tolerance = 1e-30
prev_loss = float('inf')

# -------------------------------------------
# 5) Loss 함수 정의 & 학습 루프
# -------------------------------------------
def loss_fn(X, M):
    """
    M: shape (p, 3) (Stiefel 다양체)
       M[:, 0] = base_point
       M[:, 1], M[:, 2] = tangent vectors
    """
    base_point = M[:, 0]  # shape (p,)
    t1 = M[:, 1]
    t2 = M[:, 2]

    # X를 (base_point, t1, t2)가 이루는 스피어에 투영
    X_proj = proj_to_2sphere(X, t1, t2, base_point)
    dist_sq = dist_sphere(X, X_proj) ** 2

    # 기존 코드에서는 base_point 정규화 penalize 했지만,
    # 이제 Stiefel로 인해 자동으로 || base_point||=1, t1⊥base_point 등 보장
    # => penalty 불필요
    return 0.5 * torch.sum(dist_sq)

print("Start Riemannian optimization...")
with tqdm(range(max_iter), desc="Training", unit="step") as pbar:
    for step in pbar:
        optimizer.zero_grad()
        loss_value = loss_fn(data_aug, stiefel_param)
        loss_value.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss_value.item()
        pbar.set_postfix(loss=f"{current_loss:.6f}")

        # 간단한 수렴 기준
        if current_loss < 0.9 and abs(prev_loss - current_loss) < tolerance:
            print(f"✅ Converged at step {step} with loss={current_loss:.6f}")
            break
        prev_loss = current_loss

print("Final Loss:", prev_loss)

# -------------------------------------------
# 6) 최종 파라미터로 2-Sphere 투영 & 후처리
# -------------------------------------------
with torch.no_grad():
    final_M = stiefel_param.detach().clone()  # shape: (p, 3)
    base_point = final_M[:, 0]               # (p,)
    t1 = final_M[:, 1]
    t2 = final_M[:, 2]

    sphere_vec = proj_to_2sphere(data_aug, t1, t2, base_point)
    sphere_vec_np = sphere_vec.cpu().numpy()

# -------------------------------------------
# 7) RSS (Residual Sum of Squares) 계산
#    - geomstats의 Hypersphere(dim=p-1)로 측정
# -------------------------------------------
space_full = Hypersphere(dim=p-1)
data_np = data_aug.cpu().numpy()

rss = np.sum(space_full.metric.squared_dist(data_np, sphere_vec_np))
print("RSS:", rss)

# -------------------------------------------
# 8) Dimension Reduction (QR) & 저장
#    => 최종 3D 좌표로 매핑해 S^2 상의 포인트로 사용
# -------------------------------------------
#   (기존 코드처럼 intercept, coef1, coef2 대신
#    stiefel의 첫/둘째/셋째 열 벡터를 basis로 삼음)
base_point_ = space_full.projection(base_point.cpu().numpy())
point1_ = space_full.projection(t1.cpu().numpy())
point2_ = space_full.projection(t2.cpu().numpy())

# (p,3)
basis = np.vstack((base_point_, point1_, point2_)).T  
Q_, _ = np.linalg.qr(basis)   # (p, p), 앞 3개 열이 직교기저

sphere_data = sphere_vec_np @ Q_  # (l, p)
sphere_data = np.array(sphere_data, dtype=np.float64)

# 정규화
norms = gs.linalg.norm(sphere_data, axis=1, keepdims=True)
sphere_data /= norms

# CSV로 저장
np.savetxt(output_csv, sphere_data)
print("Saved sphere data to:", output_csv)

# -------------------------------------------
# 9) Frechet Mean & 분산 계산 (Geomstats)
#    => 최종적으로 dim=2로 처리(S^2)
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
fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(sphere_data, space='S2', color='black', alpha=0.7)
ax.set_box_aspect([1, 1, 1])
plt.title(f'Fitting score: {fitting_score:.4f} RSS: {rss:.4f}')

fig.savefig(output_fig, dpi=300)
plt.show()
print("Figure saved to:", output_fig)


################################################
# 추가 스캐터 플롯(heatmap 느낌)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# Load the processed data
sphere_data = np.loadtxt(output_csv)

# Generate a color gradient for visualization
num_points = sphere_data.shape[0]
colors = np.linspace(0, 1, num_points)  # Color gradient from start to end

# Create a scatter plot for visualization
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color gradient based on index
sc = ax.scatter(sphere_data[:, 0], sphere_data[:, 1], sphere_data[:, 2],
                c=colors, cmap='hot', alpha=0.7)

# Add color bar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Epochs (index)")

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(f'Fitting score: {fitting_score:.4f} RSS: {rss:.4f}')

# Set the axis limits to [-1, 1] for all dimensions
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Generate sphere mesh (for visual reference)
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

ax.plot_wireframe(x, y, z, color='skyblue', alpha=0.4, linewidth=0.5)

# Show the plot
fig.savefig(output_heatmap_fig, dpi=300)
plt.show()
print("Heatmap-like figure saved to:", output_heatmap_fig)
