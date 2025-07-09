#!/usr/bin/env python3
"""
GPCA trajectory visualisation  (원본 기능 그대로 + -c 플래그)
-----------------------------------------------------------
* 단일 CSV  ― 기존 코드와 결과·파일명 100 % 동일.
* -c (--combine-trajectories) 를 주면
      eigenvectors/<layer>_leading.csv
      eigenvectors/<layer>_subleading.csv
  두 궤적을 **공통 (b, t₁, t₂)** 에 맞춰 함께 투영·시각화.

Added / changed 부분은  ### COMBINE MODE  주석으로 표시.
"""
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from tqdm import tqdm

import torch, geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import Stiefel
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
import plotly.graph_objects as go  # ✅ plotly for interactive 3D


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '9')
torch.manual_seed(42); np.random.seed(42)


# ────────────────────────── 헬퍼 ──────────────────────────
def normalize_sign(x: torch.Tensor):
    l = x.shape[0]
    for _ in range(3):
        for i in range(l-1):
            pos = torch.dot( x[i+1], x[i])
            neg = torch.dot(-x[i+1], x[i])
            if torch.acos(torch.clamp(neg/(x[i+1].norm()*x[i].norm()+1e-14), -1, 1)) < \
               torch.acos(torch.clamp(pos/(x[i+1].norm()*x[i].norm()+1e-14), -1, 1)):
                x[i+1].neg_()
    return x

def dist_rows(A, B):
    inner = (A*B).sum(1)
    norm  = A.norm(dim=1)*B.norm(dim=1) + 1e-14
    return torch.arccos(torch.clamp(inner/norm, -1., 1.))

def proj_geodesic(X, b, t):
    n   = t.norm()
    ang = torch.atan2((X*t).sum(1), (X*b).sum(1)) / (n+1e-14)
    P   = torch.cos(ang).unsqueeze(1)*b + torch.sin(ang).unsqueeze(1)*t/n
    flip = dist_rows(X, -P) < dist_rows(X, P)
    P[flip] *= -1
    return P

def proj_2sphere(X, b, t1, t2):
    u1 = t1 / (t1.norm()+1e-14)
    u2 = t2 - (t2 @ u1) * u1
    u2 = u2 / (u2.norm()+1e-14)
    B  = torch.stack([b, u1, u2], 1)           # (p,3)
    Xp = X @ B @ B.T
    Xp = Xp / (Xp.norm(dim=1, keepdim=True)+1e-14)
    return Xp
# ─────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument('--dirs',   nargs='+', required=True)
parser.add_argument('--layers', nargs='+', required=True)
parser.add_argument('--start-epoch', type=int, default=1)
parser.add_argument('--end-epoch',   type=int, default=100)
parser.add_argument('--gpca-dim',    type=int, default=2, choices=[1,2])
# ### COMBINE MODE flag
parser.add_argument('-c', '--combine-trajectories', action='store_true',
                    help='Use <layer>_leading.csv + _subleading.csv jointly')
args = parser.parse_args()


for base_dir in args.dirs:
    for layer in args.layers:
        print(f"\n===== {base_dir}/{layer} =====")
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ---------- 입력 파일 ----------
        single_csv = os.path.join(base_dir, f'eigenvectors/{layer}.csv')
        lead_csv   = os.path.join(base_dir, f'eigenvectors/{layer}_leading.csv')
        sub_csv    = os.path.join(base_dir, f'eigenvectors/{layer}_subleading.csv')

        combine = args.combine_trajectories and \
                  os.path.exists(lead_csv) and os.path.exists(sub_csv)

        load = lambda p: pd.read_csv(p, sep=r'\s+', header=None) \
                           .iloc[args.start_epoch-1:args.end_epoch].to_numpy()

        if combine:                                    # ### COMBINE MODE ###
            eig_lead, eig_sub = load(lead_csv), load(sub_csv)
            l  = min(len(eig_lead), len(eig_sub))
            p  = eig_lead.shape[1]
            eig_lead, eig_sub = eig_lead[:l], eig_sub[:l]
            data_lead = normalize_sign(torch.tensor(eig_lead, dtype=torch.float, device=dev))
            data_sub  = normalize_sign(torch.tensor(eig_sub , dtype=torch.float, device=dev))

            data_all  = torch.cat([data_lead, data_sub], 0)
        else:                                          # 원래 단일 CSV 경로
            if not os.path.exists(single_csv):
                print('❌ CSV missing'); continue
            eig       = load(single_csv)
            l, p      = eig.shape
            data_lead = normalize_sign(torch.tensor(eig, device=dev))
            data_all  = data_lead

        space = Hypersphere(dim=p-1)

        # ---------- Stiefel 파라미터 ----------
        k     = 2 if args.gpca_dim == 1 else 3
        Q,_   = torch.linalg.qr(torch.randn(p, k, device=dev))
        M     = ManifoldParameter(Q, manifold=Stiefel())
        opt   = geoopt.optim.RiemannianSGD([M], lr=1e-3, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 5000, 2, 1e-4)

        def loss():
            b = M[:,0]
            if args.gpca_dim == 1:
                Xp = proj_geodesic(data_all, b, M[:,1])
            else:
                Xp = proj_2sphere(data_all, b, M[:,1], M[:,2])
            return 0.5 * dist_rows(data_all, Xp).pow(2).sum()

        prev, cnt = 1e9, 0
        pbar = tqdm(range(200000), desc=f'GPCA-{args.gpca_dim}D')
        for step in pbar:
            opt.zero_grad()
            v = loss()
            v.backward()
            opt.step()
            sched.step()
            pbar.set_postfix(loss=f"{v.item():.6f}")
            opt.zero_grad(); v = loss(); v.backward(); opt.step(); sched.step()
            with torch.no_grad():
                Q, _ = torch.linalg.qr(M); M.copy_(Q)
            if abs(prev - v.item()) < 1e-5:
                cnt += 1
                if cnt >= 100:
                    print(f'✔ converged @ {step}  loss={v.item():.6f}'); break
            else:
                cnt = 0
            prev = v.item()

        # ---------- 투영 ----------
        b = M[:,0]
        if args.gpca_dim == 1:
            t = M[:,1]
            lead_proj = proj_geodesic(data_lead, b, t)
            if combine: sub_proj = proj_geodesic(data_sub, b, t)
        else:
            t1, t2 = M[:,1], M[:,2]
            lead_proj = proj_2sphere(data_lead, b, t1, t2)
            if combine: sub_proj = proj_2sphere(data_sub, b, t1, t2)

        # ---------- RSS ----------
        rss_lead = dist_rows(data_lead, lead_proj).pow(2).sum().item()
        if combine:
            rss_sub = dist_rows(data_sub, sub_proj).pow(2).sum().item()
            print(f"RSS lead={rss_lead:.4f}  sub={rss_sub:.4f}")
        else:
            print(f"RSS={rss_lead:.4f}")

        # ---------- 폴더 & epoch array ----------
        outdir = os.path.join(base_dir, 'gpca_results'); os.makedirs(outdir, exist_ok=True)
        epoch  = np.arange(args.start_epoch, args.start_epoch + l)

        # ---------- 1 D 플롯 ----------
        if args.gpca_dim == 1:
            base_np = b.cpu().numpy(); tan_np = t.cpu().numpy()
            log     = space.metric.log(lead_proj.cpu().numpy(), base_np)
            gp      = (log / tan_np)[:, 0]
            gp      = (gp + np.pi) % (2*np.pi) - np.pi

            plt.figure(figsize=(6,4))
            plt.plot(epoch, gp, '-o', label='leading')
            if combine:
                log_s = space.metric.log(sub_proj.cpu().numpy(), base_np)
                gp_s  = (log_s / tan_np)[:, 0]
                gp_s  = (gp_s + np.pi) % (2*np.pi) - np.pi
                plt.plot(epoch, gp_s, '--x', label='subleading')
            plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Geodesic Param'); plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir,
                        f'{layer}_1D_{args.start_epoch}_{args.end_epoch}.png'), dpi=300)
            plt.close()

        # ---------- 2 D 플롯 ----------
        else:
            B   = np.vstack([
                b.detach().cpu().numpy(),
                t1.detach().cpu().numpy(),
                t2.detach().cpu().numpy()
            ]).T
            Qm, _ = np.linalg.qr(B)
            lead_xyz = lead_proj.detach().cpu().numpy() @ Qm
            if combine: sub_xyz = sub_proj.detach().cpu().numpy() @ Qm

            fig = plt.figure(figsize=(8,6))
            ax  = fig.add_subplot(111, projection='3d')
            sc1 = ax.scatter(*lead_xyz.T, c=epoch, cmap='Reds',   label='leading',   alpha=.7)
            if combine:
                sc2 = ax.scatter(*sub_xyz.T,  c=epoch, cmap='Blues', label='subleading', alpha=.7)
            u = np.linspace(0, 2*np.pi, 30); v = np.linspace(0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(x, y, z, color='gray', alpha=.5, linewidth=.4)
            ax.set_box_aspect([1,1,1]); ax.legend()
            plt.tight_layout()
            # 컬러바 추가 (공통 epoch 기준)
            cbar = plt.colorbar(sc1, ax=ax, shrink=0.7, pad=0.1)
            cbar.set_label('Epoch')

            # 만약 combine인 경우 subleading은 별도 컬러바로도 가능 (선택)
            cbar2 = plt.colorbar(sc2, ax=ax, shrink=0.7, pad=0.05)
            cbar2.set_label('Epoch (subleading)')
            plt.savefig(os.path.join(outdir,
                        f'{layer}_2D_{args.start_epoch}_{args.end_epoch}.png'), dpi=300)
            
            # ---------- plotly html 저장 ----------
            plotly_fig = go.Figure()

            plotly_fig.add_trace(go.Scatter3d(
                x=lead_xyz[:,0], y=lead_xyz[:,1], z=lead_xyz[:,2],
                mode='markers',
                marker=dict(size=4, color=epoch, colorscale='Reds', opacity=0.7),
                name='leading'
            ))

            if combine:
                plotly_fig.add_trace(go.Scatter3d(
                    x=sub_xyz[:,0], y=sub_xyz[:,1], z=sub_xyz[:,2],
                    mode='markers',
                    marker=dict(size=4, color=epoch, colorscale='Blues', opacity=0.7),
                    name='subleading'
                ))

            plotly_fig.update_layout(
                scene=dict(aspectmode='data'),
                margin=dict(l=0, r=0, b=0, t=0),
                legend=dict(x=0.02, y=0.98),
                coloraxis_colorbar=dict(title='Epoch')
            )

            plotly_outfile = os.path.join(outdir, f'{layer}_2D_{args.start_epoch}_{args.end_epoch}.html')
            plotly_fig.write_html(plotly_outfile)
            print(f"✔ saved interactive HTML: {plotly_outfile}")

            plt.close()

        # ---------- CSV 저장 ----------
        if combine:
            df = pd.DataFrame(torch.cat([lead_proj, sub_proj]).detach().cpu().numpy())
            df['epoch'] = np.concatenate([epoch, epoch])
            df['type']  = ['lead']*l + ['sub']*l
            csv_name = f'{layer}_proj_both_{args.start_epoch}_{args.end_epoch}.csv'
        else:
            df = pd.DataFrame(lead_proj.cpu().numpy())
            df['epoch'] = epoch
            csv_name = f'{layer}_proj_{args.start_epoch}_{args.end_epoch}.csv'
        df.to_csv(os.path.join(outdir, csv_name), index=False)
        print('✔ saved CSV & plots')

print("\n🎉 ALL DONE")
