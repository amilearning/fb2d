"""Mean ± std heatmap for the long-schedule cumulative run."""
import numpy as np, glob, matplotlib.pyplot as plt, os
QS = ["Q1","Q2","Q3"]
mds, s10s = [], []
for d in sorted(glob.glob("/home/frl/FB2D/checkpoints_taskincr_q123_z3_long/seed*")):
    mds.append(np.load(d+"/mean_d.npy")); s10s.append(np.load(d+"/s10.npy"))
md = np.stack(mds); s = np.stack(s10s)
n = md.shape[0]
md_mu, md_sd = md.mean(0), md.std(0)
s_mu, s_sd = s.mean(0), s.std(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ylabels = ["S1 (Q1, 60k)", "S2 (Q1+Q2, 120k)", "S3 (Q1+Q2+Q3, 180k)"]
for ax, mu, sd, vmax, title, fmt, cmap in [
    (axes[0], md_mu, md_sd, 1.0, "mean final distance", "{:.3f}", "RdYlGn_r"),
    (axes[1], s_mu, s_sd, 1.0, "success rate @ r=0.10", "{:.2f}", "RdYlGn"),
]:
    im = ax.imshow(mu, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(QS)
    ax.set_yticks(range(3)); ax.set_yticklabels(ylabels)
    ax.set_xlabel("eval quadrant")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{fmt.format(mu[i,j])}\n±{fmt.format(sd[i,j])}",
                    ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title, fontweight="bold")
plt.suptitle(f"Cumulative LONG (60k / 120k / 180k)  —  z_dim=3, mean ± std over {n} seeds",
             fontsize=12, fontweight="bold")
plt.tight_layout()
os.makedirs("/home/frl/FB2D/q123_z3_summary", exist_ok=True)
p = "/home/frl/FB2D/q123_z3_summary/cumulative_long_mean.png"
plt.savefig(p, dpi=140, bbox_inches="tight")
print(f"saved {p}")
