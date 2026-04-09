"""Mean ± std heatmaps over 3 seeds for cumulative vs naive (z_dim=3, q123)."""
import numpy as np, glob, matplotlib.pyplot as plt, os

QS = ["Q1","Q2","Q3"]
OUT = "/home/frl/FB2D/q123_z3_summary"
os.makedirs(OUT, exist_ok=True)

def collect(pat):
    mds, s10s = [], []
    for d in sorted(glob.glob(pat)):
        try:
            mds.append(np.load(d+"/mean_d.npy"))
            s10s.append(np.load(d+"/s10.npy"))
        except: pass
    return np.stack(mds), np.stack(s10s)

runs = [
    ("Cumulative (Q1, Q1+Q2, Q1+Q2+Q3)",
     "/home/frl/FB2D/checkpoints_taskincr_q123_z3/seed*",
     [f"S{i+1} ({s})" for i,s in enumerate(["Q1","Q1+Q2","Q1+Q2+Q3"])]),
    ("Naive sequential (Q1 → Q2 → Q3)",
     "/home/frl/FB2D/checkpoints_naive_seq_q123_z3/seed*",
     [f"S{i+1} (only {q})" for i,q in enumerate(QS)]),
]

for label, pat, ylabels in runs:
    md, s = collect(pat)
    n = md.shape[0]
    md_mu, md_sd = md.mean(0), md.std(0)
    s_mu, s_sd = s.mean(0), s.std(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat_mu, mat_sd, vmax, title, fmt, cmap in [
        (axes[0], md_mu, md_sd, 1.0, "mean final distance", "{:.3f}", "RdYlGn_r"),
        (axes[1], s_mu, s_sd, 1.0, "success rate @ r=0.10", "{:.2f}", "RdYlGn"),
    ]:
        im = ax.imshow(mat_mu, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(QS)
        ax.set_yticks(range(3)); ax.set_yticklabels(ylabels)
        ax.set_xlabel("eval quadrant")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{fmt.format(mat_mu[i,j])}\n±{fmt.format(mat_sd[i,j])}",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="black")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, fontweight="bold")
    plt.suptitle(f"{label}  —  z_dim=3, mean ± std over {n} seeds",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = label.split(" ")[0].lower() + "_mean.png"
    p = os.path.join(OUT, fname)
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"saved {p}")
