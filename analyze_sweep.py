#!/usr/bin/env python3
"""Aggregate and analyze results from the FB continual-learning distillation sweep."""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = "/home/frl/FB2D/checkpoints_distill_real"
OUT  = "/home/frl/FB2D/sweep_analysis"
os.makedirs(OUT, exist_ok=True)

B_TARGETS = ["F", "B", "FB", "M", "Q", "pi"]
D_LOSSES  = ["l2", "cosine", "contrastive", "gram"]
C_INPUTS  = ["current", "replay"]

# ---------------------------------------------------------------------------
# 1. Load all 48 x 3 results
# ---------------------------------------------------------------------------
records = []  # list of dicts

for config_dir in sorted(os.listdir(ROOT)):
    config_path = os.path.join(ROOT, config_dir)
    if not os.path.isdir(config_path):
        continue

    # Parse B_D_C  (handle FB which contains underscore in the target)
    parts = config_dir.split("_")
    # B target may be multi-part (FB), loss and input are always single tokens
    # C (input) is always last, D (loss) is always second-to-last
    C = parts[-1]
    D = parts[-2]
    B = "_".join(parts[:-2])

    seed_dirs = sorted(glob.glob(os.path.join(config_path, "seed*")))
    if len(seed_dirs) == 0:
        print(f"WARNING: no seed dirs in {config_dir}")
        continue

    mean_d_all = []
    s10_all = []

    for sd in seed_dirs:
        md_path = os.path.join(sd, "mean_d.npy")
        s10_path = os.path.join(sd, "s10.npy")
        if not os.path.exists(md_path) or not os.path.exists(s10_path):
            print(f"WARNING: missing npy in {sd}")
            continue
        md = np.load(md_path).flatten()[:3]   # (3,)
        s10 = np.load(s10_path).flatten()[:3]  # (3,)
        mean_d_all.append(md)
        s10_all.append(s10)

    if len(mean_d_all) == 0:
        continue

    mean_d_arr = np.stack(mean_d_all)  # (n_seeds, 3)
    s10_arr = np.stack(s10_all)        # (n_seeds, 3)

    rec = {
        "config": config_dir,
        "B": B, "D": D, "C": C,
        "n_seeds": len(mean_d_all),
        # Per-quadrant stats
        "mean_d_Q1_mean": mean_d_arr[:, 0].mean(),
        "mean_d_Q1_std":  mean_d_arr[:, 0].std(),
        "mean_d_Q2_mean": mean_d_arr[:, 1].mean(),
        "mean_d_Q2_std":  mean_d_arr[:, 1].std(),
        "mean_d_Q3_mean": mean_d_arr[:, 2].mean(),
        "mean_d_Q3_std":  mean_d_arr[:, 2].std(),
        "s10_Q1_mean": s10_arr[:, 0].mean(),
        "s10_Q1_std":  s10_arr[:, 0].std(),
        "s10_Q2_mean": s10_arr[:, 1].mean(),
        "s10_Q2_std":  s10_arr[:, 1].std(),
        "s10_Q3_mean": s10_arr[:, 2].mean(),
        "s10_Q3_std":  s10_arr[:, 2].std(),
        # Averages across quadrants
        "mean_d_avg_mean": mean_d_arr.mean(axis=1).mean(),
        "mean_d_avg_std":  mean_d_arr.mean(axis=1).std(),
        "s10_avg_mean": s10_arr.mean(axis=1).mean(),
        "s10_avg_std":  s10_arr.mean(axis=1).std(),
        # Memory & Plasticity
        "memory_s10_mean": s10_arr[:, 0].mean(),  # s10_Q1
        "memory_s10_std":  s10_arr[:, 0].std(),
        "plasticity_s10_mean": s10_arr[:, 2].mean(),  # s10_Q3
        "plasticity_s10_std":  s10_arr[:, 2].std(),
        "memory_md_mean": mean_d_arr[:, 0].mean(),  # mean_d_Q1 (lower=better)
        "memory_md_std":  mean_d_arr[:, 0].std(),
        "plasticity_md_mean": mean_d_arr[:, 2].mean(),
        "plasticity_md_std":  mean_d_arr[:, 2].std(),
    }
    records.append(rec)

print(f"Loaded {len(records)} configs\n")

# ---------------------------------------------------------------------------
# 3. Print sorted table (by mean_d_avg, lower = better)
# ---------------------------------------------------------------------------
records.sort(key=lambda r: r["mean_d_avg_mean"])

header = f"{'Rank':>4} {'B':>3} {'D':>13} {'C':>8}  {'mean_d_avg':>16}  {'s10_avg':>16}  {'memory(s10_Q1)':>18}  {'plast(s10_Q3)':>18}"
print(header)
print("-" * len(header))
for i, r in enumerate(records):
    print(
        f"{i+1:4d} {r['B']:>3} {r['D']:>13} {r['C']:>8}  "
        f"{r['mean_d_avg_mean']:.4f}±{r['mean_d_avg_std']:.4f}  "
        f"{r['s10_avg_mean']:.4f}±{r['s10_avg_std']:.4f}  "
        f"{r['memory_s10_mean']:.4f}±{r['memory_s10_std']:.4f}  "
        f"{r['plasticity_s10_mean']:.4f}±{r['plasticity_s10_std']:.4f}"
    )

# ---------------------------------------------------------------------------
# 4a. Bar chart: all 48 configs sorted by mean_d_avg
# ---------------------------------------------------------------------------
B_COLORS = {
    "F": "#e41a1c", "B": "#377eb8", "FB": "#4daf4a",
    "M": "#984ea3", "Q": "#ff7f00", "pi": "#a65628"
}

fig, ax = plt.subplots(figsize=(10, 14))
configs_sorted = records[::-1]  # reverse so best is at top visually
y_pos = np.arange(len(configs_sorted))
bars = ax.barh(
    y_pos,
    [r["mean_d_avg_mean"] for r in configs_sorted],
    xerr=[r["mean_d_avg_std"] for r in configs_sorted],
    color=[B_COLORS.get(r["B"], "gray") for r in configs_sorted],
    edgecolor="black", linewidth=0.3, capsize=2
)
ax.set_yticks(y_pos)
ax.set_yticklabels([r["config"] for r in configs_sorted], fontsize=7)
ax.set_xlabel("mean_d_avg (lower = better)")
ax.set_title("All 48 configs sorted by mean_d_avg")
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=B_COLORS[b], label=b) for b in B_TARGETS]
ax.legend(handles=legend_elements, title="B target", loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "bar_mean_d_sorted.png"), dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT}/bar_mean_d_sorted.png")

# ---------------------------------------------------------------------------
# 4b. Memory vs Plasticity scatter
# ---------------------------------------------------------------------------
C_MARKERS = {"current": "o", "replay": "s"}

fig, ax = plt.subplots(figsize=(9, 7))
for r in records:
    ax.errorbar(
        r["plasticity_s10_mean"], r["memory_s10_mean"],
        xerr=r["plasticity_s10_std"], yerr=r["memory_s10_std"],
        fmt=C_MARKERS.get(r["C"], "^"),
        color=B_COLORS.get(r["B"], "gray"),
        markersize=7, capsize=2, alpha=0.8,
        markeredgecolor="black", markeredgewidth=0.4
    )
# Legends
from matplotlib.lines import Line2D
b_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=B_COLORS[b],
                     markersize=8, label=b) for b in B_TARGETS]
c_handles = [Line2D([0], [0], marker=C_MARKERS[c], color="gray",
                     markersize=8, label=c, linestyle="None") for c in C_INPUTS]
leg1 = ax.legend(handles=b_handles, title="B target", loc="upper left")
ax.add_artist(leg1)
ax.legend(handles=c_handles, title="Input (C)", loc="lower right")
ax.set_xlabel("Plasticity (s10_Q3, higher = better)")
ax.set_ylabel("Memory (s10_Q1, higher = better)")
ax.set_title("Memory vs Plasticity")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "memory_vs_plasticity.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT}/memory_vs_plasticity.png")

# ---------------------------------------------------------------------------
# 4c. Heatmap grids
# ---------------------------------------------------------------------------
# Build lookup
lookup = {}
for r in records:
    lookup[(r["B"], r["D"], r["C"])] = r

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, c_val in enumerate(C_INPUTS):
    ax = axes[idx]
    grid = np.full((len(B_TARGETS), len(D_LOSSES)), np.nan)
    for bi, b in enumerate(B_TARGETS):
        for di, d in enumerate(D_LOSSES):
            key = (b, d, c_val)
            if key in lookup:
                grid[bi, di] = lookup[key]["mean_d_avg_mean"]
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(D_LOSSES)))
    ax.set_xticklabels(D_LOSSES)
    ax.set_yticks(range(len(B_TARGETS)))
    ax.set_yticklabels(B_TARGETS)
    ax.set_title(f"mean_d_avg  |  C = {c_val}")
    # Annotate cells
    for bi in range(len(B_TARGETS)):
        for di in range(len(D_LOSSES)):
            val = grid[bi, di]
            if not np.isnan(val):
                ax.text(di, bi, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if val > np.nanmedian(grid) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Heatmap: mean_d_avg by (B target, D loss) for each C input", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "heatmap_mean_d.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT}/heatmap_mean_d.png")

# ---------------------------------------------------------------------------
# 4d. Grouped bar chart by B target
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
n_groups = len(B_TARGETS)
n_bars_per_group = len(D_LOSSES) * len(C_INPUTS)  # 8
bar_width = 0.1
group_width = n_bars_per_group * bar_width

D_HATCHES = {"l2": "", "cosine": "//", "contrastive": "xx", "gram": ".."}
C_ALPHA = {"current": 1.0, "replay": 0.5}

x_centers = np.arange(n_groups) * (group_width + 0.3)

bar_idx = 0
legend_handles = []
for di, d in enumerate(D_LOSSES):
    for ci, c in enumerate(C_INPUTS):
        vals = []
        errs = []
        for b in B_TARGETS:
            key = (b, d, c)
            if key in lookup:
                vals.append(lookup[key]["s10_avg_mean"])
                errs.append(lookup[key]["s10_avg_std"])
            else:
                vals.append(0)
                errs.append(0)
        x_pos = x_centers + bar_idx * bar_width
        color = plt.cm.Set2(di / len(D_LOSSES))
        alpha = C_ALPHA[c]
        bars = ax.bar(x_pos, vals, bar_width, yerr=errs, capsize=2,
                       color=color, alpha=alpha, hatch=D_HATCHES[d],
                       edgecolor="black", linewidth=0.5,
                       label=f"{d}_{c}")
        if bar_idx < n_bars_per_group:
            legend_handles.append(bars)
        bar_idx += 1

ax.set_xticks(x_centers + group_width / 2 - bar_width / 2)
ax.set_xticklabels(B_TARGETS)
ax.set_xlabel("B target")
ax.set_ylabel("s10_avg (higher = better)")
ax.set_title("s10_avg by B target (grouped by D loss x C input)")
ax.legend(loc="upper right", fontsize=7, ncol=2)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "grouped_bar_s10.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT}/grouped_bar_s10.png")

# ---------------------------------------------------------------------------
# 5. Summary analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Best overall config
best = records[0]
print(f"\nBest config (lowest mean_d_avg): {best['config']}")
print(f"  B={best['B']}, D={best['D']}, C={best['C']}")
print(f"  mean_d_avg = {best['mean_d_avg_mean']:.4f} +/- {best['mean_d_avg_std']:.4f}")
print(f"  s10_avg    = {best['s10_avg_mean']:.4f} +/- {best['s10_avg_std']:.4f}")

# Best B target (average mean_d_avg across all D,C)
b_scores = defaultdict(list)
for r in records:
    b_scores[r["B"]].append(r["mean_d_avg_mean"])
b_avg = {b: np.mean(v) for b, v in b_scores.items()}
best_b = min(b_avg, key=b_avg.get)
print(f"\nBest B target (avg mean_d_avg): {best_b} ({b_avg[best_b]:.4f})")
for b in B_TARGETS:
    if b in b_avg:
        print(f"  {b:>3}: {b_avg[b]:.4f}")

# Best D loss
d_scores = defaultdict(list)
for r in records:
    d_scores[r["D"]].append(r["mean_d_avg_mean"])
d_avg = {d: np.mean(v) for d, v in d_scores.items()}
best_d = min(d_avg, key=d_avg.get)
print(f"\nBest D loss (avg mean_d_avg): {best_d} ({d_avg[best_d]:.4f})")
for d in D_LOSSES:
    if d in d_avg:
        print(f"  {d:>13}: {d_avg[d]:.4f}")

# Best C input
c_scores = defaultdict(list)
for r in records:
    c_scores[r["C"]].append(r["mean_d_avg_mean"])
c_avg = {c: np.mean(v) for c, v in c_scores.items()}
best_c = min(c_avg, key=c_avg.get)
print(f"\nBest C input (avg mean_d_avg): {best_c} ({c_avg[best_c]:.4f})")
for c in C_INPUTS:
    if c in c_avg:
        print(f"  {c:>8}: {c_avg[c]:.4f}")

# Best on memory vs plasticity (highest s10_avg)
best_s10 = max(records, key=lambda r: r["s10_avg_mean"])
print(f"\nHighest s10_avg: {best_s10['config']}")
print(f"  s10_avg = {best_s10['s10_avg_mean']:.4f} +/- {best_s10['s10_avg_std']:.4f}")

best_mem = max(records, key=lambda r: r["memory_s10_mean"])
print(f"\nBest memory (s10_Q1): {best_mem['config']}")
print(f"  s10_Q1 = {best_mem['memory_s10_mean']:.4f} +/- {best_mem['memory_s10_std']:.4f}")

best_plast = max(records, key=lambda r: r["plasticity_s10_mean"])
print(f"\nBest plasticity (s10_Q3): {best_plast['config']}")
print(f"  s10_Q3 = {best_plast['plasticity_s10_mean']:.4f} +/- {best_plast['plasticity_s10_std']:.4f}")

print(f"\nAll plots saved to {OUT}/")
