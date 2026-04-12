#!/usr/bin/env python
"""Aggregate and analyze the combined pi+X FB distillation sweep."""

import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── paths ────────────────────────────────────────────────────────────────
COMBINED_DIR = "/home/frl/FB2D/checkpoints_distill_combined_real"
STANDALONE_DIR = "/home/frl/FB2D/checkpoints_distill_real"
OUT_DIR = "/home/frl/FB2D/combined_sweep_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

QUADRANT_NAMES = ["Q1", "Q2", "Q3"]

# ─── helpers ──────────────────────────────────────────────────────────────
def load_seeds(config_path):
    """Return dict with arrays (n_seeds, 3) for mean_d and s10."""
    seed_dirs = sorted(glob.glob(os.path.join(config_path, "seed*")))
    mean_ds, s10s = [], []
    for sd in seed_dirs:
        md_path = os.path.join(sd, "mean_d.npy")
        s10_path = os.path.join(sd, "s10.npy")
        if os.path.exists(md_path) and os.path.exists(s10_path):
            mean_ds.append(np.load(md_path).flatten()[:3])
            s10s.append(np.load(s10_path).flatten()[:3])
    if len(mean_ds) == 0:
        return None
    return {
        "mean_d": np.stack(mean_ds),   # (n_seeds, 3)
        "s10": np.stack(s10s),
    }

def compute_stats(data):
    """From seed arrays, compute summary stats."""
    md = data["mean_d"]  # (n_seeds, 3)
    s10 = data["s10"]
    n = md.shape[0]

    md_avg_per_seed = md.mean(axis=1)       # (n_seeds,)
    s10_avg_per_seed = s10.mean(axis=1)
    memory_per_seed = s10[:, 0]             # Q1
    plasticity_per_seed = s10[:, 2]         # Q3

    return {
        "mean_d_avg": md_avg_per_seed.mean(),
        "mean_d_avg_std": md_avg_per_seed.std(),
        "s10_avg": s10_avg_per_seed.mean(),
        "s10_avg_std": s10_avg_per_seed.std(),
        "memory": memory_per_seed.mean(),
        "memory_std": memory_per_seed.std(),
        "plasticity": plasticity_per_seed.mean(),
        "plasticity_std": plasticity_per_seed.std(),
        # per-quadrant
        "mean_d_Q1": md[:, 0].mean(), "mean_d_Q1_std": md[:, 0].std(),
        "mean_d_Q2": md[:, 1].mean(), "mean_d_Q2_std": md[:, 1].std(),
        "mean_d_Q3": md[:, 2].mean(), "mean_d_Q3_std": md[:, 2].std(),
        "s10_Q1": s10[:, 0].mean(), "s10_Q1_std": s10[:, 0].std(),
        "s10_Q2": s10[:, 1].mean(), "s10_Q2_std": s10[:, 1].std(),
        "s10_Q3": s10[:, 2].mean(), "s10_Q3_std": s10[:, 2].std(),
        "n_seeds": n,
    }

# ─── load combined sweep ──────────────────────────────────────────────────
results = {}
for cfg_name in sorted(os.listdir(COMBINED_DIR)):
    cfg_path = os.path.join(COMBINED_DIR, cfg_name)
    if not os.path.isdir(cfg_path):
        continue
    data = load_seeds(cfg_path)
    if data is None:
        print(f"WARNING: no data for {cfg_name}")
        continue
    stats = compute_stats(data)
    # parse config name: pi{Gram|L2}_{X}_{Cx}
    m = re.match(r"pi(Gram|L2)_(\w+)_(current|replay)", cfg_name)
    if m:
        stats["pi_loss"] = m.group(1)
        stats["X_target"] = m.group(2)
        stats["Cx"] = m.group(3)
    else:
        stats["pi_loss"] = "?"
        stats["X_target"] = "?"
        stats["Cx"] = "?"
    stats["is_baseline"] = False
    results[cfg_name] = stats

# ─── load standalone pi baselines ─────────────────────────────────────────
for bl_name in ["pi_gram_replay", "pi_l2_replay"]:
    bl_path = os.path.join(STANDALONE_DIR, bl_name)
    if not os.path.isdir(bl_path):
        print(f"WARNING: baseline {bl_name} not found")
        continue
    data = load_seeds(bl_path)
    if data is None:
        print(f"WARNING: no data for baseline {bl_name}")
        continue
    stats = compute_stats(data)
    if "gram" in bl_name:
        stats["pi_loss"] = "Gram"
    else:
        stats["pi_loss"] = "L2"
    stats["X_target"] = "none"
    stats["Cx"] = "replay"
    stats["is_baseline"] = True
    results[bl_name] = stats

# ─── sort by mean_d_avg ──────────────────────────────────────────────────
sorted_cfgs = sorted(results.keys(), key=lambda k: results[k]["mean_d_avg"])

# ─── print table ──────────────────────────────────────────────────────────
print("=" * 130)
print(f"{'Config':<28s} {'pi':>4s} {'X':>4s} {'Cx':>7s} {'mean_d_avg':>12s} {'s10_avg':>12s} "
      f"{'memory(Q1)':>12s} {'plast(Q3)':>12s} {'md_Q1':>10s} {'md_Q2':>10s} {'md_Q3':>10s} {'n':>3s}")
print("-" * 130)
for cfg in sorted_cfgs:
    r = results[cfg]
    tag = " ***" if r["is_baseline"] else ""
    print(f"{cfg+tag:<28s} {r['pi_loss']:>4s} {r['X_target']:>4s} {r['Cx']:>7s} "
          f"{r['mean_d_avg']:>6.3f}±{r['mean_d_avg_std']:.3f} "
          f"{r['s10_avg']:>6.3f}±{r['s10_avg_std']:.3f} "
          f"{r['memory']:>6.3f}±{r['memory_std']:.3f} "
          f"{r['plasticity']:>6.3f}±{r['plasticity_std']:.3f} "
          f"{r['mean_d_Q1']:>5.3f}±{r['mean_d_Q1_std']:.2f} "
          f"{r['mean_d_Q2']:>5.3f}±{r['mean_d_Q2_std']:.2f} "
          f"{r['mean_d_Q3']:>5.3f}±{r['mean_d_Q3_std']:.2f} "
          f"{r['n_seeds']:>3d}")

# ─── colors for X targets ────────────────────────────────────────────────
X_COLORS = {
    "F": "#1f77b4", "B": "#ff7f0e", "FB": "#2ca02c", "M": "#d62728",
    "Q": "#9467bd", "FBM": "#8c564b", "none": "#7f7f7f",
}

PI_MARKERS = {"Gram": "o", "L2": "s"}

# ─── plot a: bar chart sorted by mean_d_avg ───────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6))
x_pos = np.arange(len(sorted_cfgs))
bars_vals = [results[c]["mean_d_avg"] for c in sorted_cfgs]
bars_errs = [results[c]["mean_d_avg_std"] for c in sorted_cfgs]
bars_colors = [X_COLORS.get(results[c]["X_target"], "#333333") for c in sorted_cfgs]
edge_colors = ["gold" if results[c]["is_baseline"] else "none" for c in sorted_cfgs]
lw = [2.5 if results[c]["is_baseline"] else 0 for c in sorted_cfgs]

ax.bar(x_pos, bars_vals, yerr=bars_errs, color=bars_colors,
       edgecolor=edge_colors, linewidth=lw, capsize=3)
ax.set_xticks(x_pos)
ax.set_xticklabels(sorted_cfgs, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("mean_d_avg (lower = better)")
ax.set_title("All Configs Sorted by mean_d_avg")
# legend
handles = [mpatches.Patch(color=v, label=k) for k, v in X_COLORS.items()]
handles.append(mpatches.Patch(edgecolor="gold", facecolor="white", linewidth=2, label="standalone baseline"))
ax.legend(handles=handles, fontsize=7, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bar_mean_d_avg.png"), dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/bar_mean_d_avg.png")

# ─── plot b: memory vs plasticity scatter ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for cfg in sorted_cfgs:
    r = results[cfg]
    marker = "*" if r["is_baseline"] else PI_MARKERS.get(r["pi_loss"], "^")
    sz = 200 if r["is_baseline"] else 80
    ax.scatter(r["plasticity"], r["memory"],
               c=X_COLORS.get(r["X_target"], "#333"),
               marker=marker, s=sz, edgecolors="k", linewidths=0.5, zorder=3)
    # label baselines
    if r["is_baseline"]:
        ax.annotate(cfg, (r["plasticity"], r["memory"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=6)

ax.set_xlabel("Plasticity (s10_Q3)")
ax.set_ylabel("Memory (s10_Q1)")
ax.set_title("Memory vs Plasticity")
# legend for X targets
handles_x = [mpatches.Patch(color=v, label=k) for k, v in X_COLORS.items()]
leg1 = ax.legend(handles=handles_x, fontsize=7, loc="upper left", title="X target")
# legend for pi loss markers
from matplotlib.lines import Line2D
handles_pi = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=7, label="Gram"),
    Line2D([0], [0], marker="s", color="gray", linestyle="None", markersize=7, label="L2"),
    Line2D([0], [0], marker="*", color="gray", linestyle="None", markersize=10, label="baseline"),
]
ax.add_artist(leg1)
ax.legend(handles=handles_pi, fontsize=7, loc="lower right", title="pi loss")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "memory_vs_plasticity.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/memory_vs_plasticity.png")

# ─── plot c: heatmap grids ───────────────────────────────────────────────
X_TARGETS = ["F", "B", "FB", "M", "Q", "FBM"]
PI_LOSSES = ["Gram", "L2"]
CX_VALS = ["current", "replay"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ci, cx in enumerate(CX_VALS):
    grid = np.full((len(X_TARGETS), len(PI_LOSSES)), np.nan)
    for xi, xt in enumerate(X_TARGETS):
        for pi_i, pl in enumerate(PI_LOSSES):
            cfg_name = f"pi{pl}_{xt}_{cx}"
            if cfg_name in results:
                grid[xi, pi_i] = results[cfg_name]["mean_d_avg"]
    im = axes[ci].imshow(grid, cmap="RdYlGn_r", aspect="auto")
    axes[ci].set_xticks(range(len(PI_LOSSES)))
    axes[ci].set_xticklabels(PI_LOSSES)
    axes[ci].set_yticks(range(len(X_TARGETS)))
    axes[ci].set_yticklabels(X_TARGETS)
    axes[ci].set_title(f"Cx = {cx}")
    # annotate cells
    for xi in range(len(X_TARGETS)):
        for pi_i in range(len(PI_LOSSES)):
            if not np.isnan(grid[xi, pi_i]):
                axes[ci].text(pi_i, xi, f"{grid[xi, pi_i]:.3f}",
                              ha="center", va="center", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=axes[ci], shrink=0.8)
fig.suptitle("mean_d_avg Heatmap (lower = better)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "heatmap_mean_d_avg.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/heatmap_mean_d_avg.png")

# ─── plot d: improvement over baseline ────────────────────────────────────
gram_baseline = results.get("pi_gram_replay", {}).get("mean_d_avg", None)
l2_baseline = results.get("pi_l2_replay", {}).get("mean_d_avg", None)

fig, ax = plt.subplots(figsize=(18, 6))
combined_cfgs = [c for c in sorted_cfgs if not results[c]["is_baseline"]]
deltas = []
colors = []
for cfg in combined_cfgs:
    r = results[cfg]
    bl = gram_baseline if r["pi_loss"] == "Gram" else l2_baseline
    if bl is not None:
        delta = bl - r["mean_d_avg"]  # positive = combined is better
    else:
        delta = 0.0
    deltas.append(delta)
    colors.append(X_COLORS.get(r["X_target"], "#333"))

x_pos = np.arange(len(combined_cfgs))
bar_colors_pos = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
ax.bar(x_pos, deltas, color=colors, edgecolor=bar_colors_pos, linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(combined_cfgs, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Delta mean_d (positive = better than standalone pi)")
ax.set_title("Improvement Over Standalone pi Baseline (matched pi loss)")
handles = [mpatches.Patch(color=v, label=k) for k, v in X_COLORS.items() if k != "none"]
ax.legend(handles=handles, fontsize=7, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "improvement_over_baseline.png"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/improvement_over_baseline.png")

# ─── summary analysis ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY ANALYSIS")
print("=" * 80)

# 1. Does adding X help?
combined_only = {k: v for k, v in results.items() if not v["is_baseline"]}
best_combined = min(combined_only.values(), key=lambda x: x["mean_d_avg"])
best_combined_name = [k for k, v in combined_only.items() if v is best_combined][0]

print(f"\n1. Does adding X to pi help vs standalone pi?")
print(f"   Standalone pi_gram_replay:  mean_d_avg = {gram_baseline:.4f}")
print(f"   Standalone pi_l2_replay:    mean_d_avg = {l2_baseline:.4f}")
print(f"   Best combined config:       {best_combined_name} = {best_combined['mean_d_avg']:.4f}")

n_better_gram = sum(1 for v in combined_only.values()
                    if v["pi_loss"] == "Gram" and v["mean_d_avg"] < gram_baseline)
n_gram = sum(1 for v in combined_only.values() if v["pi_loss"] == "Gram")
n_better_l2 = sum(1 for v in combined_only.values()
                  if v["pi_loss"] == "L2" and v["mean_d_avg"] < l2_baseline)
n_l2 = sum(1 for v in combined_only.values() if v["pi_loss"] == "L2")
print(f"   Gram configs beating baseline: {n_better_gram}/{n_gram}")
print(f"   L2 configs beating baseline:   {n_better_l2}/{n_l2}")

# 2. Which X target helps most?
print(f"\n2. Which X target helps the most?")
for xt in X_TARGETS:
    subset = {k: v for k, v in combined_only.items() if v["X_target"] == xt}
    if subset:
        avg_md = np.mean([v["mean_d_avg"] for v in subset.values()])
        best_in = min(subset.items(), key=lambda x: x[1]["mean_d_avg"])
        print(f"   X={xt:>3s}: avg mean_d_avg = {avg_md:.4f}, best = {best_in[0]} ({best_in[1]['mean_d_avg']:.4f})")

# 3. Memory vs plasticity
print(f"\n3. Memory/plasticity comparison:")
gram_bl_mem = results.get("pi_gram_replay", {}).get("memory", 0)
gram_bl_pla = results.get("pi_gram_replay", {}).get("plasticity", 0)
l2_bl_mem = results.get("pi_l2_replay", {}).get("memory", 0)
l2_bl_pla = results.get("pi_l2_replay", {}).get("plasticity", 0)
print(f"   Baseline pi_gram_replay: memory={gram_bl_mem:.4f}, plasticity={gram_bl_pla:.4f}")
print(f"   Baseline pi_l2_replay:   memory={l2_bl_mem:.4f}, plasticity={l2_bl_pla:.4f}")

# configs that improve memory while maintaining plasticity
print(f"\n   Configs improving memory while maintaining plasticity (vs matched baseline):")
for cfg in sorted_cfgs:
    r = results[cfg]
    if r["is_baseline"]:
        continue
    bl_mem = gram_bl_mem if r["pi_loss"] == "Gram" else l2_bl_mem
    bl_pla = gram_bl_pla if r["pi_loss"] == "Gram" else l2_bl_pla
    if r["memory"] > bl_mem and r["plasticity"] >= bl_pla * 0.9:
        delta_mem = r["memory"] - bl_mem
        delta_pla = r["plasticity"] - bl_pla
        print(f"     {cfg:<28s} mem={r['memory']:.4f} (+{delta_mem:.4f}), "
              f"pla={r['plasticity']:.4f} ({'+' if delta_pla>=0 else ''}{delta_pla:.4f})")

# 4. Best overall
print(f"\n4. Best overall combined config:")
print(f"   Config:      {best_combined_name}")
print(f"   pi_loss:     {best_combined['pi_loss']}")
print(f"   X_target:    {best_combined['X_target']}")
print(f"   Cx:          {best_combined['Cx']}")
print(f"   mean_d_avg:  {best_combined['mean_d_avg']:.4f} ± {best_combined['mean_d_avg_std']:.4f}")
print(f"   s10_avg:     {best_combined['s10_avg']:.4f} ± {best_combined['s10_avg_std']:.4f}")
print(f"   memory:      {best_combined['memory']:.4f} ± {best_combined['memory_std']:.4f}")
print(f"   plasticity:  {best_combined['plasticity']:.4f} ± {best_combined['plasticity_std']:.4f}")
print(f"   Q1/Q2/Q3 mean_d: {best_combined['mean_d_Q1']:.3f} / {best_combined['mean_d_Q2']:.3f} / {best_combined['mean_d_Q3']:.3f}")
print(f"   Q1/Q2/Q3 s10:    {best_combined['s10_Q1']:.3f} / {best_combined['s10_Q2']:.3f} / {best_combined['s10_Q3']:.3f}")

# Top-5
print(f"\n   Top-5 by mean_d_avg:")
for i, cfg in enumerate(sorted_cfgs[:5]):
    r = results[cfg]
    bl_tag = " [BASELINE]" if r["is_baseline"] else ""
    print(f"     {i+1}. {cfg}{bl_tag}: mean_d={r['mean_d_avg']:.4f}, s10={r['s10_avg']:.4f}, "
          f"mem={r['memory']:.4f}, pla={r['plasticity']:.4f}")

print(f"\nDone. Plots saved to {OUT_DIR}/")
