#!/usr/bin/env python3
"""Aggregate and analyze results from the z-sampling FB continual-learning sweep."""

import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "/home/frl/FB2D/z_sweep_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. Load all results
# ──────────────────────────────────────────────────────────────

def load_config(base_dir, config_name):
    """Load mean_d and s10 across seeds for a config directory."""
    config_path = os.path.join(base_dir, config_name)
    seed_dirs = sorted(glob.glob(os.path.join(config_path, "seed*")))
    if not seed_dirs:
        print(f"  WARNING: no seed dirs in {config_path}")
        return None
    mean_ds, s10s = [], []
    for sd in seed_dirs:
        md_path = os.path.join(sd, "mean_d.npy")
        s10_path = os.path.join(sd, "s10.npy")
        if os.path.exists(md_path) and os.path.exists(s10_path):
            mean_ds.append(np.load(md_path).flatten()[:3])
            s10s.append(np.load(s10_path).flatten()[:3])
    if not mean_ds:
        print(f"  WARNING: no valid data in {config_path}")
        return None
    return {
        "mean_d": np.array(mean_ds),  # (n_seeds, 3)
        "s10": np.array(s10s),
    }


Z_BASE = "/home/frl/FB2D/checkpoints_z_sampling_real"
Z_CONFIGS = [
    "A1_vmf_bind", "A2_vmf_mix",
    "B1_vmf_bind", "B2_vmf_mix",
    "C1_smr_F", "C2_smr_B", "C3_smr_FB", "C4_smr_pi", "C5_smr_piFB",
    "C6_fdws_F", "C7_fdws_B", "C8_fdws_FB", "C9_fdws_pi", "C10_fdws_piFB",
    "D1_smr_F", "D2_smr_B", "D3_smr_FB", "D4_smr_pi", "D5_smr_piFB",
    "D6_fdws_F", "D7_fdws_B", "D8_fdws_FB", "D9_fdws_pi", "D10_fdws_piFB",
]

BASELINES = {
    "BL_pi_gram_replay": ("/home/frl/FB2D/checkpoints_distill_real", "pi_gram_replay"),
    "BL_piGram_FB_replay": ("/home/frl/FB2D/checkpoints_distill_combined_real", "piGram_FB_replay"),
    "BL_piGram_B_replay": ("/home/frl/FB2D/checkpoints_distill_combined_real", "piGram_B_replay"),
}

results = {}

print("Loading z-sampling configs...")
for cfg in Z_CONFIGS:
    data = load_config(Z_BASE, cfg)
    if data is not None:
        results[cfg] = data

print("Loading baselines...")
for name, (base, cfg) in BASELINES.items():
    data = load_config(base, cfg)
    if data is not None:
        results[name] = data

print(f"Loaded {len(results)} configs total.\n")

# ──────────────────────────────────────────────────────────────
# 2. Compute summary statistics
# ──────────────────────────────────────────────────────────────

def get_group(name):
    if name.startswith("BL"):
        return "baseline"
    return name[0]

def get_color(group):
    return {"A": "tab:blue", "B": "tab:green", "C": "tab:orange", "D": "tab:red", "baseline": "black"}[group]

rows = []
for name, data in results.items():
    md = data["mean_d"]  # (seeds, 3)
    s10 = data["s10"]
    md_avg = md.mean(axis=1)  # per-seed avg across quadrants
    s10_avg = s10.mean(axis=1)
    row = {
        "name": name,
        "group": get_group(name),
        "mean_d_avg_mean": md_avg.mean(),
        "mean_d_avg_std": md_avg.std(),
        "s10_avg_mean": s10_avg.mean(),
        "s10_avg_std": s10_avg.std(),
        "mean_d_Q1_mean": md[:, 0].mean(),
        "mean_d_Q1_std": md[:, 0].std(),
        "mean_d_Q2_mean": md[:, 1].mean(),
        "mean_d_Q2_std": md[:, 1].std(),
        "mean_d_Q3_mean": md[:, 2].mean(),
        "mean_d_Q3_std": md[:, 2].std(),
        "s10_Q1_mean": s10[:, 0].mean(),
        "s10_Q1_std": s10[:, 0].std(),
        "s10_Q2_mean": s10[:, 1].mean(),
        "s10_Q2_std": s10[:, 1].std(),
        "s10_Q3_mean": s10[:, 2].mean(),
        "s10_Q3_std": s10[:, 2].std(),
        # convenience
        "memory_mean_d": md[:, 0].mean(),
        "plasticity_mean_d": md[:, 2].mean(),
        "memory_s10": s10[:, 0].mean(),
        "plasticity_s10": s10[:, 2].mean(),
    }
    rows.append(row)

rows.sort(key=lambda r: r["mean_d_avg_mean"])

# ──────────────────────────────────────────────────────────────
# Print sorted table
# ──────────────────────────────────────────────────────────────

print("=" * 130)
print(f"{'Rank':>4} {'Config':<25} {'Group':>6}  {'mean_d_avg':>12} {'s10_avg':>12}  {'mem(md_Q1)':>12} {'plas(md_Q3)':>12}  {'mem(s10_Q1)':>12} {'plas(s10_Q3)':>12}")
print("-" * 130)
for i, r in enumerate(rows, 1):
    print(f"{i:>4} {r['name']:<25} {r['group']:>6}  "
          f"{r['mean_d_avg_mean']:>5.3f}±{r['mean_d_avg_std']:.3f}  "
          f"{r['s10_avg_mean']:>5.3f}±{r['s10_avg_std']:.3f}  "
          f"{r['mean_d_Q1_mean']:>5.3f}±{r['mean_d_Q1_std']:.3f}  "
          f"{r['mean_d_Q3_mean']:>5.3f}±{r['mean_d_Q3_std']:.3f}  "
          f"{r['s10_Q1_mean']:>5.3f}±{r['s10_Q1_std']:.3f}  "
          f"{r['s10_Q3_mean']:>5.3f}±{r['s10_Q3_std']:.3f}")
print("=" * 130)

# Per-quadrant breakdown
print("\n\nPer-quadrant breakdown (mean_d):")
print(f"{'Config':<25} {'Q1':>12} {'Q2':>12} {'Q3':>12}")
print("-" * 65)
for r in rows:
    print(f"{r['name']:<25} {r['mean_d_Q1_mean']:>5.3f}±{r['mean_d_Q1_std']:.3f}  "
          f"{r['mean_d_Q2_mean']:>5.3f}±{r['mean_d_Q2_std']:.3f}  "
          f"{r['mean_d_Q3_mean']:>5.3f}±{r['mean_d_Q3_std']:.3f}")

print("\n\nPer-quadrant breakdown (s10):")
print(f"{'Config':<25} {'Q1':>12} {'Q2':>12} {'Q3':>12}")
print("-" * 65)
for r in rows:
    print(f"{r['name']:<25} {r['s10_Q1_mean']:>5.3f}±{r['s10_Q1_std']:.3f}  "
          f"{r['s10_Q2_mean']:>5.3f}±{r['s10_Q2_std']:.3f}  "
          f"{r['s10_Q3_mean']:>5.3f}±{r['s10_Q3_std']:.3f}")

# ──────────────────────────────────────────────────────────────
# 3. Plots
# ──────────────────────────────────────────────────────────────

# (a) Bar chart sorted by mean_d_avg
fig, ax = plt.subplots(figsize=(18, 7))
names = [r["name"] for r in rows]
vals = [r["mean_d_avg_mean"] for r in rows]
errs = [r["mean_d_avg_std"] for r in rows]
colors = [get_color(r["group"]) for r in rows]
bars = ax.bar(range(len(names)), vals, yerr=errs, color=colors, edgecolor="grey", capsize=3)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("mean_d_avg (lower=better)")
ax.set_title("All configs sorted by mean_d_avg")
legend_patches = [mpatches.Patch(color=get_color(g), label=g) for g in ["A", "B", "C", "D", "baseline"]]
ax.legend(handles=legend_patches, loc="upper left")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "a_bar_mean_d_avg.png"), dpi=150)
plt.close()

# (b) Memory vs Plasticity scatter (mean_d)
fig, ax = plt.subplots(figsize=(10, 8))
for r in rows:
    marker = "*" if r["group"] == "baseline" else "o"
    sz = 200 if r["group"] == "baseline" else 80
    ax.scatter(r["plasticity_mean_d"], r["memory_mean_d"],
               c=get_color(r["group"]), s=sz, marker=marker, edgecolors="black", linewidths=0.5, zorder=5)
    ax.annotate(r["name"], (r["plasticity_mean_d"], r["memory_mean_d"]),
                fontsize=5, alpha=0.7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Plasticity (mean_d_Q3, lower=better)")
ax.set_ylabel("Memory (mean_d_Q1, lower=better)")
ax.set_title("Memory vs Plasticity (mean_d)")
legend_patches = [mpatches.Patch(color=get_color(g), label=g) for g in ["A", "B", "C", "D", "baseline"]]
ax.legend(handles=legend_patches, loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "b_memory_vs_plasticity.png"), dpi=150)
plt.close()

# (c) Grouped comparison by sensitivity source
sources = ["F", "B", "FB", "pi", "piFB"]
fig, axes = plt.subplots(1, len(sources) + 1, figsize=(22, 6), sharey=True)

# Helper to get mean_d_avg for a config
def get_val(name):
    for r in rows:
        if r["name"] == name:
            return r["mean_d_avg_mean"], r["mean_d_avg_std"]
    return None, None

for idx, src in enumerate(sources):
    ax = axes[idx]
    labels = [f"SMR\nno-dist", f"SMR\n+dist", f"FDWS\nno-dist", f"FDWS\n+dist"]
    # Find the config numbers
    smr_no = f"C{sources.index(src)+1}_smr_{src}"
    smr_dist = f"D{sources.index(src)+1}_smr_{src}"
    fdws_no = f"C{sources.index(src)+6}_fdws_{src}"
    fdws_dist = f"D{sources.index(src)+6}_fdws_{src}"
    cfgs = [smr_no, smr_dist, fdws_no, fdws_dist]
    vals_g, errs_g = [], []
    for c in cfgs:
        v, e = get_val(c)
        vals_g.append(v if v is not None else 0)
        errs_g.append(e if e is not None else 0)
    bar_colors = ["tab:orange", "tab:red", "tab:orange", "tab:red"]
    ax.bar(range(4), vals_g, yerr=errs_g, color=bar_colors, edgecolor="grey", capsize=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(f"Source: {src}", fontsize=10)
    if idx == 0:
        ax.set_ylabel("mean_d_avg")

# vMF panel
ax = axes[-1]
vmf_labels = ["vMF_bind\nno-dist", "vMF_mix\nno-dist", "vMF_bind\n+dist", "vMF_mix\n+dist"]
vmf_cfgs = ["A1_vmf_bind", "A2_vmf_mix", "B1_vmf_bind", "B2_vmf_mix"]
vmf_colors = ["tab:blue", "tab:blue", "tab:green", "tab:green"]
vals_g, errs_g = [], []
for c in vmf_cfgs:
    v, e = get_val(c)
    vals_g.append(v if v is not None else 0)
    errs_g.append(e if e is not None else 0)
ax.bar(range(4), vals_g, yerr=errs_g, color=vmf_colors, edgecolor="grey", capsize=3)
ax.set_xticks(range(4))
ax.set_xticklabels(vmf_labels, fontsize=7)
ax.set_title("vMF methods", fontsize=10)

plt.suptitle("Grouped comparison: SMR vs FDWS × no-distill vs +distill", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "c_grouped_comparison.png"), dpi=150)
plt.close()

# (d) Heatmap: rows = method, cols = {no_distill, +distill}
method_labels = [
    "vMF_bind", "vMF_mix",
    "smr_F", "smr_B", "smr_FB", "smr_pi", "smr_piFB",
    "fdws_F", "fdws_B", "fdws_FB", "fdws_pi", "fdws_piFB",
]
no_distill_map = {
    "vMF_bind": "A1_vmf_bind", "vMF_mix": "A2_vmf_mix",
    "smr_F": "C1_smr_F", "smr_B": "C2_smr_B", "smr_FB": "C3_smr_FB",
    "smr_pi": "C4_smr_pi", "smr_piFB": "C5_smr_piFB",
    "fdws_F": "C6_fdws_F", "fdws_B": "C7_fdws_B", "fdws_FB": "C8_fdws_FB",
    "fdws_pi": "C9_fdws_pi", "fdws_piFB": "C10_fdws_piFB",
}
distill_map = {
    "vMF_bind": "B1_vmf_bind", "vMF_mix": "B2_vmf_mix",
    "smr_F": "D1_smr_F", "smr_B": "D2_smr_B", "smr_FB": "D3_smr_FB",
    "smr_pi": "D4_smr_pi", "smr_piFB": "D5_smr_piFB",
    "fdws_F": "D6_fdws_F", "fdws_B": "D7_fdws_B", "fdws_FB": "D8_fdws_FB",
    "fdws_pi": "D9_fdws_pi", "fdws_piFB": "D10_fdws_piFB",
}

heatmap_data = np.full((len(method_labels), 2), np.nan)
for i, ml in enumerate(method_labels):
    v0, _ = get_val(no_distill_map[ml])
    v1, _ = get_val(distill_map[ml])
    if v0 is not None:
        heatmap_data[i, 0] = v0
    if v1 is not None:
        heatmap_data[i, 1] = v1

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks([0, 1])
ax.set_xticklabels(["no_distill", "+distill"])
ax.set_yticks(range(len(method_labels)))
ax.set_yticklabels(method_labels, fontsize=9)
for i in range(len(method_labels)):
    for j in range(2):
        if not np.isnan(heatmap_data[i, j]):
            ax.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if heatmap_data[i, j] > np.nanmedian(heatmap_data) else "black")
plt.colorbar(im, ax=ax, label="mean_d_avg (lower=better)")
ax.set_title("Z-sampling method × Distillation")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "d_heatmap.png"), dpi=150)
plt.close()

print(f"\nPlots saved to {OUT_DIR}/")

# ──────────────────────────────────────────────────────────────
# 4. Summary analysis
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SUMMARY ANALYSIS")
print("=" * 80)

# Best overall
best = rows[0]
print(f"\n1. BEST CONFIG (lowest mean_d_avg): {best['name']}  →  {best['mean_d_avg_mean']:.4f}±{best['mean_d_avg_std']:.4f}")

# Baselines
bl_rows = [r for r in rows if r["group"] == "baseline"]
z_rows = [r for r in rows if r["group"] != "baseline"]
print(f"\n2. STRUCTURED Z-SAMPLING vs NAIVE Z BASELINES:")
print(f"   Best baseline:       {bl_rows[0]['name']}  →  {bl_rows[0]['mean_d_avg_mean']:.4f}")
print(f"   Best z-sampling:     {z_rows[0]['name']}  →  {z_rows[0]['mean_d_avg_mean']:.4f}")
if z_rows[0]['mean_d_avg_mean'] < bl_rows[0]['mean_d_avg_mean']:
    pct = (bl_rows[0]['mean_d_avg_mean'] - z_rows[0]['mean_d_avg_mean']) / bl_rows[0]['mean_d_avg_mean'] * 100
    print(f"   → Structured z-sampling HELPS: {pct:.1f}% improvement over best baseline")
else:
    pct = (z_rows[0]['mean_d_avg_mean'] - bl_rows[0]['mean_d_avg_mean']) / bl_rows[0]['mean_d_avg_mean'] * 100
    print(f"   → Structured z-sampling does NOT help: {pct:.1f}% worse than best baseline")

# Medians by group
print(f"\n   Group medians (mean_d_avg):")
for g in ["A", "B", "C", "D", "baseline"]:
    g_rows = [r for r in rows if r["group"] == g]
    if g_rows:
        med = np.median([r["mean_d_avg_mean"] for r in g_rows])
        print(f"     {g:>10}: {med:.4f}  (n={len(g_rows)})")

# vMF vs sensitivity
print(f"\n3. vMF TASK-BINDING vs SENSITIVITY ROUTING:")
vmf_rows = [r for r in rows if r["group"] in ["A", "B"]]
sens_rows = [r for r in rows if r["group"] in ["C", "D"]]
best_vmf = min(vmf_rows, key=lambda r: r["mean_d_avg_mean"])
best_sens = min(sens_rows, key=lambda r: r["mean_d_avg_mean"])
print(f"   Best vMF:       {best_vmf['name']}  →  {best_vmf['mean_d_avg_mean']:.4f}")
print(f"   Best sensitivity: {best_sens['name']}  →  {best_sens['mean_d_avg_mean']:.4f}")
if best_vmf['mean_d_avg_mean'] < best_sens['mean_d_avg_mean']:
    print(f"   → vMF wins")
else:
    print(f"   → Sensitivity routing wins")

# Distillation compounding
print(f"\n4. DISTILLATION COMPOUNDING (no-distill → +distill):")
for ml in method_labels:
    v0, _ = get_val(no_distill_map[ml])
    v1, _ = get_val(distill_map[ml])
    if v0 is not None and v1 is not None:
        delta = v1 - v0
        pct = delta / v0 * 100
        arrow = "↓ better" if delta < 0 else "↑ worse"
        print(f"   {ml:<15}: {v0:.4f} → {v1:.4f}  ({delta:+.4f}, {pct:+.1f}% {arrow})")

# no-distill avg vs +distill avg
nd_vals = [r["mean_d_avg_mean"] for r in rows if r["group"] in ["A", "C"]]
d_vals = [r["mean_d_avg_mean"] for r in rows if r["group"] in ["B", "D"]]
print(f"\n   Avg no-distill: {np.mean(nd_vals):.4f},  Avg +distill: {np.mean(d_vals):.4f}")
if np.mean(d_vals) < np.mean(nd_vals):
    print(f"   → Distillation compounds with z-structure on average")
else:
    print(f"   → Distillation does NOT compound on average")

# Memory-plasticity balance
print(f"\n5. MEMORY vs PLASTICITY BALANCE:")
print(f"   Config with best balance (lowest |mem - plas| on mean_d, among top-10):")
top10 = rows[:10]
for r in sorted(top10, key=lambda r: abs(r["memory_mean_d"] - r["plasticity_mean_d"])):
    balance = abs(r["memory_mean_d"] - r["plasticity_mean_d"])
    print(f"   {r['name']:<25}  mem={r['memory_mean_d']:.4f}  plas={r['plasticity_mean_d']:.4f}  |diff|={balance:.4f}  avg={r['mean_d_avg_mean']:.4f}")

print(f"\n{'='*80}")
print("DONE")
