#!/usr/bin/env python3
"""Master method table: load ALL FB continual-learning sweep results, rank, select top 30."""

import os, glob, csv, sys
import numpy as np

# ============================================================================
# Shared constants
# ============================================================================
SHARED_PARAMS = {
    "z_dim": 32, "hidden_dim": 256, "lr": "1e-4",
    "batch_size": 512, "updates_per_stage": "60k", "task_sequence": "Q1->Q2->Q3",
}

# ============================================================================
# Helper: load seeds from a config directory
# ============================================================================
def load_seeds(config_path):
    """Return (mean_d_arr, s10_arr) each shape (n_seeds, 3), or (None, None)."""
    seed_dirs = sorted(glob.glob(os.path.join(config_path, "seed*")))
    if not seed_dirs:
        # Maybe the npy files are directly in config_path (baselines)
        md_path = os.path.join(config_path, "mean_d.npy")
        s10_path = os.path.join(config_path, "s10.npy")
        if os.path.exists(md_path) and os.path.exists(s10_path):
            md = np.load(md_path).flatten()[:3]
            s10 = np.load(s10_path).flatten()[:3]
            return md[None, :], s10[None, :]
        return None, None
    mean_ds, s10s = [], []
    for sd in seed_dirs:
        md_path = os.path.join(sd, "mean_d.npy")
        s10_path = os.path.join(sd, "s10.npy")
        if os.path.exists(md_path) and os.path.exists(s10_path):
            mean_ds.append(np.load(md_path).flatten()[:3])
            s10s.append(np.load(s10_path).flatten()[:3])
    if not mean_ds:
        return None, None
    return np.array(mean_ds), np.array(s10s)


def compute_stats(mean_d_arr, s10_arr):
    """Return dict of summary stats from (n_seeds, 3) arrays."""
    md_avg = mean_d_arr.mean(axis=1)  # per-seed average across Q1,Q2,Q3
    s10_avg = s10_arr.mean(axis=1)
    return {
        "n_seeds": mean_d_arr.shape[0],
        "mean_d_avg_mean": md_avg.mean(),
        "mean_d_avg_std": md_avg.std(),
        "s10_avg_mean": s10_avg.mean(),
        "s10_avg_std": s10_avg.std(),
        "s10_Q1_mean": s10_arr[:, 0].mean(),
        "s10_Q1_std": s10_arr[:, 0].std(),
        "s10_Q3_mean": s10_arr[:, 2].mean(),
        "s10_Q3_std": s10_arr[:, 2].std(),
    }


# ============================================================================
# 1. Load all results
# ============================================================================
all_records = []

# --- 1a. Naive sequential baseline ---
NAIVE_ROOT = "/home/frl/FB2D/checkpoints_naive_seq_q123_z3"
if os.path.isdir(NAIVE_ROOT):
    # Collect seeds that have results
    naive_mds, naive_s10s = [], []
    for d in sorted(os.listdir(NAIVE_ROOT)):
        dp = os.path.join(NAIVE_ROOT, d)
        md_path = os.path.join(dp, "mean_d.npy")
        s10_path = os.path.join(dp, "s10.npy")
        if os.path.exists(md_path) and os.path.exists(s10_path):
            naive_mds.append(np.load(md_path).flatten()[:3])
            naive_s10s.append(np.load(s10_path).flatten()[:3])
    if naive_mds:
        stats = compute_stats(np.array(naive_mds), np.array(naive_s10s))
        all_records.append({
            "sweep": "baseline",
            "config": "naive_seq",
            "label": "Naive Seq",
            "z_sampling": "random",
            "distill_fb": "none",
            "distill_pi": "none",
            "distill_input": "none",
            "alpha_distill": "-",
            "ema_decay": "-",
            "extra_params": "",
            **stats,
        })

# --- 1b. Cumulative (task-incremental) baseline ---
CUMUL_ROOT = "/home/frl/FB2D/checkpoints_taskincr_q123_z3"
if os.path.isdir(CUMUL_ROOT):
    cumul_mds, cumul_s10s = [], []
    for d in sorted(os.listdir(CUMUL_ROOT)):
        dp = os.path.join(CUMUL_ROOT, d)
        md_path = os.path.join(dp, "mean_d.npy")
        s10_path = os.path.join(dp, "s10.npy")
        if os.path.exists(md_path) and os.path.exists(s10_path):
            cumul_mds.append(np.load(md_path).flatten()[:3])
            cumul_s10s.append(np.load(s10_path).flatten()[:3])
    if cumul_mds:
        stats = compute_stats(np.array(cumul_mds), np.array(cumul_s10s))
        all_records.append({
            "sweep": "baseline",
            "config": "cumulative",
            "label": "Cumulative",
            "z_sampling": "random",
            "distill_fb": "none",
            "distill_pi": "none",
            "distill_input": "none",
            "alpha_distill": "-",
            "ema_decay": "-",
            "extra_params": "",
            **stats,
        })

# --- 1c. B x D x C standalone sweep (48 configs) ---
DISTILL_ROOT = "/home/frl/FB2D/checkpoints_distill_real"
if os.path.isdir(DISTILL_ROOT):
    for config_dir in sorted(os.listdir(DISTILL_ROOT)):
        config_path = os.path.join(DISTILL_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        parts = config_dir.split("_")
        C = parts[-1]   # current or replay
        D = parts[-2]   # l2, cosine, contrastive, gram
        B = "_".join(parts[:-2])  # F, B, FB, M, Q, pi
        md_arr, s10_arr = load_seeds(config_path)
        if md_arr is None:
            continue
        stats = compute_stats(md_arr, s10_arr)
        # Determine if this is actor-side or FB-side distillation
        if B == "pi":
            distill_fb = "none"
            distill_pi = f"pi_{D}"
        else:
            distill_fb = f"{B}_{D}"
            distill_pi = "none"
        all_records.append({
            "sweep": "BxDxC",
            "config": config_dir,
            "label": f"{B}_{D}_{C[:3]}",
            "z_sampling": "random",
            "distill_fb": distill_fb,
            "distill_pi": distill_pi,
            "distill_input": C,
            "alpha_distill": "1.0",
            "ema_decay": "-",
            "extra_params": "",
            **stats,
        })

# --- 1d. Combined pi+X sweep (24 configs) ---
COMBINED_ROOT = "/home/frl/FB2D/checkpoints_distill_combined_real"
if os.path.isdir(COMBINED_ROOT):
    for config_dir in sorted(os.listdir(COMBINED_ROOT)):
        config_path = os.path.join(COMBINED_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        # Parse: piGram_FB_current or piL2_FBM_replay etc.
        parts = config_dir.split("_")
        C = parts[-1]  # current or replay
        pi_type = parts[0]  # piGram or piL2
        X = "_".join(parts[1:-1])  # F, B, FB, M, Q, FBM
        md_arr, s10_arr = load_seeds(config_path)
        if md_arr is None:
            continue
        stats = compute_stats(md_arr, s10_arr)
        pi_loss = "pi_gram" if pi_type == "piGram" else "pi_l2"
        all_records.append({
            "sweep": "combined",
            "config": config_dir,
            "label": f"{pi_type}+{X}_{C[:3]}",
            "z_sampling": "random",
            "distill_fb": f"{X}_l2",
            "distill_pi": pi_loss,
            "distill_input": C,
            "alpha_distill": "1.0",
            "ema_decay": "-",
            "extra_params": "",
            **stats,
        })

# --- 1e. Z-sampling sweep (24 configs) ---
Z_ROOT = "/home/frl/FB2D/checkpoints_z_sampling_real"
Z_CONFIG_META = {
    "A1_vmf_bind":  {"z": "vMF bind K=3",  "dfb": "none",    "dpi": "none",    "di": "none"},
    "A2_vmf_mix":   {"z": "vMF mix K=3",   "dfb": "none",    "dpi": "none",    "di": "none"},
    "B1_vmf_bind":  {"z": "vMF bind K=3",  "dfb": "FB_l2",   "dpi": "pi_gram", "di": "replay"},
    "B2_vmf_mix":   {"z": "vMF mix K=3",   "dfb": "FB_l2",   "dpi": "pi_gram", "di": "replay"},
}
SENS_SOURCES = ["F", "B", "FB", "pi", "piFB"]
for i, src in enumerate(SENS_SOURCES, 1):
    Z_CONFIG_META[f"C{i}_smr_{src}"]  = {"z": f"SMR {src}", "dfb": "none",  "dpi": "none",    "di": "none"}
for i, src in enumerate(SENS_SOURCES, 6):
    Z_CONFIG_META[f"C{i}_fdws_{src}"] = {"z": f"FDWS {src}", "dfb": "none",  "dpi": "none",    "di": "none"}
for i, src in enumerate(SENS_SOURCES, 1):
    Z_CONFIG_META[f"D{i}_smr_{src}"]  = {"z": f"SMR {src}", "dfb": "FB_l2", "dpi": "pi_gram", "di": "replay"}
for i, src in enumerate(SENS_SOURCES, 6):
    Z_CONFIG_META[f"D{i}_fdws_{src}"] = {"z": f"FDWS {src}", "dfb": "FB_l2", "dpi": "pi_gram", "di": "replay"}

if os.path.isdir(Z_ROOT):
    for config_dir in sorted(os.listdir(Z_ROOT)):
        config_path = os.path.join(Z_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        md_arr, s10_arr = load_seeds(config_path)
        if md_arr is None:
            continue
        stats = compute_stats(md_arr, s10_arr)
        meta = Z_CONFIG_META.get(config_dir, {})
        has_distill = meta.get("dfb", "none") != "none"
        all_records.append({
            "sweep": "z_sampling",
            "config": config_dir,
            "label": config_dir.replace("_", " "),
            "z_sampling": meta.get("z", "?"),
            "distill_fb": meta.get("dfb", "none"),
            "distill_pi": meta.get("dpi", "none"),
            "distill_input": meta.get("di", "none"),
            "alpha_distill": "1.0" if has_distill else "-",
            "ema_decay": "0.5",
            "extra_params": "kappa=10" if "vmf" in config_dir else "",
            **stats,
        })

# --- 1f. vMF component sweep (9 configs) ---
VMF_ROOT = "/home/frl/FB2D/checkpoints_vmf_sweep_real"
if os.path.isdir(VMF_ROOT):
    for config_dir in sorted(os.listdir(VMF_ROOT)):
        config_path = os.path.join(VMF_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        # Parse: vmfMix10_Bpi_replay -> K=10, distill=B+pi, input=replay
        parts = config_dir.split("_")
        K_str = parts[0].replace("vmfMix", "")  # "5", "10", "20"
        distill_label = parts[1]  # "Bpi", "Fpi", "FBpi"
        C = parts[2]  # "replay"
        # Map distill_label to FB-side target
        fb_target_map = {"Bpi": "B_l2", "Fpi": "F_l2", "FBpi": "FB_l2"}
        md_arr, s10_arr = load_seeds(config_path)
        if md_arr is None:
            continue
        stats = compute_stats(md_arr, s10_arr)
        all_records.append({
            "sweep": "vmf_sweep",
            "config": config_dir,
            "label": f"vMFmix{K_str}+{distill_label}",
            "z_sampling": f"vMF mix K={K_str}",
            "distill_fb": fb_target_map.get(distill_label, "?"),
            "distill_pi": "pi_gram",
            "distill_input": C,
            "alpha_distill": "1.0",
            "ema_decay": "-",
            "extra_params": f"K={K_str},kappa=10",
            **stats,
        })

# --- 1g. FDWS temperature sweep (6 configs) ---
FDWS_ROOT = "/home/frl/FB2D/checkpoints_fdws_sweep_real"
TAU_MAP = {"tau05": 0.5, "tau10": 1.0, "tau50": 5.0}
if os.path.isdir(FDWS_ROOT):
    for config_dir in sorted(os.listdir(FDWS_ROOT)):
        config_path = os.path.join(FDWS_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        # Parse: tau05_sensCurrent -> tau=0.5, sens=current
        parts = config_dir.split("_")
        tau_key = parts[0]  # tau05, tau10, tau50
        sens_key = parts[1]  # sensCurrent, sensReplay
        tau_val = TAU_MAP.get(tau_key, "?")
        sens_src = "current" if "Current" in sens_key else "replay"
        md_arr, s10_arr = load_seeds(config_path)
        if md_arr is None:
            continue
        stats = compute_stats(md_arr, s10_arr)
        all_records.append({
            "sweep": "fdws_sweep",
            "config": config_dir,
            "label": f"FDWS t={tau_val} s={sens_src[:3]}",
            "z_sampling": f"FDWS FB tau={tau_val}",
            "distill_fb": "FB_l2",
            "distill_pi": "pi_gram",
            "distill_input": "replay",
            "alpha_distill": "1.0",
            "ema_decay": "0.0",
            "extra_params": f"tau={tau_val},sens={sens_src}",
            **stats,
        })


print(f"Loaded {len(all_records)} total configs across all sweeps.\n")

# ============================================================================
# 2. Rank by mean_d_avg (lower is better)
# ============================================================================
all_records.sort(key=lambda r: r["mean_d_avg_mean"])

# Print quick overview
print("=== ALL CONFIGS RANKED (mean_d_avg, lower=better) ===")
for i, r in enumerate(all_records, 1):
    print(f"  {i:3d}  {r['mean_d_avg_mean']:.4f}+/-{r['mean_d_avg_std']:.4f}  "
          f"s10={r['s10_avg_mean']:.3f}  {r['sweep']:12s}  {r['config']}")

# ============================================================================
# 3. Select top 30 interesting methods
# ============================================================================
# Strategy:
#   - Top ~20 by mean_d
#   - Naive baseline (always)
#   - Cumulative baseline (always)
#   - A few representative "bad" methods from each sweep for contrast

selected = []
selected_configs = set()

# Top 20 by mean_d
for r in all_records[:20]:
    if r["config"] not in selected_configs:
        selected.append(r)
        selected_configs.add(r["config"])

# Ensure baselines are included
for r in all_records:
    if r["config"] in ("naive_seq", "cumulative") and r["config"] not in selected_configs:
        selected.append(r)
        selected_configs.add(r["config"])

# Add representative "bad" methods from each sweep for contrast
# Pick the worst from each sweep that isn't already selected
sweeps_for_contrast = ["BxDxC", "combined", "z_sampling", "vmf_sweep", "fdws_sweep"]
for sw in sweeps_for_contrast:
    sw_records = [r for r in all_records if r["sweep"] == sw and r["config"] not in selected_configs]
    if sw_records:
        # Pick worst (last when sorted by mean_d_avg ascending)
        worst = sw_records[-1]
        selected.append(worst)
        selected_configs.add(worst["config"])

# Also add median-rank method from BxDxC for contrast
bxdxc_records = [r for r in all_records if r["sweep"] == "BxDxC" and r["config"] not in selected_configs]
if bxdxc_records:
    mid = bxdxc_records[len(bxdxc_records) // 2]
    selected.append(mid)
    selected_configs.add(mid["config"])

# Fill up to 30 if needed from next-best overall
for r in all_records:
    if len(selected) >= 30:
        break
    if r["config"] not in selected_configs:
        selected.append(r)
        selected_configs.add(r["config"])

# Re-sort by mean_d_avg and assign M1..M30
selected.sort(key=lambda r: r["mean_d_avg_mean"])
for i, r in enumerate(selected):
    r["method_id"] = f"M{i+1}"

# ============================================================================
# 4. Print table to stdout
# ============================================================================
print(f"\n{'='*160}")
print(f"MASTER METHOD TABLE — Top {len(selected)} methods")
print(f"Shared parameters: z_dim={SHARED_PARAMS['z_dim']}, hidden={SHARED_PARAMS['hidden_dim']}, "
      f"lr={SHARED_PARAMS['lr']}, batch={SHARED_PARAMS['batch_size']}, "
      f"updates/stage={SHARED_PARAMS['updates_per_stage']}, tasks={SHARED_PARAMS['task_sequence']}")
print(f"{'='*160}")

hdr = (f"{'ID':>4} {'Label':<26} {'Sweep':<12} {'z-sampling':<18} "
       f"{'Distill FB':<12} {'Distill pi':<10} {'Input':<8} {'alpha':>5} {'EMA':>5} "
       f"{'mean_d_avg':>16} {'s10_avg':>14} {'s10_Q1(mem)':>14} {'s10_Q3(plas)':>14} {'n':>3}")
print(hdr)
print("-" * len(hdr))

for r in selected:
    md_str = f"{r['mean_d_avg_mean']:.4f}+/-{r['mean_d_avg_std']:.4f}"
    s10_str = f"{r['s10_avg_mean']:.3f}+/-{r['s10_avg_std']:.3f}"
    s10q1_str = f"{r['s10_Q1_mean']:.3f}+/-{r['s10_Q1_std']:.3f}"
    s10q3_str = f"{r['s10_Q3_mean']:.3f}+/-{r['s10_Q3_std']:.3f}"
    print(f"{r['method_id']:>4} {r['label']:<26} {r['sweep']:<12} {r['z_sampling']:<18} "
          f"{r['distill_fb']:<12} {r['distill_pi']:<10} {r['distill_input']:<8} "
          f"{r['alpha_distill']:>5} {r['ema_decay']:>5} "
          f"{md_str:>16} {s10_str:>14} {s10q1_str:>14} {s10q3_str:>14} {r['n_seeds']:>3}")

# ============================================================================
# 5a. Save CSV
# ============================================================================
CSV_PATH = "/home/frl/FB2D/master_method_table.csv"
csv_fields = [
    "method_id", "label", "sweep", "z_sampling",
    "distill_fb", "distill_pi", "distill_input",
    "alpha_distill", "ema_decay", "extra_params",
    "mean_d_avg_mean", "mean_d_avg_std",
    "s10_avg_mean", "s10_avg_std",
    "s10_Q1_mean", "s10_Q1_std",
    "s10_Q3_mean", "s10_Q3_std",
    "n_seeds",
]
with open(CSV_PATH, "w", newline="") as f:
    # Write shared params as comment header
    f.write(f"# Shared: z_dim={SHARED_PARAMS['z_dim']}, hidden={SHARED_PARAMS['hidden_dim']}, "
            f"lr={SHARED_PARAMS['lr']}, batch={SHARED_PARAMS['batch_size']}, "
            f"updates/stage={SHARED_PARAMS['updates_per_stage']}, tasks={SHARED_PARAMS['task_sequence']}\n")
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    for r in selected:
        writer.writerow(r)
print(f"\nCSV saved to {CSV_PATH}")

# ============================================================================
# 5b. Generate LaTeX table
# ============================================================================
TEX_PATH = "/home/frl/FB2D/master_method_table.tex"

def esc(s):
    """Escape LaTeX special characters."""
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")

with open(TEX_PATH, "w") as f:
    f.write(r"""\documentclass[10pt,landscape]{article}
\usepackage[margin=0.5in]{geometry}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{xcolor}
\usepackage{colortbl}

\definecolor{bestrow}{rgb}{0.85,0.95,0.85}
\definecolor{baserow}{rgb}{0.95,0.90,0.80}

\begin{document}
\pagestyle{empty}

\begin{center}
{\Large\bfseries Master Method Table: FB Continual Learning Sweeps}\\[4pt]
{\small Shared parameters: $z_\text{dim}=32$, hidden$=256$, lr$=10^{-4}$, batch$=512$, 60k updates/stage, tasks Q1$\to$Q2$\to$Q3}
\end{center}

\vspace{4pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{rl l l l l l r r rr rr rr r}
\toprule
""")
    f.write(r"ID & Label & Sweep & $z$-sampling & Distill$_\text{FB}$ & Distill$_\pi$ & Input "
            r"& $\alpha$ & EMA "
            r"& \multicolumn{2}{c}{mean\_d$_\text{avg}$} "
            r"& \multicolumn{2}{c}{s10$_\text{avg}$} "
            r"& \multicolumn{2}{c}{s10$_\text{Q1}$ (mem)} "
            r"& s10$_\text{Q3}$ (plas) \\" + "\n")
    f.write(r" & & & & & & & & & mean & $\pm$std & mean & $\pm$std & mean & $\pm$std & mean \\" + "\n")
    f.write(r"\midrule" + "\n")

    for r in selected:
        # Highlight best (M1) and baselines
        if r["method_id"] == "M1":
            f.write(r"\rowcolor{bestrow}" + "\n")
        elif r["config"] in ("naive_seq", "cumulative"):
            f.write(r"\rowcolor{baserow}" + "\n")

        f.write(f"{r['method_id']} & {esc(r['label'])} & {esc(r['sweep'])} & "
                f"{esc(r['z_sampling'])} & {esc(r['distill_fb'])} & {esc(r['distill_pi'])} & "
                f"{esc(r['distill_input'])} & {r['alpha_distill']} & {r['ema_decay']} & "
                f"{r['mean_d_avg_mean']:.4f} & {r['mean_d_avg_std']:.4f} & "
                f"{r['s10_avg_mean']:.3f} & {r['s10_avg_std']:.3f} & "
                f"{r['s10_Q1_mean']:.3f} & {r['s10_Q1_std']:.3f} & "
                f"{r['s10_Q3_mean']:.3f} "
                r"\\" + "\n")

    f.write(r"""\bottomrule
\end{tabular}
\end{adjustbox}

\vspace{6pt}
{\footnotesize
\textbf{Metrics:} mean\_d$_\text{avg}$ = mean final distance averaged over Q1, Q2, Q3 (lower is better);
s10 = success rate at $r{<}0.10$ threshold;
s10$_\text{Q1}$ (mem) = memory retention on first task after training all three;
s10$_\text{Q3}$ (plas) = plasticity on most recent task.\\
\textbf{Sweeps:} baseline = no distillation baselines;
BxDxC = single distill target $\times$ loss $\times$ input;
combined = $\pi$ + X joint distillation;
z\_sampling = structured $z$-sampling methods;
vmf\_sweep = vMF component count sweep;
fdws\_sweep = FDWS temperature sweep.
}

\end{document}
""")

print(f"LaTeX saved to {TEX_PATH}")
print(f"\nDone. {len(selected)} methods in the master table.")
