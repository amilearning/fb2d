#!/usr/bin/env python3
"""Generate a comprehensive LaTeX table of ALL 113 FB continual-learning configs, compile to PDF."""

import os, glob, sys
import numpy as np

# ============================================================================
# Helper: load seeds from a config directory
# ============================================================================
def load_seeds(config_path):
    """Return (mean_d_arr, s10_arr) each shape (n_seeds, 3), or (None, None)."""
    seed_dirs = sorted(glob.glob(os.path.join(config_path, "seed*")))
    if not seed_dirs:
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


# ============================================================================
# Collect ALL configs
# ============================================================================
all_records = []

# --- Baselines ---
NAIVE_ROOT = "/home/frl/FB2D/checkpoints_naive_seq_q123_z3"
if os.path.isdir(NAIVE_ROOT):
    naive_mds, naive_s10s = [], []
    for d in sorted(os.listdir(NAIVE_ROOT)):
        dp = os.path.join(NAIVE_ROOT, d)
        md_path = os.path.join(dp, "mean_d.npy")
        s10_path = os.path.join(dp, "s10.npy")
        if os.path.exists(md_path) and os.path.exists(s10_path):
            naive_mds.append(np.load(md_path).flatten()[:3])
            naive_s10s.append(np.load(s10_path).flatten()[:3])
    if naive_mds:
        all_records.append({
            "sweep": "baseline", "config": "naive_seq",
            "label": "Naive Sequential",
            "z_sampling": "random", "distill_target": "---",
            "distill_input": "---", "K": "---", "ema": "---", "tau": "---", "alpha": "---",
        })

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
        all_records.append({
            "sweep": "baseline", "config": "cumulative",
            "label": "Cumulative (task-incr.)",
            "z_sampling": "random", "distill_target": "---",
            "distill_input": "---", "K": "---", "ema": "---", "tau": "---", "alpha": "---",
        })

# --- B x D x C Standalone (48 configs) ---
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
        if B == "pi":
            target = f"pi({D})"
        else:
            target = f"{B}({D})"
        all_records.append({
            "sweep": "BxDxC", "config": config_dir,
            "label": config_dir,
            "z_sampling": "random",
            "distill_target": target,
            "distill_input": C,
            "K": "---", "ema": "---", "tau": "---", "alpha": "1.0",
        })

# --- Combined pi+X (24 configs) ---
COMBINED_ROOT = "/home/frl/FB2D/checkpoints_distill_combined_real"
if os.path.isdir(COMBINED_ROOT):
    for config_dir in sorted(os.listdir(COMBINED_ROOT)):
        config_path = os.path.join(COMBINED_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        parts = config_dir.split("_")
        C = parts[-1]  # current or replay
        pi_type = parts[0]  # piGram or piL2
        X = "_".join(parts[1:-1])  # F, B, FB, M, Q, FBM
        pi_loss = "gram" if pi_type == "piGram" else "l2"
        target = f"pi({pi_loss})+{X}(l2)"
        all_records.append({
            "sweep": "combined", "config": config_dir,
            "label": config_dir,
            "z_sampling": "random",
            "distill_target": target,
            "distill_input": C,
            "K": "---", "ema": "---", "tau": "---", "alpha": "1.0",
        })

# --- Z-sampling (24 configs) ---
Z_ROOT = "/home/frl/FB2D/checkpoints_z_sampling_real"
Z_CONFIG_META = {
    "A1_vmf_bind":  {"z": "vMF-bind",  "target": "---",                "di": "---",    "ema": "0.5"},
    "A2_vmf_mix":   {"z": "vMF-mix",   "target": "---",                "di": "---",    "ema": "0.5"},
    "B1_vmf_bind":  {"z": "vMF-bind",  "target": "pi(gram)+FB(l2)",    "di": "replay", "ema": "0.5"},
    "B2_vmf_mix":   {"z": "vMF-mix",   "target": "pi(gram)+FB(l2)",    "di": "replay", "ema": "0.5"},
}
SENS_SOURCES = ["F", "B", "FB", "pi", "piFB"]
for i, src in enumerate(SENS_SOURCES, 1):
    Z_CONFIG_META[f"C{i}_smr_{src}"]  = {"z": f"SMR({src})", "target": "---", "di": "---", "ema": "0.5"}
for i, src in enumerate(SENS_SOURCES, 6):
    Z_CONFIG_META[f"C{i}_fdws_{src}"] = {"z": f"FDWS({src})", "target": "---", "di": "---", "ema": "0.5"}
for i, src in enumerate(SENS_SOURCES, 1):
    Z_CONFIG_META[f"D{i}_smr_{src}"]  = {"z": f"SMR({src})", "target": "pi(gram)+FB(l2)", "di": "replay", "ema": "0.5"}
for i, src in enumerate(SENS_SOURCES, 6):
    Z_CONFIG_META[f"D{i}_fdws_{src}"] = {"z": f"FDWS({src})", "target": "pi(gram)+FB(l2)", "di": "replay", "ema": "0.5"}

if os.path.isdir(Z_ROOT):
    for config_dir in sorted(os.listdir(Z_ROOT)):
        config_path = os.path.join(Z_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        meta = Z_CONFIG_META.get(config_dir, {})
        has_distill = meta.get("target", "---") != "---"
        all_records.append({
            "sweep": "z_sampling", "config": config_dir,
            "label": config_dir,
            "z_sampling": meta.get("z", "?"),
            "distill_target": meta.get("target", "---"),
            "distill_input": meta.get("di", "---"),
            "K": "3", "ema": meta.get("ema", "0.5"),
            "tau": "---", "alpha": "1.0" if has_distill else "---",
        })

# --- vMF K sweep (9 configs) ---
VMF_ROOT = "/home/frl/FB2D/checkpoints_vmf_sweep_real"
if os.path.isdir(VMF_ROOT):
    for config_dir in sorted(os.listdir(VMF_ROOT)):
        config_path = os.path.join(VMF_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        parts = config_dir.split("_")
        K_str = parts[0].replace("vmfMix", "")
        distill_label = parts[1]  # Bpi, Fpi, FBpi
        C = parts[2]
        fb_map = {"Bpi": "B", "Fpi": "F", "FBpi": "FB"}
        fb_part = fb_map.get(distill_label, "?")
        target = f"pi(gram)+{fb_part}(l2)"
        all_records.append({
            "sweep": "vmf_sweep", "config": config_dir,
            "label": config_dir,
            "z_sampling": "vMF-mix",
            "distill_target": target,
            "distill_input": C,
            "K": K_str, "ema": "---", "tau": "---", "alpha": "1.0",
        })

# --- FDWS tau sweep (6 configs) ---
FDWS_ROOT = "/home/frl/FB2D/checkpoints_fdws_sweep_real"
TAU_MAP = {"tau05": "0.5", "tau10": "1.0", "tau50": "5.0"}
if os.path.isdir(FDWS_ROOT):
    for config_dir in sorted(os.listdir(FDWS_ROOT)):
        config_path = os.path.join(FDWS_ROOT, config_dir)
        if not os.path.isdir(config_path):
            continue
        parts = config_dir.split("_")
        tau_key = parts[0]
        sens_key = parts[1]
        tau_val = TAU_MAP.get(tau_key, "?")
        sens_src = "current" if "Current" in sens_key else "replay"
        all_records.append({
            "sweep": "fdws_sweep", "config": config_dir,
            "label": config_dir,
            "z_sampling": f"FDWS(FB)",
            "distill_target": "pi(gram)+FB(l2)",
            "distill_input": sens_src,
            "K": "---", "ema": "---", "tau": tau_val, "alpha": "1.0",
        })

print(f"Collected {len(all_records)} configs total.")

# Assign M1..M{N} IDs
for i, r in enumerate(all_records):
    r["method_id"] = f"M{i+1}"

# ============================================================================
# Generate LaTeX
# ============================================================================

def esc(s):
    """Escape LaTeX special characters."""
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("~", r"\textasciitilde{}")

SWEEP_ORDER = [
    ("baseline",    "Baselines"),
    ("BxDxC",       r"B$\times$C$\times$D Standalone Distillation (48 configs)"),
    ("combined",    r"Combined $\pi$+X Joint Distillation (24 configs)"),
    ("z_sampling",  "Z-Sampling Methods (24 configs)"),
    ("vmf_sweep",   "vMF Component Count Sweep (9 configs)"),
    ("fdws_sweep",  r"FDWS Temperature $\tau$ Sweep (6 configs)"),
]

TEX_PATH = "/home/frl/FB2D/full_method_table.tex"

with open(TEX_PATH, "w") as f:
    f.write(r"""\documentclass[8pt,landscape]{extarticle}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[landscape,left=1cm,right=1cm,top=1.5cm,bottom=1.5cm]{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{array}

\definecolor{sectionbg}{RGB}{200,210,230}

\pagestyle{plain}

\begin{document}

\begin{center}
{\Large\bfseries Complete Method Configuration Table --- FB Continual Learning}\\[4pt]
{\small Shared: $z_\text{dim}=32$, hidden$=256$, lr$=10^{-4}$, batch$=512$, 60k updates/stage, Q1$\to$Q2$\to$Q3, 3 seeds each}
\end{center}

\vspace{6pt}

\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{r l l l l >{\centering\arraybackslash}p{1cm} >{\centering\arraybackslash}p{1cm} >{\centering\arraybackslash}p{1cm} >{\centering\arraybackslash}p{1cm}}
\toprule
\textbf{ID} & \textbf{Label} & \textbf{z-sampling} & \textbf{Distill Target} & \textbf{Distill Input} & \textbf{K} & \textbf{EMA} & \textbf{$\tau$} & \textbf{$\alpha$} \\
\midrule
\endfirsthead

\toprule
\textbf{ID} & \textbf{Label} & \textbf{z-sampling} & \textbf{Distill Target} & \textbf{Distill Input} & \textbf{K} & \textbf{EMA} & \textbf{$\tau$} & \textbf{$\alpha$} \\
\midrule
\endhead

\midrule
\multicolumn{9}{r}{\textit{Continued on next page\ldots}} \\
\endfoot

\bottomrule
\endlastfoot

""")

    for sweep_key, sweep_title in SWEEP_ORDER:
        sweep_records = [r for r in all_records if r["sweep"] == sweep_key]
        if not sweep_records:
            continue
        # Section header row
        f.write(r"\rowcolor{sectionbg}" + "\n")
        f.write(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{sweep_title}}}}} \\\\\n")
        f.write(r"\midrule" + "\n")

        for r in sweep_records:
            mid = r["method_id"]
            label = esc(r["label"])
            zsamp = esc(r["z_sampling"])
            dtarget = esc(r["distill_target"])
            dinput = esc(r["distill_input"])
            K = r["K"]
            ema = r["ema"]
            tau = r["tau"]
            alpha = r["alpha"]
            f.write(f"{mid} & {label} & {zsamp} & {dtarget} & {dinput} & {K} & {ema} & {tau} & {alpha} \\\\\n")

        f.write(r"\addlinespace[3pt]" + "\n")

    f.write(r"""
\end{longtable}

\end{document}
""")

print(f"LaTeX written to {TEX_PATH}")
print(f"Now compile with: pdflatex -output-directory=/home/frl/FB2D {TEX_PATH}")
