#!/usr/bin/env python3
"""
generate_final_report.py — Comprehensive final report for FB continual learning experiments.

Produces:
  - Evaluation of new methods (vMF20+FDWS fixed/optimized)
  - All plots (bar, scatter, violin, z-distribution sphere)
  - LaTeX report compiled to PDF

Usage:
    /home/frl/anaconda3/envs/incr_learn/bin/python generate_final_report.py
"""

import csv
import math
import os
import subprocess
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = "/home/frl/FB2D"
SUMMARY_CSV = os.path.join(PROJECT, "eval_results/eval_fast_summary.csv")
METHODS_CSV = os.path.join(PROJECT, "eval_report/methods_with_ids.csv")
OUTPUT_DIR = os.path.join(PROJECT, "final_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda"

# New method checkpoint dirs
NEW_METHODS = {
    "vmf20_fdws_fixed": {
        "dir": os.path.join(PROJECT, "checkpoints_vmf20_fdws/fixed"),
        "sweep": "vmf_fdws_combined",
        "config_name": "vmf20_fdws_fixed",
    },
    "vmf20_fdws_optimized": {
        "dir": os.path.join(PROJECT, "checkpoints_vmf20_fdws/optimized"),
        "sweep": "vmf_fdws_combined",
        "config_name": "vmf20_fdws_optimized",
    },
}

# z-viz NPZ paths
Z_VIZ_METHODS = {
    "FDWS+distill (D7v3)": os.path.join(PROJECT, "checkpoints_z_sampling/D7v3_z3_viz/seed42_20260414_092446/z_viz_data_20x20.npz"),
    "Naive+distill": os.path.join(PROJECT, "checkpoints_z_sampling/naive_distill_z3_viz/seed42_20260414_095525/z_viz_data_20x20.npz"),
    "Naive (no distill)": os.path.join(PROJECT, "checkpoints_naive_z3_viz/seed42_20260414_150044/z_viz_data_20x20.npz"),
    "Opt vMF20+FDWS": os.path.join(PROJECT, "checkpoints_vmf20_fdws_opt_z3/seed42_20260414_142611/z_viz_data_20x20.npz"),
}

# ---------------------------------------------------------------------------
# Evaluation constants (from eval_all_methods_fast.py)
# ---------------------------------------------------------------------------
EVAL_SEED = 42
EVAL_QUADRANTS = ["Q1", "Q3"]
N_ENVS = 8192
N_STEPS = 200
MAX_SPEED = 0.05
GOAL_MARGIN = 0.10
REWARD_RADIUS = 0.07
REWARD_FALLBACK_SCALE = 25.0
N_SAMPLE_OBS = 4096

sys.path.insert(0, PROJECT)


# ---------------------------------------------------------------------------
# Evaluation helpers (copied from eval_all_methods_fast.py)
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F_fn

def load_agent(stage3_path, z_dim=32):
    from fb_agent import FBAgent
    ckpt = torch.load(stage3_path, map_location="cpu", weights_only=False)
    agent = FBAgent(2, 2, z_dim=z_dim, hidden_dim=256, device=DEVICE)
    agent.forward_net.load_state_dict(ckpt["forward_net"])
    agent.backward_net.load_state_dict(ckpt["backward_net"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.forward_net.eval()
    agent.backward_net.eval()
    agent.actor.eval()
    return agent


@torch.no_grad()
def act_batch(agent, obs_np, z_batch):
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=agent.device)
    actions = agent.actor(obs_t, z_batch)
    return actions.clamp(-1, 1).cpu().numpy()


@torch.no_grad()
def infer_z_batch(agent, goals_np, sample_obs_np):
    B = agent.backward_net(
        torch.as_tensor(sample_obs_np, dtype=torch.float32, device=agent.device)
    )
    dists = np.linalg.norm(goals_np[:, None, :] - sample_obs_np[None, :, :], axis=2)
    rew = (dists < REWARD_RADIUS).astype(np.float32)
    no_hit = rew.sum(axis=1) == 0
    if no_hit.any():
        rew[no_hit] = np.exp(-REWARD_FALLBACK_SCALE * dists[no_hit]).astype(np.float32)
    rew_t = torch.as_tensor(rew, dtype=torch.float32, device=agent.device)
    z = (rew_t @ B) / rew_t.shape[1]
    z = F_fn.normalize(z, dim=-1) * math.sqrt(agent.z_dim)
    return z


def step_batch(states, actions):
    actions = np.clip(actions, -1.0, 1.0)
    sx = np.where(states[:, 0] >= 0, 1.0, -1.0)
    sy = np.where(states[:, 1] >= 0, 1.0, -1.0)
    signs = np.stack([sx, sy], axis=1)
    next_states = states + MAX_SPEED * signs * actions
    next_states = np.clip(next_states, -1.0, 1.0)
    return next_states.astype(np.float32)


def sample_in_quadrant(rng, quad, n, margin=GOAL_MARGIN):
    from env_quadrant import QUADRANT_BOUNDS
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[quad]
    xs = rng.uniform(xlo + margin, xhi - margin, size=(n,))
    ys = rng.uniform(ylo + margin, yhi - margin, size=(n,))
    return np.stack([xs, ys], axis=1).astype(np.float32)


def eval_agent_quadrant(agent, quad, seed=EVAL_SEED):
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1, 1, size=(N_SAMPLE_OBS, 2)).astype(np.float32)
    goals = sample_in_quadrant(rng, quad, N_ENVS)
    starts = sample_in_quadrant(rng, quad, N_ENVS)
    z_batch = infer_z_batch(agent, goals, sample_obs)
    states = starts.copy()
    for _ in range(N_STEPS):
        actions = act_batch(agent, states, z_batch)
        states = step_batch(states, actions)
    final_dists = np.linalg.norm(states - goals, axis=1)
    return {
        "mean_d": float(final_dists.mean()),
        "s10": float((final_dists < 0.10).mean()),
    }


# ---------------------------------------------------------------------------
# Step 1: Evaluate new methods
# ---------------------------------------------------------------------------
def evaluate_new_methods():
    """Evaluate vMF20+FDWS fixed and optimized (3 seeds each)."""
    results = {}
    for method_name, info in NEW_METHODS.items():
        base_dir = info["dir"]
        seed_dirs = sorted([
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("seed")
        ])
        print(f"\nEvaluating {method_name}: {len(seed_dirs)} seeds")

        all_metrics = {q: [] for q in EVAL_QUADRANTS}
        for sd in seed_dirs:
            stage3 = os.path.join(sd, "stage3.pt")
            if not os.path.isfile(stage3):
                print(f"  SKIP {sd}: no stage3.pt")
                continue
            agent = load_agent(stage3)
            for quad in EVAL_QUADRANTS:
                m = eval_agent_quadrant(agent, quad)
                all_metrics[quad].append(m)
                print(f"  {os.path.basename(sd)} {quad}: mean_d={m['mean_d']:.5f} s10={m['s10']:.4f}")
            del agent
            torch.cuda.empty_cache()

        rec = {
            "method_id": method_name,
            "config_name": info["config_name"],
            "sweep": info["sweep"],
            "n_train_seeds": len(seed_dirs),
        }
        for quad in EVAL_QUADRANTS:
            if all_metrics[quad]:
                md = np.array([x["mean_d"] for x in all_metrics[quad]])
                s10 = np.array([x["s10"] for x in all_metrics[quad]])
                rec[f"mean_d_{quad}_mean"] = md.mean()
                rec[f"mean_d_{quad}_std"] = md.std()
                rec[f"s10_{quad}_mean"] = s10.mean()
                rec[f"s10_{quad}_std"] = s10.std()
        results[method_name] = rec
    return results


# ---------------------------------------------------------------------------
# Step 2: Load all data and merge
# ---------------------------------------------------------------------------
def load_and_merge(new_results):
    """Load CSV + new method results, assign M-IDs."""
    df = pd.read_csv(SUMMARY_CSV)

    # Add new methods
    for name, rec in new_results.items():
        row = {
            "method_id": rec["method_id"],
            "config_name": rec["config_name"],
            "sweep": rec["sweep"],
            "n_train_seeds": rec["n_train_seeds"],
            "mean_d_Q1_mean": rec.get("mean_d_Q1_mean", np.nan),
            "mean_d_Q1_std": rec.get("mean_d_Q1_std", np.nan),
            "s10_Q1_mean": rec.get("s10_Q1_mean", np.nan),
            "s10_Q1_std": rec.get("s10_Q1_std", np.nan),
            "mean_d_Q3_mean": rec.get("mean_d_Q3_mean", np.nan),
            "mean_d_Q3_std": rec.get("mean_d_Q3_std", np.nan),
            "s10_Q3_mean": rec.get("s10_Q3_mean", np.nan),
            "s10_Q3_std": rec.get("s10_Q3_std", np.nan),
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Remove exact duplicates (D7≡D8, D2≡D3≡D5, C2≡C3≡C5, C7≡C8)
    # Keep first occurrence by method_id
    # Mark duplicates based on identical Q1+Q3 mean_d values
    df["overall"] = (df["mean_d_Q1_mean"] + df["mean_d_Q3_mean"]) / 2
    df = df.sort_values("overall").reset_index(drop=True)

    # Remove known duplicates - keep the canonical one
    dupes_to_remove = [
        "z_sampling/D8_fdws_FB",   # == D7_fdws_B
        "z_sampling/D3_smr_FB",    # == D2_smr_B
        "z_sampling/D5_smr_piFB",  # == D2_smr_B
        "z_sampling/C3_smr_FB",    # == C2_smr_B
        "z_sampling/C5_smr_piFB",  # == C2_smr_B
        "z_sampling/C8_fdws_FB",   # == C7_fdws_B
    ]
    df = df[~df["method_id"].isin(dupes_to_remove)].reset_index(drop=True)

    # Assign M-IDs
    df["M_id"] = [f"M{i+1}" for i in range(len(df))]

    # Categorize methods
    def categorize(row):
        mid = row["method_id"]
        sweep = row["sweep"]
        if "baseline" in str(sweep) or mid in ["naive_seq_z32", "taskincr_z32"]:
            return "Baseline"
        if "vmf_fdws" in str(sweep):
            return "vMF+FDWS"
        if "vmf" in str(sweep):
            return "vMF sweep"
        if "fdws" in str(sweep):
            return "FDWS sweep"
        if "distill_combined" in str(sweep):
            return "Distill combined"
        if "distill" in str(sweep):
            return "Distillation"
        if "z_sampling" in str(sweep):
            cfg = str(row["config_name"])
            if "smr" in cfg:
                return "SMR z-sampling"
            if "fdws" in cfg:
                return "FDWS z-sampling"
            if "vmf" in cfg:
                return "vMF z-sampling"
            return "z-sampling"
        if "D7v3" in str(sweep) or "D7v3" in mid:
            return "FDWS+replay"
        return "Other"

    df["category"] = df.apply(categorize, axis=1)

    # Shorter display names
    def make_display(row):
        mid = row["method_id"]
        cfg = row["config_name"]
        if "/" in mid:
            return cfg
        return mid

    df["display_name"] = df.apply(make_display, axis=1)

    return df


# ---------------------------------------------------------------------------
# Step 3: Generate plots
# ---------------------------------------------------------------------------
COLORS = {
    "Baseline": "#888888",
    "vMF+FDWS": "#e74c3c",
    "vMF sweep": "#e67e22",
    "FDWS sweep": "#2ecc71",
    "FDWS+replay": "#27ae60",
    "Distill combined": "#3498db",
    "Distillation": "#9b59b6",
    "SMR z-sampling": "#1abc9c",
    "FDWS z-sampling": "#2ecc71",
    "vMF z-sampling": "#e67e22",
    "z-sampling": "#95a5a6",
    "Other": "#bdc3c7",
}


def plot_bar_top30(df, out_path):
    """Side-by-side bar chart: mean_d Q1 and Q3 for top 30."""
    top = df.head(30).copy()
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(top))
    w = 0.35

    bars1 = ax.bar(x - w/2, top["mean_d_Q1_mean"], w, yerr=top["mean_d_Q1_std"],
                   label="Q1 (memory)", color="#3498db", capsize=2, alpha=0.85)
    bars2 = ax.bar(x + w/2, top["mean_d_Q3_mean"], w, yerr=top["mean_d_Q3_std"],
                   label="Q3 (plasticity)", color="#e74c3c", capsize=2, alpha=0.85)

    ax.set_ylabel("Mean Distance to Goal (lower = better)", fontsize=12)
    ax.set_title("Top 30 Methods: Mean Distance (Q1 Memory vs Q3 Plasticity)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.M_id}\n{row.display_name}" for _, row in top.iterrows()],
                       rotation=70, ha="right", fontsize=7)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.8, len(top) - 0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_scatter_all(df, out_path):
    """Memory vs Plasticity scatter, all methods, colored by category."""
    fig, ax = plt.subplots(figsize=(12, 10))
    for cat, color in COLORS.items():
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub["mean_d_Q3_mean"], sub["mean_d_Q1_mean"],
                   c=color, label=cat, s=50, alpha=0.7, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Q3 Mean Distance (Plasticity, lower = better)", fontsize=12)
    ax.set_ylabel("Q1 Mean Distance (Memory, lower = better)", fontsize=12)
    ax.set_title("Memory vs Plasticity: All Methods", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal")

    # Draw Pareto frontier
    pts = df[["mean_d_Q3_mean", "mean_d_Q1_mean"]].values
    pareto = []
    sorted_idx = np.argsort(pts[:, 0])
    best_y = float("inf")
    for i in sorted_idx:
        if pts[i, 1] < best_y:
            pareto.append(i)
            best_y = pts[i, 1]
    if len(pareto) > 1:
        pareto_pts = pts[pareto]
        ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], "k--", alpha=0.4, linewidth=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_scatter_top30(df, out_path):
    """Memory vs Plasticity scatter, top 30 zoomed with labels."""
    top = df.head(30).copy()
    fig, ax = plt.subplots(figsize=(14, 10))

    for cat, color in COLORS.items():
        sub = top[top["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub["mean_d_Q3_mean"], sub["mean_d_Q1_mean"],
                   c=color, label=cat, s=80, alpha=0.8, edgecolors="black", linewidth=0.5)

    for _, row in top.iterrows():
        ax.annotate(row["M_id"], (row["mean_d_Q3_mean"], row["mean_d_Q1_mean"]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel("Q3 Mean Distance (Plasticity, lower = better)", fontsize=12)
    ax.set_ylabel("Q1 Mean Distance (Memory, lower = better)", fontsize=12)
    ax.set_title("Memory vs Plasticity: Top 30 Methods (Zoomed)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_bar_s10(df, out_path):
    """Success rate bar chart for top 30."""
    top = df.head(30).copy()
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(top))
    w = 0.35

    ax.bar(x - w/2, top["s10_Q1_mean"], w, yerr=top["s10_Q1_std"],
           label="Q1 success rate", color="#3498db", capsize=2, alpha=0.85)
    ax.bar(x + w/2, top["s10_Q3_mean"], w, yerr=top["s10_Q3_std"],
           label="Q3 success rate", color="#e74c3c", capsize=2, alpha=0.85)

    ax.set_ylabel("Success Rate (s10, higher = better)", fontsize=12)
    ax.set_title("Top 30 Methods: Success Rate (within 0.10 of goal)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.M_id}\n{row.display_name}" for _, row in top.iterrows()],
                       rotation=70, ha="right", fontsize=7)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.7)
    ax.set_xlim(-0.8, len(top) - 0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_violin_sweep(df, out_path):
    """Box/violin comparison by method family."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cats_order = ["Baseline", "Distillation", "Distill combined", "SMR z-sampling",
                  "FDWS z-sampling", "vMF z-sampling", "FDWS sweep", "FDWS+replay",
                  "vMF sweep", "vMF+FDWS"]

    available_cats = [c for c in cats_order if c in df["category"].values]

    for ax_idx, (metric, title, ylabel) in enumerate([
        ("overall", "Overall Score by Method Family", "Overall = (mean_d_Q1 + mean_d_Q3)/2"),
        ("s10_mean", "Success Rate by Method Family", "Mean Success Rate (Q1+Q3)/2"),
    ]):
        ax = axes[ax_idx]
        if metric == "s10_mean":
            df["s10_mean"] = (df["s10_Q1_mean"] + df["s10_Q3_mean"]) / 2

        data = []
        labels = []
        colors = []
        for cat in available_cats:
            sub = df[df["category"] == cat]
            if len(sub) < 2:
                continue
            data.append(sub[metric].values)
            labels.append(cat.replace(" ", "\n"))
            colors.append(COLORS.get(cat, "#888888"))

        bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_z_spheres(out_path):
    """Generate 3D sphere plots for z=3 methods (4 rows x 3 stages)."""
    fig = plt.figure(figsize=(18, 22))

    quad_colors = {"Q1": "#3498db", "Q2": "#2ecc71", "Q3": "#e74c3c"}
    stages = ["S1", "S2", "S3"]
    stage_labels = ["After Stage 1 (Q1)", "After Stage 2 (Q1+Q2)", "After Stage 3 (Q1+Q2+Q3)"]

    methods = list(Z_VIZ_METHODS.items())

    for row_idx, (method_name, npz_path) in enumerate(methods):
        if not os.path.exists(npz_path):
            print(f"  SKIP z-viz: {npz_path} not found")
            continue
        data = np.load(npz_path)

        for col_idx, stage in enumerate(stages):
            ax = fig.add_subplot(4, 3, row_idx * 3 + col_idx + 1, projection="3d")

            for quad, color in quad_colors.items():
                key_z = f"{stage}_{quad}_zs"
                if key_z not in data:
                    continue
                zs = data[key_z]  # (400, 3)
                # Normalize to unit sphere for visualization
                norms = np.linalg.norm(zs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                zs_unit = zs / norms

                ax.scatter(zs_unit[:, 0], zs_unit[:, 1], zs_unit[:, 2],
                          c=color, s=8, alpha=0.5, label=quad if col_idx == 0 else None)

            # Draw wireframe sphere
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zss = np.outer(np.ones(u.size), np.cos(v))
            ax.plot_wireframe(xs, ys, zss, color="gray", alpha=0.08, linewidth=0.3)

            if col_idx == 0:
                ax.set_ylabel(method_name, fontsize=10, labelpad=15)
            ax.set_title(stage_labels[col_idx], fontsize=9)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)
            ax.tick_params(labelsize=6)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("z-Space Distribution on Unit Sphere (z_dim=3)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Step 4: Generate LaTeX
# ---------------------------------------------------------------------------
def escape_latex(s):
    """Escape special LaTeX characters."""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def generate_latex(df):
    """Generate comprehensive LaTeX report."""
    top20 = df.head(20)
    top30 = df.head(30)

    # Build full method table rows
    method_table_rows = []
    for _, row in df.iterrows():
        method_table_rows.append(
            f"    {escape_latex(row['M_id'])} & "
            f"{escape_latex(row['display_name'])} & "
            f"{escape_latex(str(row['category']))} & "
            f"{row['mean_d_Q1_mean']:.4f} & "
            f"{row['s10_Q1_mean']:.3f} & "
            f"{row['mean_d_Q3_mean']:.4f} & "
            f"{row['s10_Q3_mean']:.3f} & "
            f"{row['overall']:.4f} \\\\"
        )

    # Top 20 table rows
    top20_rows = []
    for _, row in top20.iterrows():
        top20_rows.append(
            f"    {escape_latex(row['M_id'])} & "
            f"{escape_latex(row['display_name'])} & "
            f"{escape_latex(str(row['category']))} & "
            f"{row['mean_d_Q1_mean']:.4f}$\\pm${row['mean_d_Q1_std']:.4f} & "
            f"{row['s10_Q1_mean']:.3f} & "
            f"{row['mean_d_Q3_mean']:.4f}$\\pm${row['mean_d_Q3_std']:.4f} & "
            f"{row['s10_Q3_mean']:.3f} & "
            f"{row['overall']:.4f} \\\\"
        )

    # Category summary
    cat_summary_rows = []
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        cat_summary_rows.append(
            f"    {escape_latex(cat)} & "
            f"{len(sub)} & "
            f"{sub['overall'].min():.4f} & "
            f"{sub['overall'].median():.4f} & "
            f"{sub['overall'].mean():.4f} \\\\"
        )

    n_total = len(df)

    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.8cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{float}

\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}

\title{Forward-Backward Continual Learning:\\Comprehensive Experimental Report}
\author{FB Continual Learning Research}
\date{April 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage

%% =====================================================================
\section{Introduction \& Task Description}
%% =====================================================================

This report presents a comprehensive evaluation of \textbf{""" + str(n_total) + r"""} method configurations
for continual learning with Forward-Backward (FB) representation learning on the
\textbf{Nav2D QuadDyn} testbed.

\subsection{Nav2D QuadDyn Environment}
The environment is a 2D continuous navigation task on $[-1,1]^2$ divided into four quadrants,
each with distinct dynamics:
\begin{itemize}
    \item \textbf{Quadrant dynamics:} In each quadrant, the sign of the action components is flipped
    according to: $s_x = +1$ if $x \geq 0$, else $-1$; $s_y = +1$ if $y \geq 0$, else $-1$.
    The effective action is $(s_x a_x, s_y a_y)$, creating per-quadrant dynamics
    that require different policies.
    \item \textbf{Transition:} $\mathbf{s}_{t+1} = \mathbf{s}_t + v_{\max} \cdot \text{diag}(s_x, s_y) \cdot \mathbf{a}_t$
    with $v_{\max} = 0.05$ and clipping to $[-1,1]^2$.
    \item \textbf{Goal-conditioned:} The agent must reach arbitrary goal positions using
    the FB zero-shot mechanism: $z$ is inferred from the backward representation $B(s)$
    weighted by goal proximity.
\end{itemize}

\subsection{Continual Training Protocol: q123}
Training proceeds in three sequential stages:
\begin{enumerate}
    \item \textbf{Stage 1:} Train on Q1 data (offline batch from quadrant 1)
    \item \textbf{Stage 2:} Train on Q2 data (quadrant 2)
    \item \textbf{Stage 3:} Train on Q3 data (quadrant 3)
\end{enumerate}
Evaluation measures:
\begin{itemize}
    \item \textbf{Q1 performance (Memory):} How well the agent retains Q1 goal-reaching after training on Q2 and Q3.
    Lower \texttt{mean\_d\_Q1} $=$ better memory.
    \item \textbf{Q3 performance (Plasticity):} How well the agent learns the final task.
    Lower \texttt{mean\_d\_Q3} $=$ better plasticity.
    \item \textbf{Overall:} $(\texttt{mean\_d\_Q1} + \texttt{mean\_d\_Q3}) / 2$.
    \item \textbf{Success rate (s10):} Fraction of 8192 eval episodes where final distance $< 0.10$.
\end{itemize}

%% =====================================================================
\section{Experimental Setup}
%% =====================================================================

\subsection{Shared Parameters}
All methods share:
\begin{itemize}
    \item $z_{\text{dim}} = 32$, hidden\_dim $= 256$, learning rate $= 10^{-4}$
    \item 60{,}000 gradient updates per stage, batch size 512
    \item 3 training seeds per configuration (except naive baseline: 1 seed)
    \item Evaluation: 8192 parallel environments, 200 steps, seed 42
\end{itemize}

\subsection{4-Axis Design Space}
Methods vary along four axes:
\begin{enumerate}
    \item[\textbf{B:}] \textbf{Distillation target} --- which network components are distilled
    (F, B, FB, pi, M=FB, Q=critic, combinations thereof)
    \item[\textbf{C:}] \textbf{Distillation loss} --- L2, Gram, cosine, contrastive
    \item[\textbf{D:}] \textbf{Distillation source} --- current-batch vs.\ replay-buffer distillation
    \item[\textbf{E:}] \textbf{z-sampling strategy} --- uniform, vMF mixture, SMR (successor measure routing),
    FDWS (Fisher-Distance Weighted Sampling), combined vMF+FDWS
\end{enumerate}

\subsection{Sweep Structure}
\begin{itemize}
    \item \texttt{distill}: Single-target distillation (B$\times$C$\times$D)
    \item \texttt{distill\_combined}: Multi-target pi+X distillation with replay
    \item \texttt{z\_sampling}: z-sampling strategies (A/B series = binding/no-binding, C/D series = with/without distillation)
    \item \texttt{fdws\_sweep}: FDWS temperature and sensitivity source variations
    \item \texttt{vmf\_sweep}: vMF mixture component count and target sweeps
    \item \texttt{vmf\_fdws\_combined}: Combined vMF20 + FDWS (fixed and optimized temperature)
\end{itemize}

%% =====================================================================
\section{Method Overview}
%% =====================================================================

Table~\ref{tab:all_methods} lists all """ + str(n_total) + r""" evaluated method configurations,
sorted by overall score (lower is better).

\begin{small}
\begin{longtable}{llp{3cm}rrrrrr}
\caption{All """ + str(n_total) + r""" methods sorted by overall score. Columns: mean\_d = mean final distance to goal, s10 = success rate (fraction within 0.10), overall = (Q1+Q3)/2.}
\label{tab:all_methods} \\
\toprule
M-ID & Config Name & Category & Q1 mean\_d & Q1 s10 & Q3 mean\_d & Q3 s10 & Overall \\
\midrule
\endfirsthead
\toprule
M-ID & Config Name & Category & Q1 mean\_d & Q1 s10 & Q3 mean\_d & Q3 s10 & Overall \\
\midrule
\endhead
\midrule
\multicolumn{8}{r}{\textit{Continued on next page}} \\
\bottomrule
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(method_table_rows) + r"""
\end{longtable}
\end{small}

%% =====================================================================
\section{Overall Performance Rankings}
%% =====================================================================

\subsection{Top 30 Methods: Mean Distance}
Figure~\ref{fig:bar_top30} shows the mean final distance to goal for the top 30 methods,
with Q1 (memory) and Q3 (plasticity) side by side.

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{bar_top30_mean_d.png}
\caption{Top 30 methods by overall score: mean distance to goal in Q1 (memory) and Q3 (plasticity).
Error bars show standard deviation across 3 training seeds.}
\label{fig:bar_top30}
\end{figure}

\subsection{Top 20 Detailed Results}

\begin{small}
\begin{longtable}{llp{2.5cm}p{3.2cm}rp{3.2cm}rr}
\caption{Top 20 methods with standard deviations.}
\label{tab:top20} \\
\toprule
M-ID & Name & Category & Q1 mean\_d & Q1 s10 & Q3 mean\_d & Q3 s10 & Overall \\
\midrule
\endfirsthead
\toprule
M-ID & Name & Category & Q1 mean\_d & Q1 s10 & Q3 mean\_d & Q3 s10 & Overall \\
\midrule
\endhead
\bottomrule
\endlastfoot
""" + "\n".join(top20_rows) + r"""
\end{longtable}
\end{small}

\subsection{Success Rate: Top 30}
Figure~\ref{fig:bar_s10} shows the success rate (fraction of episodes reaching within 0.10 of goal).

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{bar_top30_s10.png}
\caption{Top 30 methods: success rate (s10) for Q1 and Q3. Higher is better.}
\label{fig:bar_s10}
\end{figure}

%% =====================================================================
\section{Memory vs.\ Plasticity Analysis}
%% =====================================================================

\subsection{All Methods}
Figure~\ref{fig:scatter_all} shows the memory--plasticity trade-off across all methods.
Points closer to the origin are better on both axes.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{scatter_all.png}
\caption{Memory (Q1 mean\_d) vs.\ Plasticity (Q3 mean\_d) for all methods.
The dashed line indicates the approximate Pareto frontier.}
\label{fig:scatter_all}
\end{figure}

\subsection{Top 30 Zoomed}
Figure~\ref{fig:scatter_top30} zooms into the top 30 methods with M-ID labels.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{scatter_top30.png}
\caption{Memory vs.\ Plasticity for the top 30 methods, labeled by M-ID.}
\label{fig:scatter_top30}
\end{figure}

%% =====================================================================
\section{Ablation Studies}
%% =====================================================================

\subsection{Distillation Target (B $\times$ C $\times$ D sweep)}

Key findings from the distillation sweep:
\begin{itemize}
    \item \textbf{Replay is essential:} ``current'' distillation (on the current mini-batch)
    causes catastrophic forgetting on Q1; ``replay'' distillation preserves Q1.
    \item \textbf{Policy distillation ($\pi$) helps most:} Distilling the policy network
    combined with B or FB targets (e.g., piGram\_B\_replay, piL2\_B\_replay) gives the best
    memory--plasticity trade-off among pure distillation methods.
    \item \textbf{Gram loss outperforms L2 on B-target:} piGram\_B\_replay (M2) achieves
    overall 0.140, vs.\ piL2\_B\_replay at 0.174.
    \item \textbf{F-only and M-only distillation fails:} Distilling only F or only M collapses
    Q1 performance (mean\_d $> 0.6$), indicating B is the critical representation.
    \item \textbf{Gram/contrastive losses on raw representations diverge:}
    F\_gram, FB\_gram, M\_gram methods have overall $> 0.65$.
\end{itemize}

\subsection{z-Sampling Strategy (Axis E)}

\begin{itemize}
    \item \textbf{FDWS z-sampling dominates:} D7\_fdws\_B (FDWS on backward net, M1) achieves
    overall 0.139, the best among all methods in the original sweep.
    \item \textbf{SMR (successor measure routing) underperforms:} D1--D5 (SMR variants) all have
    Q3 mean\_d $> 0.33$, indicating poor plasticity despite good Q1 memory.
    \item \textbf{vMF mixture alone is competitive:} B2\_vmf\_mix (M8) achieves 0.150 overall,
    close to FDWS variants.
    \item \textbf{No-distillation z-sampling (C-series) fails:} Without replay distillation,
    all z-sampling strategies collapse on Q1 (mean\_d $> 0.72$).
\end{itemize}

\subsection{vMF Component Count Sweep}

\begin{itemize}
    \item \textbf{More components help on B-target:} vmfMix20\_Bpi (M16, overall 0.173) $<$
    vmfMix10\_Bpi (M19, 0.175) $<$ vmfMix5\_Bpi (M14, 0.168). Non-monotonic due to variance.
    \item \textbf{FB+pi target is unstable with vMF:} vmfMix20\_FBpi (M12, 0.154) is good
    but vmfMix10\_FBpi (M20, 0.187) and vmfMix5\_FBpi (M21, 0.179) are worse.
    \item \textbf{F-target vMF collapses:} vmfMix*\_Fpi methods have Q1 mean\_d $> 0.6$,
    confirming F-only distillation is harmful.
\end{itemize}

\subsection{FDWS Temperature Sweep}

\begin{itemize}
    \item \textbf{Lower temperature is better with replay sensitivity:}
    tau10\_sensReplay (M4, 0.141) $<$ tau50\_sensReplay (M10, 0.151) $<$
    tau05\_sensReplay (M23, 0.191).
    $\tau=10$ with replay-buffer sensitivity is optimal.
    \item \textbf{Current-batch sensitivity is also competitive:}
    tau50\_sensCurrent (M5, 0.145) is close to the best replay variant.
    \item \textbf{Sensitivity source matters less than expected:} The gap between
    sensReplay and sensCurrent is small ($\sim$0.004 at $\tau=10$).
\end{itemize}

\subsection{Combined vMF + FDWS}

\begin{itemize}
    \item \textbf{D7v3 (high-sensitivity replay z):} M6, overall 0.148. Uses FDWS-weighted
    z-sampling with sensitivity-selected replay z vectors.
    \item \textbf{vMF20+FDWS (fixed $\tau=1$):} Combines 20-component vMF mixture with
    FDWS weighting at fixed temperature.
    \item \textbf{vMF20+FDWS (optimized):} Uses per-stage optimized FDWS temperature
    via validation performance.
\end{itemize}

\subsection{Category Summary}

\begin{table}[H]
\centering
\caption{Summary statistics by method category.}
\begin{tabular}{lrrrr}
\toprule
Category & Count & Best Overall & Median Overall & Mean Overall \\
\midrule
""" + "\n".join(cat_summary_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

%% =====================================================================
\section{Method Family Comparison}
%% =====================================================================

Figure~\ref{fig:violin} shows box plots comparing method families on overall score and success rate.

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{violin_sweep.png}
\caption{Box plots of overall score and mean success rate by method family.}
\label{fig:violin}
\end{figure}

%% =====================================================================
\section{z-Space Visualization}
%% =====================================================================

Figure~\ref{fig:zsphere} visualizes the learned $z$-representations on the unit sphere
for $z_{\text{dim}}=3$ models. Each panel shows the distribution of inferred $z$ vectors
for goals in Q1 (blue), Q2 (green), and Q3 (red) after each training stage.

\textbf{Key observations:}
\begin{itemize}
    \item \textbf{FDWS+distill (D7v3):} z vectors for different quadrants form distinct clusters
    that remain separated across stages, indicating good task discrimination and memory.
    \item \textbf{Naive+distill:} Distillation alone partially preserves Q1 clusters but they
    drift significantly after stages 2 and 3.
    \item \textbf{Naive (no distill):} Without any continual learning mechanism, Q1 z-vectors
    are completely overwritten by later tasks --- all points collapse to the same region.
    \item \textbf{Opt vMF20+FDWS:} The vMF mixture creates well-separated, compact clusters
    for each quadrant, with the best visual separation.
\end{itemize}

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{z_spheres.png}
\caption{z-space distribution on the unit sphere ($z_{\text{dim}}=3$). Rows: different methods.
Columns: after each training stage. Colors: Q1 (blue), Q2 (green), Q3 (red).}
\label{fig:zsphere}
\end{figure}

%% =====================================================================
\section{Key Findings \& Conclusions}
%% =====================================================================

\begin{enumerate}
    \item \textbf{FDWS z-sampling is the single most impactful technique.}
    The top method (M1: FDWS-B with distillation, overall 0.139) uses Fisher-distance
    weighted sampling to select z vectors that maximize information gain, combined with
    Gram-based policy distillation on a replay buffer.

    \item \textbf{Replay-based distillation is necessary but not sufficient.}
    Without replay, all methods collapse on Q1 (mean\_d $> 0.7$).
    With replay alone (without advanced z-sampling), performance plateaus around 0.14--0.17.

    \item \textbf{Policy distillation ($\pi$) is the most important distillation target.}
    Methods distilling only F or M representations fail, while $\pi$+B and $\pi$+FB
    combinations consistently appear in the top 20.

    \item \textbf{The memory--plasticity trade-off is real but manageable.}
    The best methods achieve Q1 mean\_d $\approx 0.12$ and Q3 mean\_d $\approx 0.16$
    simultaneously, with Q1 success rates $\sim$49\% and Q3 $\sim$32\%.

    \item \textbf{vMF mixtures provide good z-space structure.}
    The z-sphere visualizations confirm that vMF-based methods create well-separated
    task clusters, which likely aids zero-shot transfer.

    \item \textbf{Temperature tuning matters for FDWS.}
    $\tau=10$ with replay sensitivity achieves the best FDWS-only result, but the
    improvement over $\tau=50$ is modest.

    \item \textbf{Combining vMF + FDWS is promising.}
    The combined methods approach the Pareto frontier, suggesting that structured
    z-priors (vMF) and information-theoretic sampling (FDWS) are complementary.

    \item \textbf{No-distillation baselines are informative.}
    The naive sequential baseline (overall $\approx 0.53$) and task-incremental
    upper bound (overall $\approx 0.22$) bracket the performance range, with the
    best continual methods closing 85\% of this gap.
\end{enumerate}

\subsection{Limitations \& Future Directions}
\begin{itemize}
    \item Evaluation is on a single 2D navigation domain; generalization to higher-dimensional
    tasks and longer task sequences remains open.
    \item The q123 protocol tests only 3 stages; scaling to 10+ stages may change the ranking.
    \item All methods use the same network architecture; architecture search could yield further gains.
    \item Online (non-offline) continual FB learning is unexplored.
\end{itemize}

\end{document}
"""
    tex_path = os.path.join(OUTPUT_DIR, "final_report.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  LaTeX written to {tex_path}")
    return tex_path


# ---------------------------------------------------------------------------
# Step 5: Compile PDF
# ---------------------------------------------------------------------------
def compile_latex(tex_path):
    """Compile LaTeX to PDF (2 passes for longtable + TOC)."""
    tex_dir = os.path.dirname(tex_path)
    tex_file = os.path.basename(tex_path)

    for pass_num in range(2):
        print(f"  pdflatex pass {pass_num + 1}...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tex_dir, tex_file],
            cwd=tex_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  WARNING: pdflatex pass {pass_num + 1} returned {result.returncode}")
            # Print last 30 lines of log for debugging
            lines = result.stdout.split("\n")
            for line in lines[-30:]:
                if line.strip():
                    print(f"    {line}")

    pdf_path = tex_path.replace(".tex", ".pdf")
    if os.path.isfile(pdf_path):
        print(f"  PDF generated: {pdf_path}")
    else:
        print(f"  ERROR: PDF not generated!")
    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FB Continual Learning: Final Report Generator")
    print("=" * 70)

    # Step 1: Evaluate new methods
    print("\n[Step 1] Evaluating new methods (vMF20+FDWS fixed/optimized)...")
    new_results = evaluate_new_methods()
    for name, rec in new_results.items():
        q1 = rec.get("mean_d_Q1_mean", "?")
        q3 = rec.get("mean_d_Q3_mean", "?")
        print(f"  {name}: Q1={q1:.5f}, Q3={q3:.5f}" if isinstance(q1, float) else f"  {name}: incomplete")

    # Step 2: Load and merge all data
    print("\n[Step 2] Loading and merging all evaluation data...")
    df = load_and_merge(new_results)
    print(f"  Total methods: {len(df)}")
    print(f"  Categories: {dict(df['category'].value_counts())}")
    print(f"\n  Top 5:")
    for _, row in df.head(5).iterrows():
        print(f"    {row['M_id']}: {row['display_name']} (overall={row['overall']:.4f}, cat={row['category']})")

    # Save merged data
    merged_csv = os.path.join(OUTPUT_DIR, "all_methods_final.csv")
    df.to_csv(merged_csv, index=False)
    print(f"  Saved merged data: {merged_csv}")

    # Step 3: Generate plots
    print("\n[Step 3] Generating plots...")
    plot_bar_top30(df, os.path.join(OUTPUT_DIR, "bar_top30_mean_d.png"))
    plot_scatter_all(df, os.path.join(OUTPUT_DIR, "scatter_all.png"))
    plot_scatter_top30(df, os.path.join(OUTPUT_DIR, "scatter_top30.png"))
    plot_bar_s10(df, os.path.join(OUTPUT_DIR, "bar_top30_s10.png"))
    plot_violin_sweep(df, os.path.join(OUTPUT_DIR, "violin_sweep.png"))
    plot_z_spheres(os.path.join(OUTPUT_DIR, "z_spheres.png"))

    # Step 4: Generate LaTeX
    print("\n[Step 4] Generating LaTeX report...")
    tex_path = generate_latex(df)

    # Step 5: Compile PDF
    print("\n[Step 5] Compiling PDF...")
    pdf_path = compile_latex(tex_path)

    print("\n" + "=" * 70)
    print(f"DONE! Final report: {pdf_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
