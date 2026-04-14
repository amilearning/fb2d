"""
eval_all_methods_fast.py — Fast vectorized evaluation of all 113 FB continual-
learning methods using 4096 parallel environments on GPU.

Usage:
    /home/frl/anaconda3/envs/incr_learn/bin/python eval_all_methods_fast.py
"""

import csv
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F_fn

sys.path.insert(0, "/home/frl/FB2D")
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEED = 42
EVAL_QUADRANTS = ["Q1", "Q3"]
N_ENVS = 8192
N_STEPS = 200
MAX_SPEED = 0.05
GOAL_MARGIN = 0.10  # margin from quadrant edges for goal/start sampling
REWARD_RADIUS = 0.07
REWARD_FALLBACK_SCALE = 25.0
N_SAMPLE_OBS = 4096  # reference set size for z inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SWEEP_DIRS = {
    "distill":          "/home/frl/FB2D/checkpoints_distill_real",
    "distill_combined": "/home/frl/FB2D/checkpoints_distill_combined_real",
    "z_sampling":       "/home/frl/FB2D/checkpoints_z_sampling_real",
    "vmf_sweep":        "/home/frl/FB2D/checkpoints_vmf_sweep_real",
    "fdws_sweep":       "/home/frl/FB2D/checkpoints_fdws_sweep_real",
}

BASELINE_DIRS = {
    "naive_seq_z32": "/home/frl/FB2D/checkpoints_naive_seq",
    "taskincr_z32":  "/home/frl/FB2D/checkpoints_taskincr",
}

RESULTS_DIR = "/home/frl/FB2D/eval_results"
RAW_CSV = os.path.join(RESULTS_DIR, "eval_fast.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "eval_fast_summary.csv")


# ---------------------------------------------------------------------------
# Batched agent helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def act_batch(agent, obs_np, z_batch):
    """obs_np: (N, 2) numpy, z_batch: (N, z_dim) tensor -> actions (N, 2) numpy"""
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=agent.device)
    actions = agent.actor(obs_t, z_batch)
    return actions.clamp(-1, 1).cpu().numpy()


@torch.no_grad()
def infer_z_batch(agent, goals_np, sample_obs_np):
    """Infer z for each goal using reward-weighted B.
    goals_np: (N_goals, 2) numpy
    sample_obs_np: (M, 2) numpy  -- shared reference set
    Returns: z tensor (N_goals, z_dim) on agent.device
    """
    # Compute B for all sample_obs once: (M, z_dim)
    B = agent.backward_net(
        torch.as_tensor(sample_obs_np, dtype=torch.float32, device=agent.device)
    )  # (M, z_dim)

    # Compute distances: (N_goals, M)
    # goals_np: (N, 2), sample_obs_np: (M, 2)
    # Use numpy for distance computation (memory efficient)
    dists = np.linalg.norm(
        goals_np[:, None, :] - sample_obs_np[None, :, :], axis=2
    )  # (N_goals, M)

    # Compute rewards
    rew = (dists < REWARD_RADIUS).astype(np.float32)  # (N_goals, M)

    # Fallback for goals with no nearby sample points
    no_hit = rew.sum(axis=1) == 0  # (N_goals,)
    if no_hit.any():
        rew[no_hit] = np.exp(-REWARD_FALLBACK_SCALE * dists[no_hit]).astype(np.float32)

    # Weighted mean: z_i = mean_j(r_ij * B_j), then normalize
    rew_t = torch.as_tensor(rew, dtype=torch.float32, device=agent.device)  # (N_goals, M)
    # (N_goals, M) @ (M, z_dim) -> (N_goals, z_dim)
    z = (rew_t @ B) / rew_t.shape[1]
    z = F_fn.normalize(z, dim=-1) * math.sqrt(agent.z_dim)
    return z


# ---------------------------------------------------------------------------
# Vectorized env step
# ---------------------------------------------------------------------------
def step_batch(states, actions):
    """Vectorized Nav2DQuadDyn step.
    states: (N, 2) numpy, actions: (N, 2) numpy -> next_states (N, 2) numpy

    Dynamics: next = state + max_speed * sign_flip(state) * action
    Sign convention:
      sx = +1 if x >= 0 (Q1,Q4), -1 if x < 0 (Q2,Q3)
      sy = +1 if y >= 0 (Q1,Q2), -1 if y < 0 (Q3,Q4)
      effective_action = (sx * ax, sy * ay)
    """
    actions = np.clip(actions, -1.0, 1.0)
    sx = np.where(states[:, 0] >= 0, 1.0, -1.0)  # (N,)
    sy = np.where(states[:, 1] >= 0, 1.0, -1.0)  # (N,)
    signs = np.stack([sx, sy], axis=1)  # (N, 2)
    next_states = states + MAX_SPEED * signs * actions
    next_states = np.clip(next_states, -1.0, 1.0)
    return next_states.astype(np.float32)


# ---------------------------------------------------------------------------
# Sample goals and starts in a quadrant
# ---------------------------------------------------------------------------
def sample_in_quadrant(rng, quad, n, margin=GOAL_MARGIN):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[quad]
    xs = rng.uniform(xlo + margin, xhi - margin, size=(n,))
    ys = rng.uniform(ylo + margin, yhi - margin, size=(n,))
    return np.stack([xs, ys], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Full vectorized evaluation for one agent in one quadrant
# ---------------------------------------------------------------------------
def eval_agent_quadrant(agent, quad, seed=EVAL_SEED):
    """Run 4096 parallel (goal, start) pairs for N_STEPS.
    Returns dict with mean_d, s10, median_d, pct90_d.
    """
    rng = np.random.RandomState(seed)

    # Shared reference set for z inference
    sample_obs = rng.uniform(-1, 1, size=(N_SAMPLE_OBS, 2)).astype(np.float32)

    # Sample goals and starts
    goals = sample_in_quadrant(rng, quad, N_ENVS)   # (4096, 2)
    starts = sample_in_quadrant(rng, quad, N_ENVS)   # (4096, 2)

    # Infer z for all goals in one batch
    z_batch = infer_z_batch(agent, goals, sample_obs)  # (4096, z_dim) on device

    # Run vectorized rollout
    states = starts.copy()  # (4096, 2)
    for _ in range(N_STEPS):
        actions = act_batch(agent, states, z_batch)  # (4096, 2) numpy
        states = step_batch(states, actions)

    # Compute final distances
    final_dists = np.linalg.norm(states - goals, axis=1)  # (4096,)

    return {
        "mean_d": float(final_dists.mean()),
        "s10": float((final_dists < 0.10).mean()),
        "median_d": float(np.median(final_dists)),
        "pct90_d": float(np.percentile(final_dists, 90)),
    }


# ---------------------------------------------------------------------------
# Discovery (same logic as eval_all_methods.py)
# ---------------------------------------------------------------------------
def discover_methods():
    methods = []

    for sweep_name, sweep_path in SWEEP_DIRS.items():
        if not os.path.isdir(sweep_path):
            print(f"[WARN] sweep dir missing: {sweep_path}")
            continue
        for config_name in sorted(os.listdir(sweep_path)):
            config_path = os.path.join(sweep_path, config_name)
            if not os.path.isdir(config_path):
                continue
            method_id = f"{sweep_name}/{config_name}"
            for seed_dir in sorted(os.listdir(config_path)):
                sd = os.path.join(config_path, seed_dir)
                stage3 = os.path.join(sd, "stage3.pt")
                if os.path.isdir(sd) and os.path.isfile(stage3):
                    methods.append({
                        "method_id": method_id,
                        "config_name": config_name,
                        "sweep": sweep_name,
                        "seed_dir": sd,
                        "stage3_path": stage3,
                        "z_dim": 32,
                    })

    for base_name, base_path in BASELINE_DIRS.items():
        if not os.path.isdir(base_path):
            print(f"[WARN] baseline dir missing: {base_path}")
            continue
        method_id = base_name
        seed_runs = {}
        for entry in sorted(os.listdir(base_path)):
            sd = os.path.join(base_path, entry)
            stage3 = os.path.join(sd, "stage3.pt")
            if not os.path.isdir(sd) or not os.path.isfile(stage3):
                continue
            parts = entry.split("_")
            if not parts[0].startswith("seed"):
                continue
            seed_num = parts[0]
            ts = "_".join(parts[1:])
            if seed_num not in seed_runs or ts > seed_runs[seed_num][0]:
                seed_runs[seed_num] = (ts, sd, stage3)

        for seed_num, (ts, sd, stage3) in sorted(seed_runs.items()):
            try:
                ckpt = torch.load(stage3, map_location="cpu", weights_only=False)
                args = ckpt["args"]
                z_dim = args["z_dim"] if isinstance(args, dict) else args.z_dim
                if z_dim != 32:
                    print(f"[SKIP] {sd}: z_dim={z_dim} (need 32)")
                    continue
            except Exception as e:
                print(f"[WARN] cannot read {stage3}: {e}")
                continue
            methods.append({
                "method_id": method_id,
                "config_name": base_name,
                "sweep": "baseline",
                "seed_dir": sd,
                "stage3_path": stage3,
                "z_dim": 32,
            })

    return methods


# ---------------------------------------------------------------------------
# Load agent on GPU
# ---------------------------------------------------------------------------
def load_agent(stage3_path, z_dim=32):
    ckpt = torch.load(stage3_path, map_location="cpu", weights_only=False)
    agent = FBAgent(2, 2, z_dim=z_dim, hidden_dim=256, device=DEVICE)
    agent.forward_net.load_state_dict(ckpt["forward_net"])
    agent.backward_net.load_state_dict(ckpt["backward_net"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.forward_net.eval()
    agent.backward_net.eval()
    agent.actor.eval()
    return agent


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
RAW_FIELDS = [
    "method_id", "config_name", "sweep", "train_seed_dir",
    "quadrant", "n_envs", "mean_d", "s10", "median_d", "pct90_d",
]


def append_row(filepath, row, write_header=False):
    mode = "w" if write_header else "a"
    with open(filepath, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def build_summary(raw_csv_path):
    """Read raw CSV and aggregate per method_id."""
    rows = []
    with open(raw_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method_id"]].append(r)

    summary = []
    for mid, mrows in sorted(by_method.items()):
        config_name = mrows[0]["config_name"]
        sweep = mrows[0]["sweep"]
        n_seeds = len(set(r["train_seed_dir"] for r in mrows))

        rec = {
            "method_id": mid,
            "config_name": config_name,
            "sweep": sweep,
            "n_train_seeds": n_seeds,
        }
        for quad in EVAL_QUADRANTS:
            qrows = [r for r in mrows if r["quadrant"] == quad]
            if not qrows:
                continue
            md = np.array([float(r["mean_d"]) for r in qrows])
            s10 = np.array([float(r["s10"]) for r in qrows])
            rec[f"mean_d_{quad}_mean"] = f"{md.mean():.5f}"
            rec[f"mean_d_{quad}_std"] = f"{md.std():.5f}"
            rec[f"s10_{quad}_mean"] = f"{s10.mean():.4f}"
            rec[f"s10_{quad}_std"] = f"{s10.std():.4f}"

        summary.append(rec)

    # Sort by combined mean_d
    def sort_key(r):
        try:
            return float(r.get("mean_d_Q1_mean", 99)) + float(r.get("mean_d_Q3_mean", 99))
        except (ValueError, TypeError):
            return 999
    summary.sort(key=sort_key)

    summary_fields = [
        "method_id", "config_name", "sweep", "n_train_seeds",
        "mean_d_Q1_mean", "mean_d_Q1_std", "s10_Q1_mean", "s10_Q1_std",
        "mean_d_Q3_mean", "mean_d_Q3_std", "s10_Q3_mean", "s10_Q3_std",
    ]
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in summary_fields})

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Discovering methods ...")
    methods = discover_methods()
    n_configs = len(set(m["method_id"] for m in methods))
    print(f"Found {len(methods)} seed-dirs across {n_configs} method configs.\n")

    from collections import Counter
    sweep_counts = Counter(m["sweep"] for m in methods)
    for sw, cnt in sorted(sweep_counts.items()):
        nc = len(set(m["method_id"] for m in methods if m["sweep"] == sw))
        print(f"  {sw}: {nc} configs, {cnt} seed-dirs")

    # Check which methods already have results (for resume)
    done_keys = set()
    if os.path.isfile(RAW_CSV):
        with open(RAW_CSV, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["method_id"], r["train_seed_dir"], r["quadrant"])
                done_keys.add(key)
        print(f"\nResuming: {len(done_keys)} rows already in {RAW_CSV}")
        write_header = False
    else:
        write_header = True

    if write_header:
        # Write header
        with open(RAW_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            w.writeheader()

    total = len(methods)
    t0 = time.time()

    for idx, m in enumerate(methods):
        method_id = m["method_id"]
        seed_dir = m["seed_dir"]
        train_seed_label = os.path.basename(seed_dir)

        # Check if all quadrants done for this seed_dir
        all_done = all(
            (method_id, train_seed_label, q) in done_keys
            for q in EVAL_QUADRANTS
        )
        if all_done:
            print(f"[{idx+1}/{total}] SKIP (already done): {method_id} / {train_seed_label}")
            continue

        # Load agent
        try:
            agent = load_agent(m["stage3_path"], z_dim=m["z_dim"])
        except Exception as e:
            print(f"[{idx+1}/{total}] ERROR loading {m['stage3_path']}: {e}")
            continue

        for quad in EVAL_QUADRANTS:
            key = (method_id, train_seed_label, quad)
            if key in done_keys:
                continue

            t1 = time.time()
            try:
                metrics = eval_agent_quadrant(agent, quad)
            except Exception as e:
                print(f"  ERROR eval {quad}: {e}")
                continue
            dt = time.time() - t1

            row = {
                "method_id": method_id,
                "config_name": m["config_name"],
                "sweep": m["sweep"],
                "train_seed_dir": train_seed_label,
                "quadrant": quad,
                "n_envs": N_ENVS,
                "mean_d": f"{metrics['mean_d']:.5f}",
                "s10": f"{metrics['s10']:.4f}",
                "median_d": f"{metrics['median_d']:.5f}",
                "pct90_d": f"{metrics['pct90_d']:.5f}",
            }
            append_row(RAW_CSV, row)
            done_keys.add(key)

        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
        print(f"[{idx+1}/{total}] {method_id} / {train_seed_label} "
              f"| Q1 mean_d={metrics.get('mean_d', '?')} "
              f"| {elapsed:.0f}s elapsed, ~{rate:.1f} methods/min")

        # Free GPU memory
        del agent
        torch.cuda.empty_cache()

    total_time = time.time() - t0
    print(f"\nAll done in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Build summary
    print("Building summary ...")
    summary = build_summary(RAW_CSV)
    print(f"Summary saved to {SUMMARY_CSV} ({len(summary)} methods)")

    # Print top-20
    print("\n" + "=" * 120)
    print(f"{'Rank':>4}  {'Method':50s}  {'mean_d_Q1':>10} {'s10_Q1':>8}"
          f"  {'mean_d_Q3':>10} {'s10_Q3':>8}")
    print("-" * 120)
    for i, r in enumerate(summary[:20]):
        q1 = r.get("mean_d_Q1_mean", "?")
        s1 = r.get("s10_Q1_mean", "?")
        q3 = r.get("mean_d_Q3_mean", "?")
        s3 = r.get("s10_Q3_mean", "?")
        print(f"{i+1:4d}  {r['method_id']:50s}  {q1:>10} {s1:>8}  {q3:>10} {s3:>8}")
    print("=" * 120)


if __name__ == "__main__":
    main()
