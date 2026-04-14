"""
eval_all_methods.py — Evaluate all 113 FB continual-learning methods on Q1 (memory)
and Q3 (plasticity) with 5 eval seeds.

Usage:
    python eval_all_methods.py [--workers 16] [--dry-run]
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, Value, Lock

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from FB2D
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/frl/FB2D")
from env_quaddyn import Nav2DQuadDyn
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEEDS = [100, 200, 300, 400, 500]
EVAL_QUADRANTS = ["Q1", "Q3"]
N_GOALS = 40
N_STARTS = 4
N_STEPS = 200

# Checkpoint directories to scan  (name -> path)
SWEEP_DIRS = {
    "distill":          "/home/frl/FB2D/checkpoints_distill_real",
    "distill_combined": "/home/frl/FB2D/checkpoints_distill_combined_real",
    "z_sampling":       "/home/frl/FB2D/checkpoints_z_sampling_real",
    "vmf_sweep":        "/home/frl/FB2D/checkpoints_vmf_sweep_real",
    "fdws_sweep":       "/home/frl/FB2D/checkpoints_fdws_sweep_real",
}

# z=32 baselines (not multi-config sweeps — just individual dirs)
BASELINE_DIRS = {
    "naive_seq_z32": "/home/frl/FB2D/checkpoints_naive_seq",
    "taskincr_z32":  "/home/frl/FB2D/checkpoints_taskincr",
}

RESULTS_DIR = "/home/frl/FB2D/eval_results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "eval_all_methods.csv")
RAW_CSV     = os.path.join(RESULTS_DIR, "eval_all_methods_raw.csv")


# ---------------------------------------------------------------------------
# Eval function (copied from training scripts)
# ---------------------------------------------------------------------------
def in_q(rng, q, m=0.10):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
    return np.array([rng.uniform(xlo + m, xhi - m),
                     rng.uniform(ylo + m, yhi - m)], dtype=np.float32)


def eval_in(agent, eq, n_goals=N_GOALS, n_starts=N_STARTS,
            n_steps=N_STEPS, seed=0):
    """Evaluate agent in quadrant *eq*. Returns array of final distances."""
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1, 1, size=(4096, 2)).astype(np.float32)
    env = Nav2DQuadDyn(max_steps=n_steps)
    finals = []
    for _ in range(n_goals):
        g = in_q(rng, eq)
        d = np.linalg.norm(sample_obs - g[None, :], axis=1)
        rew = (d < 0.07).astype(np.float32)
        if rew.sum() == 0:
            rew = np.exp(-25.0 * d).astype(np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rew)
        for _ in range(n_starts):
            s0 = in_q(rng, eq)
            obs = env.reset(state=s0)
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs, _, done, _ = env.step(a)
                if done:
                    break
            finals.append(float(np.linalg.norm(obs - g)))
    return np.array(finals)


# ---------------------------------------------------------------------------
# Discovery: find all (method_id, config_name, sweep, stage3_path)
# ---------------------------------------------------------------------------
def discover_methods():
    """Return list of dicts describing every evaluation unit.

    Each dict: {method_id, config_name, sweep, seed_dir, stage3_path, z_dim}
    method_id groups all training seeds of the same config.
    """
    methods = []  # list of (method_id, config_name, sweep, seed_dir, stage3_path, z_dim)

    # --- sweep dirs: each subdir is a config, inside it seed* subdirs ---
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

    # --- baseline dirs: seed subdirs directly under the dir ---
    for base_name, base_path in BASELINE_DIRS.items():
        if not os.path.isdir(base_path):
            print(f"[WARN] baseline dir missing: {base_path}")
            continue
        method_id = base_name
        # Pick the latest run for each seed number (skip duplicates)
        seed_runs = {}  # seed_num -> (timestamp, path)
        for entry in sorted(os.listdir(base_path)):
            sd = os.path.join(base_path, entry)
            stage3 = os.path.join(sd, "stage3.pt")
            if not os.path.isdir(sd) or not os.path.isfile(stage3):
                continue
            # Parse seed number from e.g. "seed42_20260408_155620"
            parts = entry.split("_")
            if not parts[0].startswith("seed"):
                continue
            seed_num = parts[0]  # "seed42"
            ts = "_".join(parts[1:])  # "20260408_155620"
            if seed_num not in seed_runs or ts > seed_runs[seed_num][0]:
                seed_runs[seed_num] = (ts, sd, stage3)

        for seed_num, (ts, sd, stage3) in sorted(seed_runs.items()):
            # Verify z_dim = 32
            try:
                ckpt = torch.load(stage3, map_location="cpu")
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
# Worker: evaluate one (method, training seed) on all quadrants × eval seeds
# ---------------------------------------------------------------------------
_counter = None
_lock = None
_total = None


def _init_worker(counter, lock, total):
    global _counter, _lock, _total
    _counter = counter
    _lock = lock
    _total = total


def eval_one_seed_dir(task):
    """Evaluate a single stage3.pt on Q1 and Q3 for all eval seeds.

    Returns list of raw-result dicts.
    """
    method_id = task["method_id"]
    config_name = task["config_name"]
    sweep = task["sweep"]
    seed_dir = task["seed_dir"]
    stage3_path = task["stage3_path"]
    z_dim = task["z_dim"]

    results = []
    try:
        ckpt = torch.load(stage3_path, map_location="cpu")
        agent = FBAgent(2, 2, z_dim=z_dim, hidden_dim=256, device="cpu")
        agent.forward_net.load_state_dict(ckpt["forward_net"])
        agent.backward_net.load_state_dict(ckpt["backward_net"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.forward_net.eval()
        agent.backward_net.eval()
        agent.actor.eval()

        train_seed_label = os.path.basename(seed_dir)

        for quad in EVAL_QUADRANTS:
            for es in EVAL_SEEDS:
                finals = eval_in(agent, quad, seed=es)
                mean_d = float(finals.mean())
                s10 = float((finals < 0.10).mean())
                results.append({
                    "method_id": method_id,
                    "config_name": config_name,
                    "sweep": sweep,
                    "train_seed": train_seed_label,
                    "eval_seed": es,
                    "quadrant": quad,
                    "mean_d": mean_d,
                    "s10": s10,
                })
    except Exception as e:
        print(f"[ERROR] {stage3_path}: {e}", flush=True)

    # Increment shared counter
    with _lock:
        _counter.value += 1
        cnt = _counter.value
    print(f"  [{cnt}/{_total.value}] done: {method_id} / {os.path.basename(seed_dir)}",
          flush=True)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(raw_rows):
    """Group raw rows by method_id and compute summary stats."""
    by_method = defaultdict(list)
    for r in raw_rows:
        by_method[r["method_id"]].append(r)

    summary = []
    for mid, rows in sorted(by_method.items()):
        config_name = rows[0]["config_name"]
        sweep = rows[0]["sweep"]
        n_train = len(set(r["train_seed"] for r in rows))
        n_eval = len(EVAL_SEEDS)

        vals = {}
        for quad in EVAL_QUADRANTS:
            qrows = [r for r in rows if r["quadrant"] == quad]
            md = np.array([r["mean_d"] for r in qrows])
            s10 = np.array([r["s10"] for r in qrows])
            vals[f"mean_d_{quad}_mean"] = float(md.mean())
            vals[f"mean_d_{quad}_std"]  = float(md.std())
            vals[f"s10_{quad}_mean"]    = float(s10.mean())
            vals[f"s10_{quad}_std"]     = float(s10.std())

        summary.append({
            "method_id": mid,
            "config_name": config_name,
            "sweep": sweep,
            "n_train_seeds": n_train,
            "n_eval_seeds": n_eval,
            **vals,
        })

    # Sort by combined mean_d (lower is better)
    summary.sort(key=lambda r: r["mean_d_Q1_mean"] + r["mean_d_Q3_mean"])
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only discover methods, do not evaluate")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Discovering methods ...")
    methods = discover_methods()
    print(f"Found {len(methods)} seed-dirs across "
          f"{len(set(m['method_id'] for m in methods))} method configs.\n")

    # Print per-sweep breakdown
    from collections import Counter
    sweep_counts = Counter(m["sweep"] for m in methods)
    for sw, cnt in sorted(sweep_counts.items()):
        n_configs = len(set(m["method_id"] for m in methods if m["sweep"] == sw))
        print(f"  {sw}: {n_configs} configs, {cnt} seed-dirs")

    if args.dry_run:
        print("\n[dry-run] Exiting without evaluation.")
        for m in methods[:10]:
            print(f"  {m['method_id']} / {os.path.basename(m['seed_dir'])}")
        if len(methods) > 10:
            print(f"  ... and {len(methods)-10} more")
        return

    # Prepare tasks with shared counter
    counter = Value("i", 0)
    lock = Lock()
    total_val = Value("i", len(methods))
    tasks = list(methods)  # no extra keys needed, counter is global

    print(f"\nStarting evaluation with {args.workers} workers ...")
    print(f"Each seed-dir: {len(EVAL_QUADRANTS)} quadrants × "
          f"{len(EVAL_SEEDS)} eval seeds = "
          f"{len(EVAL_QUADRANTS)*len(EVAL_SEEDS)} evaluations\n")

    t0 = time.time()
    raw_rows = []
    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(counter, lock, total_val)) as pool:
        for result_batch in pool.imap_unordered(eval_one_seed_dir, tasks):
            raw_rows.extend(result_batch)

    elapsed = time.time() - t0
    print(f"\nAll evaluations done in {elapsed/60:.1f} min.  "
          f"({len(raw_rows)} raw measurements)")

    # --- Save raw CSV ---
    raw_fields = ["method_id", "config_name", "sweep", "train_seed",
                  "eval_seed", "quadrant", "mean_d", "s10"]
    with open(RAW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    print(f"Raw results saved to {RAW_CSV}")

    # --- Aggregate and save summary CSV ---
    summary = aggregate(raw_rows)
    summary_fields = [
        "method_id", "config_name", "sweep", "n_train_seeds", "n_eval_seeds",
        "mean_d_Q1_mean", "mean_d_Q1_std", "s10_Q1_mean", "s10_Q1_std",
        "mean_d_Q3_mean", "mean_d_Q3_std", "s10_Q3_mean", "s10_Q3_std",
    ]
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"Summary saved to {SUMMARY_CSV}")

    # --- Print top-20 table ---
    print("\n" + "=" * 110)
    print(f"{'Rank':>4}  {'Method':50s}  {'mean_d_Q1':>10} {'s10_Q1':>8}"
          f"  {'mean_d_Q3':>10} {'s10_Q3':>8}  {'combined':>10}")
    print("-" * 110)
    for i, r in enumerate(summary[:20]):
        combined = r["mean_d_Q1_mean"] + r["mean_d_Q3_mean"]
        print(f"{i+1:4d}  {r['method_id']:50s}"
              f"  {r['mean_d_Q1_mean']:10.4f} {r['s10_Q1_mean']:8.3f}"
              f"  {r['mean_d_Q3_mean']:10.4f} {r['s10_Q3_mean']:8.3f}"
              f"  {combined:10.4f}")
    if len(summary) > 20:
        print(f"  ... {len(summary)-20} more methods (see CSV)")
    print("=" * 110)


if __name__ == "__main__":
    main()
