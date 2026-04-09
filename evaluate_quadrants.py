"""
evaluate_quadrants.py — Test each quadrant-trained FB model on all 4 quadrant tasks.

Produces a 4x4 performance matrix:
  rows    = trained model (Q1, Q2, Q3, Q4)
  columns = test task (target = Q1, Q2, Q3, Q4)
  cell    = mean reward (fraction of time spent in target quadrant) over rollouts

Each model is evaluated by:
  1. Inferring z from sampled (s, r) pairs where r = quadrant_indicator(s)
     Sampling is done over the FULL [-1,1]^2 box (not the model's training quadrant)
  2. Rolling out the policy from a few starts INSIDE the model's training quadrant
     (because the model only knows its own quadrant — we can't start it elsewhere)
  3. Measuring fraction of time in the target quadrant
"""

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from env_quadrant import Nav2DQuadrant, QUADRANT_BOUNDS
from fb_agent import FBAgent


QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]


def find_latest_dir(base="checkpoints_quadrant"):
    dirs = sorted(glob.glob(os.path.join(base, "*")))
    if not dirs:
        raise FileNotFoundError(f"No runs found in {base}")
    return dirs[-1]


def load_agent(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    agent = FBAgent(
        obs_dim=2,
        action_dim=2,
        z_dim=args["z_dim"],
        hidden_dim=args["hidden_dim"],
        device=device,
    )
    agent.forward_net.load_state_dict(ckpt["forward_net"])
    agent.backward_net.load_state_dict(ckpt["backward_net"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.forward_net.eval()
    agent.backward_net.eval()
    agent.actor.eval()
    return agent, ckpt["quadrant"]


def in_quadrant(s, q):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
    return (s[0] >= xlo) and (s[0] <= xhi) and (s[1] >= ylo) and (s[1] <= yhi)


def quadrant_reward(target_q):
    def fn(s):
        return 1.0 if in_quadrant(s, target_q) else 0.0
    return fn


def evaluate_model_on_task(agent, train_q, target_q, n_starts=8, n_steps=200,
                            n_z_samples=4096, device="cpu"):
    """
    For a model trained on `train_q`, infer z for the `target_q` task
    and run rollouts. Returns mean fraction of time in target quadrant.

    Sampling for z is done over the model's TRAINING quadrant (since that's
    where its B is well-defined).
    """
    rng = np.random.RandomState(0)
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[train_q]

    # Sample over the model's training distribution
    sample_obs = np.stack([
        rng.uniform(xlo, xhi, size=n_z_samples),
        rng.uniform(ylo, yhi, size=n_z_samples),
    ], axis=-1).astype(np.float32)
    rewards = np.array([quadrant_reward(target_q)(s) for s in sample_obs],
                       dtype=np.float32)
    z = agent.infer_z_from_rewards(sample_obs, rewards)

    # Rollout from random starts inside the training quadrant.
    # Use the FULL [-1,1] env so the agent CAN move to other quadrants if it learned to.
    from env import Nav2D
    env = Nav2D(max_steps=n_steps)

    fractions = []
    trajectories = []
    for _ in range(n_starts):
        start = np.array([
            rng.uniform(xlo, xhi),
            rng.uniform(ylo, yhi),
        ], dtype=np.float32)
        obs = env.reset(state=start)
        traj = [obs.copy()]
        in_target = 0
        for _ in range(n_steps):
            action = agent.act(obs, z, noise=False)
            obs, _, done, _ = env.step(action)
            traj.append(obs.copy())
            if in_quadrant(obs, target_q):
                in_target += 1
            if done:
                break
        fractions.append(in_target / n_steps)
        trajectories.append(np.array(traj))

    return float(np.mean(fractions)), trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.run_dir is None:
        args.run_dir = find_latest_dir()
    print(f"Loading models from: {args.run_dir}")

    # Load all 4 models
    agents = {}
    for q in QUADRANTS:
        ckpt_path = os.path.join(args.run_dir, f"fb_agent_{q}.pt")
        agent, _ = load_agent(ckpt_path, device=args.device)
        agents[q] = agent
        print(f"  Loaded {q}")

    # Evaluate: 4x4 matrix
    print("\nEvaluating each model on each task...")
    perf = np.zeros((4, 4))
    all_trajs = {}  # (train_q, target_q) -> list of trajectories
    for i, train_q in enumerate(QUADRANTS):
        for j, target_q in enumerate(QUADRANTS):
            mean_frac, trajs = evaluate_model_on_task(
                agents[train_q], train_q, target_q, device=args.device
            )
            perf[i, j] = mean_frac
            all_trajs[(train_q, target_q)] = trajs
            print(f"  Model={train_q}  Task={target_q}  fraction_in_target={mean_frac:.3f}")

    # Print table
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE MATRIX (fraction of time in target quadrant)")
    print(f"{'='*60}")
    print(f"  {'Model \\ Task':<15}" + "".join(f"{q:>10}" for q in QUADRANTS))
    print(f"  {'-'*55}")
    for i, train_q in enumerate(QUADRANTS):
        vals = "".join(f"{perf[i,j]:>10.3f}" for j in range(4))
        print(f"  {train_q:<15}{vals}")

    # ---- Heatmap ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(perf, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(QUADRANTS)
    ax.set_yticks(range(4)); ax.set_yticklabels(QUADRANTS)
    ax.set_xlabel("Test Task (target quadrant)")
    ax.set_ylabel("Trained Model")
    for i in range(4):
        for j in range(4):
            color = "white" if perf[i, j] < 0.5 else "black"
            ax.text(j, i, f"{perf[i,j]:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="fraction of time in target")
    ax.set_title("Quadrant FB Models: Cross-Quadrant Performance",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(args.run_dir, "quadrant_performance_matrix.png")
    plt.savefig(heatmap_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {heatmap_path}")

    # ---- Trajectory grid ----
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Trajectories: Each row = trained model, each col = task target",
                 fontsize=13, fontweight="bold")

    quad_colors = {"Q1": "#90EE90", "Q2": "#ADD8E6", "Q3": "#FFB6C1", "Q4": "#FFE4B5"}

    for i, train_q in enumerate(QUADRANTS):
        for j, target_q in enumerate(QUADRANTS):
            ax = axes[i, j]
            # Shade target quadrant
            (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[target_q]
            ax.fill_between([xlo, xhi], ylo, yhi, color="green", alpha=0.15)
            # Outline training quadrant
            (txlo, txhi), (tylo, tyhi) = QUADRANT_BOUNDS[train_q]
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((txlo, tylo), txhi - txlo, tyhi - tylo,
                                    fill=False, edgecolor="blue", linewidth=2,
                                    linestyle="--"))

            for traj in all_trajs[(train_q, target_q)]:
                ax.plot(traj[:, 0], traj[:, 1], '-', alpha=0.6, linewidth=1.5)
                ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=6,
                        markeredgecolor='black')

            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.axhline(0, color='k', linewidth=0.4); ax.axvline(0, color='k', linewidth=0.4)
            ax.set_title(f"Model {train_q} -> Target {target_q}\nfrac={perf[i,j]:.2f}",
                         fontsize=9)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    traj_path = os.path.join(args.run_dir, "quadrant_trajectories_grid.png")
    plt.savefig(traj_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {traj_path}")


if __name__ == "__main__":
    main()
