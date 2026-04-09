"""
eval_freeroam.py — Rich evaluation suite for FB agents on Nav2DQuadDyn.

Metrics:
  1. Success-rate-vs-radius curve (radii 0.05..0.3)
  2. Time-to-goal histogram
  3. 10x10 per-goal-location success heatmap
  4. Oracle-relative regret (oracle = greedy-toward-goal with sign flips known)
"""
import argparse, os, glob, json
import numpy as np
import torch
import matplotlib.pyplot as plt

from env_quaddyn import Nav2DQuadDyn, state_quadrant, transform_action
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RADII = [0.05, 0.10, 0.15, 0.20, 0.30]


def load_agent(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    ag = FBAgent(2, 2, z_dim=a["z_dim"], hidden_dim=a["hidden_dim"], lr=1e-4, device=device)
    ag.forward_net.load_state_dict(ck["forward_net"])
    ag.backward_net.load_state_dict(ck["backward_net"])
    ag.actor.load_state_dict(ck["actor"])
    return ag


def infer_z_for_goal(agent, goal, sample_obs):
    dists = np.linalg.norm(sample_obs - goal[None,:], axis=1)
    rewards = (dists < 0.07).astype(np.float32)
    if rewards.sum() == 0:
        rewards = np.exp(-25.0 * dists).astype(np.float32)
    return agent.infer_z_from_rewards(sample_obs, rewards)


def rollout(agent, z, start, goal, n_steps=200):
    env = Nav2DQuadDyn(max_steps=n_steps)
    obs = env.reset(state=start)
    pts = [obs.copy()]
    first_reach = None
    for t in range(n_steps):
        a = agent.act(obs, z, noise=False)
        obs,_,done,_ = env.step(a)
        pts.append(obs.copy())
        if first_reach is None and np.linalg.norm(obs - goal) < 0.10:
            first_reach = t
        if done: break
    pts = np.array(pts)
    final_d = float(np.linalg.norm(pts[-1] - goal))
    path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    straight = float(np.linalg.norm(start - goal)) + 1e-8
    return final_d, first_reach, path_len, straight, pts


def oracle_rollout(start, goal, n_steps=200, max_speed=0.05):
    env = Nav2DQuadDyn(max_steps=n_steps, max_speed=max_speed)
    obs = env.reset(state=start)
    first_reach = None
    for t in range(n_steps):
        d = goal - obs
        # commanded = transform(sign(d), q): then env transform applies again, net = sign(d)
        cmd = transform_action(np.sign(d).astype(np.float32), state_quadrant(obs))
        obs,_,done,_ = env.step(cmd)
        if first_reach is None and np.linalg.norm(obs - goal) < 0.10:
            first_reach = t
        if done: break
    final_d = float(np.linalg.norm(obs - goal))
    return final_d, first_reach


def _sample_in_quadrant(rng, q, margin=0.10):
    (xlo, xhi), (ylo, yhi) = QUADRANT_BOUNDS[q]
    return np.array([rng.uniform(xlo + margin, xhi - margin),
                      rng.uniform(ylo + margin, yhi - margin)], dtype=np.float32)


def evaluate(agent, n_random_goals=40, n_starts_per_goal=4, n_steps=200,
              grid=10, n_z=4096, seed=0):
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1, 1, size=(n_z, 2)).astype(np.float32)

    # ---- 1+2+6: random goals + oracle regret ----
    # Goals and starts BOTH sampled inside the same quadrant.
    finals = []; ttgs = []; effs = []; ora_finals = []; ora_ttgs = []
    QS = ["Q1","Q2","Q3","Q4"]
    goals = []
    for _ in range(n_random_goals):
        q = QS[rng.randint(4)]
        goals.append((q, _sample_in_quadrant(rng, q)))
    for q_g, g in goals:
        z = infer_z_for_goal(agent, g, sample_obs)
        for _ in range(n_starts_per_goal):
            s0 = _sample_in_quadrant(rng, q_g)
            fd, fr, pl, st, _ = rollout(agent, z, s0, g, n_steps)
            finals.append(fd); ttgs.append(fr if fr is not None else n_steps)
            effs.append(pl / st)
            ofd, ofr = oracle_rollout(s0, g, n_steps)
            ora_finals.append(ofd); ora_ttgs.append(ofr if ofr is not None else n_steps)

    finals = np.array(finals); ttgs = np.array(ttgs); effs = np.array(effs)
    ora_finals = np.array(ora_finals); ora_ttgs = np.array(ora_ttgs)
    succ_curve = {r: float((finals < r).mean()) for r in RADII}
    ora_succ = {r: float((ora_finals < r).mean()) for r in RADII}

    # ---- 4: per-goal heatmap on a fixed grid (start always sampled in goal's quadrant) ----
    xs = np.linspace(-0.85, 0.85, grid)
    ys = np.linspace(-0.85, 0.85, grid)
    heat = np.zeros((grid, grid))
    for ix, gx in enumerate(xs):
        for iy, gy in enumerate(ys):
            g = np.array([gx, gy], dtype=np.float32)
            qg = state_quadrant(g)
            z = infer_z_for_goal(agent, g, sample_obs)
            succs = 0
            for _ in range(4):
                s0 = _sample_in_quadrant(rng, qg)
                fd,_,_,_,_ = rollout(agent, z, s0, g, n_steps)
                if fd < 0.10: succs += 1
            heat[iy, ix] = succs / 4

    return {
        "finals": finals, "ttgs": ttgs, "effs": effs,
        "ora_finals": ora_finals, "ora_ttgs": ora_ttgs,
        "succ_curve": succ_curve, "ora_succ_curve": ora_succ,
        "heat": heat, "heat_xs": xs, "heat_ys": ys,
    }


def summary_str(name, res):
    s = f"=== {name} ===\n"
    s += "  success@radius:\n"
    for r in RADII:
        s += f"    r={r:.2f}: agent={res['succ_curve'][r]:.2f}  oracle={res['ora_succ_curve'][r]:.2f}\n"
    s += f"  mean final dist:  agent={res['finals'].mean():.3f}  oracle={res['ora_finals'].mean():.3f}\n"
    s += f"  median TTG@0.1:   agent={float(np.median(res['ttgs'])):.0f}  oracle={float(np.median(res['ora_ttgs'])):.0f}\n"
    s += f"  mean path eff.:   {res['effs'].mean():.2f}  (1.0 = optimal straight line)\n"
    return s


def plot_all(results, names, out_path):
    n = len(results)
    fig = plt.figure(figsize=(15, 4 * n))
    for i, (name, res) in enumerate(zip(names, results)):
        # success curve
        ax = fig.add_subplot(n, 4, 4*i + 1)
        ax.plot(RADII, [res["succ_curve"][r] for r in RADII], 'o-', label="agent", lw=2)
        ax.plot(RADII, [res["ora_succ_curve"][r] for r in RADII], 's--', label="oracle", lw=2)
        ax.set_xlabel("success radius"); ax.set_ylabel("success rate")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"{name} — success curve", fontsize=9)

        # TTG hist
        ax = fig.add_subplot(n, 4, 4*i + 2)
        ax.hist(res["ttgs"], bins=20, alpha=0.6, label="agent")
        ax.hist(res["ora_ttgs"], bins=20, alpha=0.6, label="oracle")
        ax.set_xlabel("first time in r=0.1"); ax.set_ylabel("count")
        ax.legend(fontsize=8); ax.set_title(f"{name} — TTG", fontsize=9)

        # path efficiency
        ax = fig.add_subplot(n, 4, 4*i + 3)
        ax.hist(res["effs"], bins=30, color='tab:purple')
        ax.axvline(1.0, color='black', ls='--', label='optimal')
        ax.set_xlabel("path / straight distance"); ax.set_ylabel("count")
        ax.legend(fontsize=8); ax.set_title(f"{name} — path efficiency", fontsize=9)
        ax.set_xlim(0, min(10, np.percentile(res["effs"], 95)*1.2))

        # heatmap
        ax = fig.add_subplot(n, 4, 4*i + 4)
        im = ax.imshow(res["heat"], origin='lower', extent=[-0.85,0.85,-0.85,0.85],
                       cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
        ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"{name} — success/goal location\n(grid 10×10, 4 starts each)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="paths to fb_agent.pt files")
    p.add_argument("--out", default="checkpoints_offline_freeroam/eval_full.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results, names = [], []
    for c in args.ckpts:
        ag = load_agent(c)
        res = evaluate(ag, seed=args.seed)
        name = os.path.basename(os.path.dirname(c))
        results.append(res); names.append(name)
        print(summary_str(name, res))
    plot_all(results, names, args.out)
    print(f"\nsaved {args.out}")

    # also dump JSON summary
    js = []
    for n, r in zip(names, results):
        js.append({
            "name": n,
            "succ_curve": r["succ_curve"],
            "ora_succ_curve": r["ora_succ_curve"],
            "mean_final": float(r["finals"].mean()),
            "ora_mean_final": float(r["ora_finals"].mean()),
            "median_ttg": float(np.median(r["ttgs"])),
            "ora_median_ttg": float(np.median(r["ora_ttgs"])),
            "mean_path_eff": float(r["effs"].mean()),
            "heat_mean": float(r["heat"].mean()),
        })
    with open(args.out.replace(".png", ".json"), "w") as f:
        json.dump(js, f, indent=2)


if __name__ == "__main__":
    main()
