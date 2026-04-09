"""Train FB on the full freeroam dataset, eval on each quadrant during training."""
import argparse, os, time
from datetime import datetime
import numpy as np, torch, matplotlib.pyplot as plt
from env import ReplayBuffer
from env_quaddyn import Nav2DQuadDyn
from env_quadrant import QUADRANT_BOUNDS
from fb_agent import FBAgent

QS = ["Q1","Q2","Q3","Q4"]
device = torch.device("cuda")


def in_q(rng, q, m=0.10):
    (xlo,xhi),(ylo,yhi) = QUADRANT_BOUNDS[q]
    return np.array([rng.uniform(xlo+m,xhi-m), rng.uniform(ylo+m,yhi-m)], dtype=np.float32)


def eval_in(agent, q, n_goals=20, n_starts=4, n_steps=200, seed=0):
    rng = np.random.RandomState(seed)
    sample_obs = rng.uniform(-1,1, size=(4096,2)).astype(np.float32)
    env = Nav2DQuadDyn(max_steps=n_steps)
    finals = []
    for _ in range(n_goals):
        g = in_q(rng, q)
        d = np.linalg.norm(sample_obs - g[None,:], axis=1)
        rew = (d < 0.07).astype(np.float32)
        if rew.sum() == 0: rew = np.exp(-25.0*d).astype(np.float32)
        z = agent.infer_z_from_rewards(sample_obs, rew)
        for _ in range(n_starts):
            s0 = in_q(rng, q)
            obs = env.reset(state=s0)
            for _ in range(n_steps):
                a = agent.act(obs, z, noise=False)
                obs,_,done,_ = env.step(a)
                if done: break
            finals.append(float(np.linalg.norm(obs - g)))
    f = np.array(finals)
    return float(f.mean()), float((f<0.10).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="checkpoints_offline_quaddyn/freeroam_states_v2.npz")
    p.add_argument("--updates", type=int, default=150_000)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints_freeroam_curve")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    d = np.load(args.data)
    obs, act, nxt = d["obs"], d["act"], d["next_obs"]
    N = obs.shape[0]
    print(f"Freeroam transitions: {N}")
    buf = ReplayBuffer(N+100, 2, 2)
    for i in range(N):
        buf.add(obs[i], act[i], nxt[i])

    agent = FBAgent(2, 2, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                     lr=args.lr, device=device)

    rows = []  # step, [mean_d Q1..Q4], [s10 Q1..Q4]
    def do_eval(step):
        mds, s10s = [], []
        for j, q in enumerate(QS):
            md, s = eval_in(agent, q, seed=step*10 + j)
            mds.append(md); s10s.append(s)
        rows.append((step, mds, s10s))
        msg = f"  step {step:6d}/{args.updates}  " + "  ".join(
            f"{q}:{md:.3f}/{s:.2f}" for q,md,s in zip(QS, mds, s10s))
        print(msg, flush=True)

    do_eval(0)
    t0 = time.time()
    for step in range(1, args.updates + 1):
        batch = buf.sample(args.batch_size, device=device)
        agent.update(batch)
        if step % args.eval_every == 0:
            do_eval(step)
            print(f"    elapsed {time.time()-t0:.0f}s", flush=True)

    steps = np.array([r[0] for r in rows])
    mds = np.array([r[1] for r in rows])  # (T, 4)
    s10 = np.array([r[2] for r in rows])  # (T, 4)
    np.savez(os.path.join(save_dir, "curve.npz"), steps=steps, mds=mds, s10=s10)
    torch.save({"forward_net": agent.forward_net.state_dict(),
                "backward_net": agent.backward_net.state_dict(),
                "actor": agent.actor.state_dict(), "args": vars(args)},
               os.path.join(save_dir, "fb_agent.pt"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["tab:blue","tab:orange","tab:green","tab:red"]
    for j, (q, c) in enumerate(zip(QS, colors)):
        axes[0].plot(steps, mds[:,j], 'o-', color=c, label=q, ms=3, lw=1)
        axes[1].plot(steps, s10[:,j], 'o-', color=c, label=q, ms=3, lw=1)
    axes[0].plot(steps, mds.mean(1), 'k--', lw=2, label='mean')
    axes[1].plot(steps, s10.mean(1), 'k--', lw=2, label='mean')
    axes[0].set_xlabel("update step"); axes[0].set_ylabel("mean final dist"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8); axes[0].set_title("Freeroam training: mean final distance per quadrant")
    axes[1].set_xlabel("update step"); axes[1].set_ylabel("success @ r=0.10"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8); axes[1].set_title("Freeroam training: success @ 0.10 per quadrant")
    axes[1].set_ylim(-0.02, 1.02)
    plt.suptitle(f"FB on full freeroam dataset (300k) — eval during training (seed{args.seed})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve.png"), dpi=140, bbox_inches="tight")
    print(f"\nsaved {save_dir}/")


if __name__ == "__main__":
    main()
