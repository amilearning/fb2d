"""Train one FB model from a single per-quadrant sub-dataset."""
import argparse, os, time
from datetime import datetime
import numpy as np, torch
from env import ReplayBuffer
from fb_agent import FBAgent
from eval_freeroam import evaluate as rich_eval, summary_str

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quadrant", required=True, choices=["Q1","Q2","Q3","Q4"])
    p.add_argument("--data", default="checkpoints_offline_quaddyn/sub_perq.npz")
    p.add_argument("--updates", type=int, default=60_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints_offline_subq")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.quadrant}_seed{args.seed}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    d = np.load(args.data)
    obs = d[f"{args.quadrant}_obs"]
    act = d[f"{args.quadrant}_act"]
    nxt = d[f"{args.quadrant}_next"]
    N = obs.shape[0]
    print(f"{args.quadrant}: {N} transitions")
    buf = ReplayBuffer(N + 100, 2, 2)
    for i in range(N):
        buf.add(obs[i], act[i], nxt[i])

    agent = FBAgent(2, 2, z_dim=args.z_dim, hidden_dim=args.hidden_dim,
                     lr=args.lr, device=device)
    t0 = time.time()
    for step in range(1, args.updates + 1):
        batch = buf.sample(args.batch_size, device=device)
        m = agent.update(batch)
        if step % 5000 == 0:
            print(f"  {args.quadrant} step {step:6d}  fb={m['fb_loss']:.2f} "
                  f"actor={m['actor_loss']:.2f}  ({time.time()-t0:.0f}s)")

    res = rich_eval(agent, seed=0)
    print(summary_str(args.quadrant, res))
    np.savez(os.path.join(save_dir, "eval.npz"),
             finals=res["finals"], ttgs=res["ttgs"], effs=res["effs"],
             ora_finals=res["ora_finals"], ora_ttgs=res["ora_ttgs"],
             heat=res["heat"])
    torch.save({"forward_net": agent.forward_net.state_dict(),
                "backward_net": agent.backward_net.state_dict(),
                "actor": agent.actor.state_dict(),
                "args": vars(args)},
               os.path.join(save_dir, "fb_agent.pt"))


if __name__ == "__main__":
    main()
