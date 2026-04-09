"""distill_lib.py — shared helpers for the FB B×C×D distillation sweep.

Each script in `distill_sweep/` runs the same naive-sequential training loop
on the q123 quaddyn task with axis E fixed to "naive FB" (random uniform z).
The only thing that varies between scripts is:

  axis B — *what* to distill: F, B, F+B, M, Q, π
  axis C — *where* the distill input comes from: current batch | (s,a) replay
  axis D — *how* the loss is computed:           L2 | cosine | contrastive | gram

This module provides the building blocks every script imports.

Design principle: do **not** modify FBAgent. After every standard
`agent.update(batch)` step, we do a separate `distill_step` that adds an
extra gradient on the agent's FB optimizer (or actor optimizer for π).

Conventions:
  - Teacher = a frozen deep copy of the agent at the end of the previous stage.
  - z is sampled fresh per distill step from the naive FB distribution
    (uniform on sphere of radius sqrt(z_dim)).
  - For replay (axis C2) we store (s, a) only — never s_next.
"""
import copy
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Teacher snapshot
# ---------------------------------------------------------------------------

def make_teacher(agent):
    """Frozen deep copy of agent's *online* F, B, actor at the end of a stage.

    Note: we copy the online networks, not the Polyak target nets, because the
    online nets are what evaluation uses.
    """
    teacher = {
        "forward_net":  copy.deepcopy(agent.forward_net).eval(),
        "backward_net": copy.deepcopy(agent.backward_net).eval(),
        "actor":        copy.deepcopy(agent.actor).eval(),
    }
    for net in teacher.values():
        for p in net.parameters():
            p.requires_grad_(False)
    return teacher


# ---------------------------------------------------------------------------
# (s, a) replay buffer — for axis C2
# ---------------------------------------------------------------------------

class SAReplayBuffer:
    """Fixed-capacity buffer of (s, a) tuples accumulated across previous stages."""
    def __init__(self, max_size, obs_dim, action_dim, device):
        self.max_size = max_size
        self.obs = torch.zeros(max_size, obs_dim,    device=device)
        self.act = torch.zeros(max_size, action_dim, device=device)
        self.size = 0
        self.ptr  = 0

    def add_batch(self, obs, act):
        """Add a batch of (s, a) — circular if buffer is full."""
        n = obs.size(0)
        for i in range(n):
            self.obs[self.ptr] = obs[i]
            self.act[self.ptr] = act[i]
            self.ptr  = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs.device)
        return self.obs[idx], self.act[idx]


# ---------------------------------------------------------------------------
# z sampling — naive FB (uniform on sphere of radius sqrt(z_dim))
# ---------------------------------------------------------------------------

def sample_z(n, z_dim, device):
    z = torch.randn(n, z_dim, device=device)
    return F.normalize(z, dim=-1) * math.sqrt(z_dim)


# ---------------------------------------------------------------------------
# Output extractors  (work for both `FBAgent` and `teacher` dict)
# ---------------------------------------------------------------------------

def _fwd(net_or_teacher):
    return net_or_teacher["forward_net"]  if isinstance(net_or_teacher, dict) \
           else net_or_teacher.forward_net

def _bwd(net_or_teacher):
    return net_or_teacher["backward_net"] if isinstance(net_or_teacher, dict) \
           else net_or_teacher.backward_net

def _act(net_or_teacher):
    return net_or_teacher["actor"]        if isinstance(net_or_teacher, dict) \
           else net_or_teacher.actor


def get_F(model, s, a, z):
    """Forward feature, twin heads → returns tuple (F1, F2) of shape (n, z_dim)."""
    return _fwd(model)(s, a, z)

def get_B(model, s):
    """Backward feature, shape (n, z_dim)."""
    return _bwd(model)(s)

def get_M(model, s, a, z):
    """Successor measure M = F1 @ B(s)^T  →  (n, n).
    We use head1 of the twin F (consistent across student/teacher)."""
    F1, _ = _fwd(model)(s, a, z)
    B     = _bwd(model)(s)
    return F1 @ B.T

def get_Q(model, s, a, z):
    """Per-sample Q = (F1 · z).sum(-1)  →  (n,)."""
    F1, _ = _fwd(model)(s, a, z)
    return (F1 * z).sum(-1)

def get_pi(model, s, z):
    """Deterministic actor output → (n, action_dim)."""
    return _act(model)(s, z)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
# All loss functions accept either a single tensor or a tuple of tensors
# (the F target uses a tuple because F has twin heads). The reduction is
# always over all elements / mean over the batch as appropriate.
# ---------------------------------------------------------------------------

def loss_l2(student, teacher):
    """Mean squared error."""
    if isinstance(student, tuple):
        return sum(F.mse_loss(s, t) for s, t in zip(student, teacher))
    return F.mse_loss(student, teacher)


def loss_cosine(student, teacher):
    """1 - cosine_similarity, averaged over the batch."""
    def _cos(s, t):
        if s.dim() == 1:  # scalar per sample (Q): treat batch as a vector
            s = s.unsqueeze(0); t = t.unsqueeze(0)
        return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
    if isinstance(student, tuple):
        return sum(_cos(s, t) for s, t in zip(student, teacher))
    return _cos(student, teacher)


def loss_contrastive(student_M, teacher_M, temperature=0.1):
    """InfoNCE where each row of student_M should match the same row of teacher_M.
    Used primarily on the M = F·Bᵀ matrix.

    For non-matrix targets (F, B, π), each sample's vector is the row.
    For 1D scalar-per-sample targets (Q), we promote them to (n, 1) so that
    each sample becomes a 1-dim "feature" — InfoNCE then degenerates to
    matching scalar magnitudes via signed similarity. Not very meaningful but
    runs without errors.
    """
    if isinstance(student_M, tuple):
        return sum(loss_contrastive(s, t, temperature)
                   for s, t in zip(student_M, teacher_M))
    if student_M.dim() == 1:
        student_M = student_M.unsqueeze(-1)
        teacher_M = teacher_M.unsqueeze(-1)
    s = F.normalize(student_M, dim=-1)
    t = F.normalize(teacher_M, dim=-1)
    logits = s @ t.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def loss_gram(student, teacher):
    """Match Gram matrices  X X^T  vs  Y Y^T  in MSE."""
    def _gram(s, t):
        return F.mse_loss(s @ s.T, t @ t.T)
    if isinstance(student, tuple):
        return sum(_gram(s, t) for s, t in zip(student, teacher))
    return _gram(student, teacher)


# ---------------------------------------------------------------------------
# Joint update — mirror of FBAgent.update with optional distillation losses
# ---------------------------------------------------------------------------
# This is the **single combined backward pass** alternative to running
# `agent.update(batch)` and then a separate distill_step. It re-implements the
# FB body so that any extra distillation losses can be added BEFORE the
# backward/step calls. This way the distillation gradient is fused with the
# FB gradient inside one Adam step (one momentum / variance update), exactly
# like a paper would write `loss = fb_loss + alpha * distill_loss`.
#
# The two callables `fb_extra_fn` and `pi_extra_fn` are closures provided by
# each per-script wrapper. They are called *inside* this function so the loss
# tensors are built on the same forward graph as the FB and actor losses.
#
#   fb_extra_fn(ctx) → scalar tensor, added to total_fb_loss before
#                      fb_opt.step(). Use this for B1..B5 targets
#                      (F, B, F+B, M, Q). The `ctx` dict contains everything
#                      from the FB step's forward pass:
#                        ctx["obs"], ctx["act"], ctx["next_obs"]
#                        ctx["z"]            – z used by FB step
#                        ctx["F1"], ctx["F2"]– student forward features (twin)
#                        ctx["B"]            – student backward features
#                      For axis C = current the closure can directly reuse
#                      ctx["F1"], ctx["F2"], ctx["B"] (no extra forward pass).
#                      For axis C = replay it samples its own (s, a) and runs
#                      its own forward through the agent.
#
#   pi_extra_fn(ctx) → scalar tensor, added to actor_loss before actor_opt.step()
#                      (use this for B7: π distillation). The `ctx` here is
#                      a separate dict containing:
#                        ctx["obs"], ctx["z_actor"], ctx["action_pi"]
#                      so the closure can reuse the actor's forward pass.
#
# Setting either to None disables that branch (e.g. stage 1 has no teacher).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Weights & Biases logging helpers
# ---------------------------------------------------------------------------
# The scripts never import wandb directly — they go through `maybe_init_wandb`
# and `wandb_log`, which are no-ops if wandb isn't installed or if
# `--wandb_project` is not set. This way every script works out of the box
# without wandb, and opting in is a single CLI flag.
# ---------------------------------------------------------------------------

def maybe_init_wandb(args, B, D, C):
    """Initialise a W&B run if `args.wandb_project` is set. Returns the run
    object (or None). Safe to call when wandb is not installed."""
    if not getattr(args, "wandb_project", None):
        return None
    try:
        import wandb
    except ImportError:
        print("[distill_lib] wandb not installed; logging disabled", flush=True)
        return None
    run_name = getattr(args, "wandb_run_name", None) \
               or f"{B}_{D}_{C}_seed{args.seed}"
    run = wandb.init(
        project=args.wandb_project,
        entity=getattr(args, "wandb_entity", None) or None,
        name=run_name,
        config={
            "B":                 B,
            "D":                 D,
            "C":                 C,
            "z_dim":             args.z_dim,
            "hidden_dim":        args.hidden_dim,
            "lr":                args.lr,
            "batch_size":        args.batch_size,
            "alpha_distill":     args.alpha_distill,
            "updates_per_stage": args.updates_per_stage,
            "seed":              args.seed,
        },
        tags=[B, D, C, "q123", "naive_FB_z_sampler"],
        reinit=True,
    )
    return run


def wandb_log(run, payload, step=None):
    """Safe no-op wrapper around wandb.log."""
    if run is None:
        return
    import wandb
    wandb.log(payload, step=step)


def wandb_finish(run, summary=None):
    """Save final summary metrics and finish the run (no-op if run is None)."""
    if run is None:
        return
    import wandb
    if summary:
        for k, v in summary.items():
            wandb.summary[k] = v
    wandb.finish()


def joint_update(agent, batch, fb_extra_fn=None, pi_extra_fn=None):
    import torch  # local to keep this module lightweight

    obs      = batch["obs"]
    act      = batch["act"]
    next_obs = batch["next_obs"]
    n = obs.size(0)

    # Sample z and mix (identical to FBAgent.update)
    z = agent.sample_z(n)
    z = agent.mix_z_with_B(z, next_obs)

    # ---- FB TD target ----
    with torch.no_grad():
        next_action = agent.actor(next_obs, z)
        next_action = next_action + torch.randn_like(next_action) * agent.actor_noise
        next_action = next_action.clamp(-1.0, 1.0)
        target_F1, target_F2 = agent.forward_target(next_obs, next_action, z)
        target_B  = agent.backward_target(next_obs)
        target_M1 = target_F1 @ target_B.T
        target_M2 = target_F2 @ target_B.T
        target_M  = torch.minimum(target_M1, target_M2)

    F1, F2 = agent.forward_net(obs, act, z)
    B      = agent.backward_net(next_obs)
    M1 = F1 @ B.T
    M2 = F2 @ B.T

    I        = torch.eye(n, device=agent.device)
    off_diag = ~I.bool()

    fb_offdiag = 0.0
    for M in [M1, M2]:
        diff = M - agent.discount * target_M
        fb_offdiag = fb_offdiag + 0.5 * (diff[off_diag].pow(2)).mean()
    fb_diag = -sum(M.diag().mean() for M in [M1, M2])
    fb_loss = fb_offdiag + fb_diag

    Cov = B @ B.T
    ortho_diag    = -2.0 * Cov.diag().mean()
    ortho_offdiag = (Cov[off_diag]).pow(2).mean()
    ortho_loss    = ortho_diag + ortho_offdiag

    total_fb_loss = fb_loss + agent.ortho_coef * ortho_loss

    # ---- FB-side distillation (B1..B5 targets) ----
    distill_fb_val = 0.0
    if fb_extra_fn is not None:
        fb_ctx = {
            "obs": obs, "act": act, "next_obs": next_obs,
            "z":   z,
            "F1":  F1, "F2": F2,
            "B":   B,
        }
        extra = fb_extra_fn(fb_ctx)
        total_fb_loss = total_fb_loss + extra
        distill_fb_val = float(extra.item())

    agent.fb_opt.zero_grad()
    total_fb_loss.backward()
    agent.fb_opt.step()

    # ---- Actor loss ----
    with torch.no_grad():
        z_actor = agent.sample_z(n)
    action_pi   = agent.actor(obs, z_actor)
    F1_pi, F2_pi = agent.forward_net(obs, action_pi, z_actor)
    Q1 = (F1_pi * z_actor).sum(dim=-1)
    Q2 = (F2_pi * z_actor).sum(dim=-1)
    Q  = torch.minimum(Q1, Q2)
    actor_loss = -Q.mean()

    # ---- π-side distillation (B7) ----
    distill_pi_val = 0.0
    if pi_extra_fn is not None:
        pi_ctx = {
            "obs":       obs,
            "z_actor":   z_actor,
            "action_pi": action_pi,
        }
        extra = pi_extra_fn(pi_ctx)
        actor_loss = actor_loss + extra
        distill_pi_val = float(extra.item())

    agent.actor_opt.zero_grad()
    actor_loss.backward()
    agent.actor_opt.step()

    # ---- Polyak target update ----
    with torch.no_grad():
        for p, tp in zip(agent.forward_net.parameters(),
                         agent.forward_target.parameters()):
            tp.data.mul_(1 - agent.tau).add_(agent.tau * p.data)
        for p, tp in zip(agent.backward_net.parameters(),
                         agent.backward_target.parameters()):
            tp.data.mul_(1 - agent.tau).add_(agent.tau * p.data)

    return {
        "fb_loss":    fb_loss.item(),
        "actor_loss": actor_loss.item(),
        "distill_fb": distill_fb_val,
        "distill_pi": distill_pi_val,
    }


# Convenience dispatch tables — used by the per-script wrappers.
TARGETS = {
    "F":  get_F,    # returns (F1, F2)
    "B":  get_B,    # returns (n, z_dim)
    "FB": None,     # special: see scripts (they call both)
    "M":  get_M,    # returns (n, n)
    "Q":  get_Q,    # returns (n,)
    "pi": get_pi,   # returns (n, action_dim) — uses actor_opt, not fb_opt
}

LOSSES = {
    "l2":          loss_l2,
    "cosine":      loss_cosine,
    "contrastive": loss_contrastive,
    "gram":        loss_gram,
}
