"""z_sampling_lib.py — z-sampling strategies for axis E of the FB distillation sweep.

Implements:
  - vMF task-binding and non-task-binding samplers
  - z-sensitivity computation (per z-dim, through F / B(M) / π / combinations)
  - SMR (Sensitivity-Masked Routing): hard-mask high-sensitivity z-dims
  - FDWS (Fisher-weighted z-sampling): soft-weight z-dims inversely by sensitivity

All samplers return z tensors on the sphere of radius sqrt(z_dim), same as naive FB.
"""
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# vMF sampling
# ---------------------------------------------------------------------------

class VMFTaskSampler:
    """Per-task vMF sampler. Stage k samples z ~ vMF(mu_k, kappa).

    mu's are initialized as equally-spaced orthogonal directions (first K
    standard basis vectors, scaled to sqrt(z_dim)).
    """
    def __init__(self, n_tasks, z_dim, kappa=10.0, device="cpu"):
        self.z_dim = z_dim
        self.kappa = kappa
        self.device = device
        # Initialize mus as orthogonal basis vectors
        self.mus = torch.zeros(n_tasks, z_dim, device=device)
        for k in range(min(n_tasks, z_dim)):
            self.mus[k, k] = math.sqrt(z_dim)

    def sample(self, n, task_idx):
        """Sample n z-vectors from vMF(mu_{task_idx}, kappa)."""
        mu = self.mus[task_idx]  # (z_dim,)
        return _sample_vmf(mu, self.kappa, n, self.device)

    def sample_mixture(self, n):
        """Sample from the uniform mixture of all components."""
        k = self.mus.size(0)
        assignments = torch.randint(0, k, (n,))
        zs = torch.zeros(n, self.z_dim, device=self.device)
        for i in range(k):
            mask = (assignments == i)
            ni = mask.sum().item()
            if ni > 0:
                zs[mask] = _sample_vmf(self.mus[i], self.kappa, ni, self.device)
        return zs


def _sample_vmf(mu, kappa, n, device):
    """Approximate vMF sampling via rejection-free Wood's algorithm simplified
    for moderate kappa. For small kappa falls back to perturbed mean."""
    d = mu.size(0)
    mu_norm = F.normalize(mu, dim=0)

    # Sample direction around north pole, then rotate to mu
    # Use the "tangent normal" approximation: z = mu + noise, renormalize
    noise = torch.randn(n, d, device=device) / math.sqrt(kappa)
    z = mu_norm.unsqueeze(0) + noise
    z = F.normalize(z, dim=-1) * math.sqrt(d)
    return z


# ---------------------------------------------------------------------------
# z-sensitivity computation
# ---------------------------------------------------------------------------

def compute_z_sensitivity(agent, teacher, replay_obs, replay_act, source="FB",
                          n_samples=512):
    """Compute per-z-dim sensitivity on old-task data.

    The sensitivity measures how much each z-dimension affects the specified
    network's output. High sensitivity = that z-dim is important for old tasks.

    Args:
        agent: current FBAgent
        teacher: frozen teacher dict (from make_teacher)
        replay_obs: (N, obs_dim) old-task observations
        replay_act: (N, act_dim) old-task actions
        source: which network(s) to measure sensitivity through
            "F"    — sensitivity of F(s,a,z) w.r.t. z
            "B"    — sensitivity of M=F·B^T w.r.t. z (B contributes via M)
            "FB"   — sensitivity of M=F·B^T w.r.t. z (full path)
            "pi"   — sensitivity of pi(s,z) w.r.t. z
            "piFB" — combined pi + FB sensitivity
        n_samples: number of replay samples to use

    Returns:
        sensitivity: (z_dim,) vector, non-negative, higher = more sensitive
    """
    device = next(agent.forward_net.parameters()).device
    z_dim = agent.z_dim
    idx = torch.randint(0, replay_obs.size(0), (n_samples,))
    s = replay_obs[idx].to(device)
    a = replay_act[idx].to(device)

    # Create z that requires grad
    z = torch.randn(n_samples, z_dim, device=device)
    z = F.normalize(z, dim=-1) * math.sqrt(z_dim)
    z = z.detach().requires_grad_(True)

    sensitivity = torch.zeros(z_dim, device=device)

    if source in ("F", "FB", "B", "piFB"):
        if source == "F":
            # Sensitivity of F only (B detached)
            F1, F2 = agent.forward_net(s, a, z)
            loss = F1.pow(2).sum() + F2.pow(2).sum()
        elif source in ("FB", "B"):
            # Sensitivity of M = F @ B^T w.r.t. z.
            # Note: z only enters M through F, not B. For source="B" we still
            # compute the full M sensitivity (not detaching F) because the
            # B matrix weights how F's z-dependence manifests in M. Detaching F
            # would zero out the gradient entirely since B(s) has no z path.
            F1, _ = agent.forward_net(s, a, z)
            B_val = agent.backward_net(s)
            M = F1 @ B_val.T
            loss = M.pow(2).sum()
        elif source == "piFB":
            F1, F2 = agent.forward_net(s, a, z)
            B_val = agent.backward_net(s)
            M = F1 @ B_val.T
            pi_out = agent.actor(s, z)
            loss = M.pow(2).sum() + pi_out.pow(2).sum()

        loss.backward()
        sensitivity += z.grad.abs().mean(dim=0)
        z.grad.zero_()

    if source in ("pi", "piFB"):
        if source == "pi":
            z2 = z.detach().requires_grad_(True)
            pi_out = agent.actor(s, z2)
            loss_pi = pi_out.pow(2).sum()
            loss_pi.backward()
            sensitivity += z2.grad.abs().mean(dim=0)

    # Normalize to [0, 1]
    if sensitivity.max() > 0:
        sensitivity = sensitivity / sensitivity.max()

    return sensitivity.detach()


# ---------------------------------------------------------------------------
# SMR: Sensitivity-Masked Routing
# ---------------------------------------------------------------------------

def smr_sample_z(n, z_dim, sensitivity, mask_ratio=0.5, device="cpu"):
    """Sample z in the LOW-sensitivity subspace.

    Masks out the top `mask_ratio` fraction of z-dims by sensitivity,
    samples only in the remaining dims, then normalizes to the sphere.

    Args:
        n: batch size
        z_dim: dimensionality
        sensitivity: (z_dim,) sensitivity vector
        mask_ratio: fraction of dims to mask (0.5 = mask top 50%)
        device: torch device

    Returns:
        z: (n, z_dim) on sphere of radius sqrt(z_dim)
    """
    n_mask = max(1, int(z_dim * mask_ratio))
    # Identify high-sensitivity dims to mask
    _, high_sens_idx = sensitivity.topk(n_mask)
    mask = torch.ones(z_dim, device=device)
    mask[high_sens_idx] = 0.0  # zero out high-sensitivity dims

    z = torch.randn(n, z_dim, device=device)
    z = z * mask.unsqueeze(0)  # zero out masked dims
    z = F.normalize(z, dim=-1) * math.sqrt(z_dim)
    return z


# ---------------------------------------------------------------------------
# FDWS: Fisher-weighted z-sampling
# ---------------------------------------------------------------------------

def fdws_sample_z(n, z_dim, sensitivity, temperature=1.0, device="cpu"):
    """Sample z with dims weighted inversely by sensitivity (soft routing).

    Low-sensitivity dims get higher variance (more freedom for new task),
    high-sensitivity dims get lower variance (preserve old task).

    Args:
        n: batch size
        z_dim: dimensionality
        sensitivity: (z_dim,) sensitivity vector in [0, 1]
        temperature: controls how aggressively to suppress sensitive dims
        device: torch device

    Returns:
        z: (n, z_dim) on sphere of radius sqrt(z_dim)
    """
    # Inverse sensitivity weights: low sensitivity -> high weight
    weights = torch.exp(-temperature * sensitivity)
    weights = weights / weights.mean()  # normalize so avg weight ≈ 1

    z = torch.randn(n, z_dim, device=device)
    z = z * weights.unsqueeze(0)  # scale each dim by inverse sensitivity
    z = F.normalize(z, dim=-1) * math.sqrt(z_dim)
    return z


def fdws_sample_z_high_sens(n, z_dim, sensitivity, temperature=1.0, device="cpu"):
    """Sample z biased toward HIGH-sensitivity dims (for distillation replay).

    The opposite of fdws_sample_z: high-sensitivity dims get higher variance,
    low-sensitivity dims get lower variance. This focuses the distillation
    on z-regions that old tasks actually use.

    Args:
        n: batch size
        z_dim: dimensionality
        sensitivity: (z_dim,) sensitivity vector in [0, 1]
        temperature: controls how aggressively to boost sensitive dims
        device: torch device

    Returns:
        z: (n, z_dim) on sphere of radius sqrt(z_dim)
    """
    # Positive sensitivity weights: high sensitivity -> high weight
    weights = torch.exp(+temperature * sensitivity)
    weights = weights / weights.mean()  # normalize so avg weight ≈ 1

    z = torch.randn(n, z_dim, device=device)
    z = z * weights.unsqueeze(0)  # scale each dim by sensitivity
    z = F.normalize(z, dim=-1) * math.sqrt(z_dim)
    return z
