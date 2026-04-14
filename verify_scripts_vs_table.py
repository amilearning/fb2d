#!/usr/bin/env python3
"""Verify that all 113 training scripts match their method-table descriptions.

Re-generates the config entries (same logic as generate_full_table_pdf.py),
then reads each actual training script and checks z-sampling, distill target,
distill input, loss functions, K (vMF), tau (FDWS), and update method.
"""

import os, re, sys

# ============================================================================
# 1. Re-generate the 113 config entries (mirror generate_full_table_pdf.py)
# ============================================================================
all_records = []

# --- Baselines (M1, M2) ---
all_records.append({
    "sweep": "baseline", "config": "naive_seq",
    "label": "Naive Sequential",
    "z_sampling": "random", "distill_target": "---",
    "distill_input": "---", "K": "---", "ema": "---", "tau": "---", "alpha": "---",
})
all_records.append({
    "sweep": "baseline", "config": "cumulative",
    "label": "Cumulative (task-incr.)",
    "z_sampling": "random", "distill_target": "---",
    "distill_input": "---", "K": "---", "ema": "---", "tau": "---", "alpha": "---",
})

# --- B x D x C Standalone (48 configs, M3-M50) ---
DISTILL_ROOT = "/home/frl/FB2D/checkpoints_distill_real"
for config_dir in sorted(os.listdir(DISTILL_ROOT)):
    config_path = os.path.join(DISTILL_ROOT, config_dir)
    if not os.path.isdir(config_path):
        continue
    parts = config_dir.split("_")
    C = parts[-1]
    D = parts[-2]
    B = "_".join(parts[:-2])
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

# --- Combined pi+X (24 configs, M51-M74) ---
COMBINED_ROOT = "/home/frl/FB2D/checkpoints_distill_combined_real"
for config_dir in sorted(os.listdir(COMBINED_ROOT)):
    config_path = os.path.join(COMBINED_ROOT, config_dir)
    if not os.path.isdir(config_path):
        continue
    parts = config_dir.split("_")
    C = parts[-1]
    pi_type = parts[0]
    X = "_".join(parts[1:-1])
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

# --- Z-sampling (24 configs, M75-M98) ---
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

Z_ROOT = "/home/frl/FB2D/checkpoints_z_sampling_real"
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

# --- vMF K sweep (9 configs, M99-M107) ---
VMF_ROOT = "/home/frl/FB2D/checkpoints_vmf_sweep_real"
for config_dir in sorted(os.listdir(VMF_ROOT)):
    config_path = os.path.join(VMF_ROOT, config_dir)
    if not os.path.isdir(config_path):
        continue
    parts = config_dir.split("_")
    K_str = parts[0].replace("vmfMix", "")
    distill_label = parts[1]
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

# --- FDWS tau sweep (6 configs, M108-M113) ---
FDWS_ROOT = "/home/frl/FB2D/checkpoints_fdws_sweep_real"
TAU_MAP = {"tau05": "0.5", "tau10": "1.0", "tau50": "5.0"}
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
        "z_sampling": "FDWS(FB)",
        "distill_target": "pi(gram)+FB(l2)",
        "distill_input": sens_src,
        "K": "---", "ema": "---", "tau": tau_val, "alpha": "1.0",
    })

# Assign method IDs
for i, r in enumerate(all_records):
    r["method_id"] = f"M{i+1}"

print(f"Collected {len(all_records)} config entries.\n")

# ============================================================================
# 2. Locate script for each method
# ============================================================================
def find_script(rec):
    """Return the absolute path to the training script for this record."""
    sweep = rec["sweep"]
    config = rec["config"]

    if sweep == "baseline":
        if config == "naive_seq":
            return "/home/frl/FB2D/train_fb_naive_seq_q123.py"
        elif config == "cumulative":
            return "/home/frl/FB2D/train_fb_taskincr_q123.py"

    elif sweep == "BxDxC":
        # config = e.g. "B_l2_current" -> script = train_fb_distill_B_l2_current.py
        return f"/home/frl/FB2D/distill_sweep/train_fb_distill_{config}.py"

    elif sweep == "combined":
        # config = e.g. "piGram_FB_replay" -> script = train_fb_combined_piGram_FB_replay.py
        return f"/home/frl/FB2D/distill_sweep_combined/train_fb_combined_{config}.py"

    elif sweep == "z_sampling":
        # config = e.g. "A1_vmf_bind" -> script = train_z_A1_vmf_bind.py
        return f"/home/frl/FB2D/z_sampling_sweep/train_z_{config}.py"

    elif sweep == "vmf_sweep":
        # config = e.g. "vmfMix5_FBpi_replay" -> script = train_z_vmfMix5_FBpi_replay.py
        return f"/home/frl/FB2D/z_sampling_sweep/train_z_{config}.py"

    elif sweep == "fdws_sweep":
        # config = e.g. "tau05_sensCurrent" -> script = train_fdws_tau05_sensCurrent.py
        return f"/home/frl/FB2D/fdws_sweep/train_fdws_{config}.py"

    return None


# ============================================================================
# 3. Verification checks
# ============================================================================
def check_z_sampling(source, expected_z):
    """Check z-sampling method in script source against expected table value."""
    if expected_z == "random":
        # Baselines/distill: no VMFTaskSampler, no smr_sample_z, no fdws_sample_z
        has_vmf = "VMFTaskSampler" in source
        has_smr = "smr_sample_z" in source
        has_fdws = "fdws_sample_z" in source
        if has_vmf or has_smr or has_fdws:
            return False, f"expected random z but found specialized sampler"
        return True, "OK"

    elif expected_z.startswith("vMF-bind"):
        if "VMFTaskSampler" not in source:
            return False, "expected VMFTaskSampler but not found"
        # Should use vmf_sampler.sample(n, stage) not sample_mixture
        if "sample_mixture" in source:
            return False, "expected task-binding (sample) but found sample_mixture"
        return True, "OK"

    elif expected_z.startswith("vMF-mix"):
        if "VMFTaskSampler" not in source:
            return False, "expected VMFTaskSampler but not found"
        if "sample_mixture" not in source:
            return False, "expected sample_mixture but not found"
        return True, "OK"

    elif expected_z.startswith("SMR"):
        if "smr_sample_z" not in source:
            return False, "expected smr_sample_z but not found"
        # Extract sensitivity source from expected_z: SMR(F), SMR(B), etc.
        m = re.search(r'SMR\((\w+)\)', expected_z)
        if m:
            exp_src = m.group(1)
            # Check compute_z_sensitivity(..., source="X", ...)
            sens_match = re.search(r'compute_z_sensitivity\([^)]*source\s*=\s*"(\w+)"', source)
            if sens_match:
                actual_src = sens_match.group(1)
                if actual_src != exp_src:
                    return False, f"SMR source mismatch: expected {exp_src}, got {actual_src}"
        return True, "OK"

    elif expected_z.startswith("FDWS"):
        if "fdws_sample_z" not in source:
            return False, "expected fdws_sample_z but not found"
        m = re.search(r'FDWS\((\w+)\)', expected_z)
        if m:
            exp_src = m.group(1)
            sens_match = re.search(r'compute_z_sensitivity\([^)]*source\s*=\s*"(\w+)"', source)
            if sens_match:
                actual_src = sens_match.group(1)
                if actual_src != exp_src:
                    return False, f"FDWS source mismatch: expected {exp_src}, got {actual_src}"
        return True, "OK"

    return True, "OK (unchecked)"


def check_distill_target(source, expected_target, sweep):
    """Check which get_X calls are in the distillation closures."""
    if expected_target == "---":
        # No distillation expected
        # Baselines: no fb_extra_fn or pi_extra_fn
        if sweep == "baseline":
            if "fb_extra_fn" in source or "pi_extra_fn" in source:
                return False, "expected no distillation but found extra_fn"
            return True, "OK"
        # z-sampling A/C groups: no distillation, but joint_update still used
        if "fb_extra_fn" in source or "pi_extra_fn" in source:
            return False, "expected no distillation but found extra_fn"
        return True, "OK"

    issues = []

    # Parse expected target: e.g. "B(l2)", "FB(gram)", "pi(gram)+FB(l2)", "pi(l2)+FBM(l2)"
    # Split on '+' to get individual targets
    target_parts = expected_target.split("+")

    for tp in target_parts:
        m = re.match(r'(\w+)\((\w+)\)', tp)
        if not m:
            issues.append(f"cannot parse target part: {tp}")
            continue
        what = m.group(1)  # e.g. pi, B, F, FB, M, Q, FBM
        loss_name = m.group(2)  # e.g. l2, gram, cosine, contrastive

        # Check the right get_X is called
        if what == "pi":
            if "get_pi" not in source:
                issues.append(f"expected get_pi for pi distill but not found")
            # Should use pi_extra_fn for standalone pi, or pi_extra_fn in combined
            if sweep == "BxDxC":
                if "pi_extra_fn" not in source:
                    issues.append("expected pi_extra_fn for pi distill")
            # Check loss
            expected_loss = f"loss_{loss_name}"
            # In pi_extra_fn closure, check the loss
            pi_fn_match = re.search(r'def pi_extra_fn.*?(?=def \w|$)', source, re.DOTALL)
            if pi_fn_match:
                fn_body = pi_fn_match.group(0)
                if expected_loss not in fn_body:
                    issues.append(f"pi closure: expected {expected_loss} but not found")

        elif what in ("F", "B", "FB", "M", "Q", "FBM"):
            # These go into fb_extra_fn
            if what == "F":
                if "get_F" not in source:
                    issues.append("expected get_F but not found in fb_extra_fn")
            elif what == "B":
                if "get_B" not in source:
                    issues.append("expected get_B but not found")
            elif what == "FB":
                if "get_F" not in source or "get_B" not in source:
                    issues.append("expected get_F+get_B for FB distill but missing")
            elif what == "FBM":
                if "get_F" not in source or "get_B" not in source or "get_M" not in source:
                    issues.append("expected get_F+get_B+get_M for FBM distill but missing")
            elif what == "M":
                if "get_M" not in source:
                    issues.append("expected get_M but not found")
            elif what == "Q":
                if "get_Q" not in source:
                    issues.append("expected get_Q but not found")

            # Check loss in fb_extra_fn
            if sweep == "BxDxC":
                fb_fn_match = re.search(r'def fb_extra_fn.*?(?=def \w|$)', source, re.DOTALL)
                if fb_fn_match:
                    fn_body = fb_fn_match.group(0)
                    expected_loss = f"loss_{loss_name}"
                    if expected_loss not in fn_body:
                        issues.append(f"fb closure: expected {expected_loss} but not found")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def check_distill_input(source, expected_input, sweep):
    """Check whether distillation uses current context or replay buffer."""
    if expected_input == "---":
        return True, "OK"

    if expected_input == "current":
        # fb_extra_fn should use ctx["..."] not sa_replay.sample
        # Check that no SAReplayBuffer is used for distillation input
        if "SAReplayBuffer" in source and "sa_replay.sample" in source:
            # For some pi-only configs with "current", they use ctx directly
            # Check if fb_extra_fn uses ctx
            fb_fn_match = re.search(r'def fb_extra_fn.*?(?=\ndef \w|\nmetrics)', source, re.DOTALL)
            if fb_fn_match and "sa_replay.sample" in fb_fn_match.group(0):
                return False, "expected current input but fb_extra_fn uses sa_replay"
            pi_fn_match = re.search(r'def pi_extra_fn.*?(?=\ndef \w|\nmetrics)', source, re.DOTALL)
            if pi_fn_match and "sa_replay.sample" in pi_fn_match.group(0):
                return False, "expected current input but pi_extra_fn uses sa_replay"
        return True, "OK"

    elif expected_input == "replay":
        if "sa_replay" not in source and "SAReplayBuffer" not in source:
            return False, "expected replay input but no SAReplayBuffer found"
        return True, "OK"

    return True, "OK"


def check_update_method(source, expected_target, sweep):
    """Scripts with distillation should use joint_update; baselines without should use agent.update."""
    if sweep == "baseline":
        if "joint_update" in source:
            return False, "baseline should use agent.update not joint_update"
        if "agent.update" not in source:
            return False, "expected agent.update in baseline"
        return True, "OK"

    # All other sweeps use joint_update
    if "joint_update" not in source:
        return False, "expected joint_update but not found"
    return True, "OK"


def check_vmf_k(source, expected_k):
    """For vMF methods, check the K (n_tasks) parameter."""
    if expected_k == "---":
        return True, "OK"

    k_int = int(expected_k)
    m = re.search(r'VMFTaskSampler\s*\(\s*n_tasks\s*=\s*(\d+)', source)
    if m:
        actual_k = int(m.group(1))
        if actual_k != k_int:
            return False, f"expected K={k_int} but found n_tasks={actual_k}"
        return True, "OK"

    # For z_sampling group, K=3 always
    if "VMFTaskSampler" in source:
        # Try alternate pattern
        m2 = re.search(r'n_tasks\s*=\s*(\d+)', source)
        if m2:
            actual_k = int(m2.group(1))
            if actual_k != k_int:
                return False, f"expected K={k_int} but found n_tasks={actual_k}"
            return True, "OK"
    return True, "OK"


def check_fdws_tau(source, expected_tau):
    """For FDWS sweep, check the temperature value."""
    if expected_tau == "---":
        return True, "OK"

    tau_float = float(expected_tau)
    # Look for FDWS_TAU = X.X
    m = re.search(r'FDWS_TAU\s*=\s*([\d.]+)', source)
    if m:
        actual_tau = float(m.group(1))
        if abs(actual_tau - tau_float) > 0.01:
            return False, f"expected tau={tau_float} but found FDWS_TAU={actual_tau}"
        return True, "OK"
    # Also check fdws_temperature default
    m2 = re.search(r'fdws_temperature.*?default\s*=\s*([\d.]+)', source)
    if m2:
        actual_tau = float(m2.group(1))
        if abs(actual_tau - tau_float) > 0.01:
            return False, f"expected tau={tau_float} but found default={actual_tau}"
        return True, "OK"
    return True, "OK"


def check_ema(source, expected_ema):
    """For z-sampling methods, check EMA blending coefficient."""
    if expected_ema == "---":
        return True, "OK"

    ema_float = float(expected_ema)
    # Look for: sensitivity = X * sensitivity + X * new_sens
    m = re.search(r'sensitivity\s*=\s*([\d.]+)\s*\*\s*sensitivity\s*\+\s*([\d.]+)\s*\*\s*new_sens', source)
    if m:
        old_w = float(m.group(1))
        new_w = float(m.group(2))
        # EMA decay = old_w, so ema = old_w
        actual_ema = old_w
        if abs(actual_ema - ema_float) > 0.01:
            return False, f"expected EMA={ema_float} but found {actual_ema}"
        return True, "OK"

    # FDWS sweep: EMA=0.0 means no accumulation (sensitivity = new_sens directly)
    # Check if it replaces instead of blending
    if "0.5 * sensitivity + 0.5 * new_sens" in source:
        if abs(ema_float - 0.5) > 0.01:
            return False, f"expected EMA={ema_float} but found 0.5 blending"
    return True, "OK"


# ============================================================================
# 4. Run verification
# ============================================================================
results = []
pass_count = 0
fail_count = 0
missing_count = 0

for rec in all_records:
    mid = rec["method_id"]
    sweep = rec["sweep"]
    config = rec["config"]

    script_path = find_script(rec)
    if script_path is None:
        results.append((mid, config, "MISSING", "could not determine script path"))
        missing_count += 1
        continue

    if not os.path.exists(script_path):
        results.append((mid, config, "MISSING", f"script not found: {script_path}"))
        missing_count += 1
        continue

    with open(script_path, "r") as f:
        source = f.read()

    issues = []

    # Check z-sampling
    ok, msg = check_z_sampling(source, rec["z_sampling"])
    if not ok:
        issues.append(f"z-sampling: {msg}")

    # Check distill target
    ok, msg = check_distill_target(source, rec["distill_target"], sweep)
    if not ok:
        issues.append(f"distill-target: {msg}")

    # Check distill input
    ok, msg = check_distill_input(source, rec["distill_input"], sweep)
    if not ok:
        issues.append(f"distill-input: {msg}")

    # Check update method
    ok, msg = check_update_method(source, rec["distill_target"], sweep)
    if not ok:
        issues.append(f"update-method: {msg}")

    # Check vMF K
    ok, msg = check_vmf_k(source, rec["K"])
    if not ok:
        issues.append(f"vmf-K: {msg}")

    # Check FDWS tau
    ok, msg = check_fdws_tau(source, rec["tau"])
    if not ok:
        issues.append(f"fdws-tau: {msg}")

    # Check EMA (for z_sampling sweep)
    if sweep in ("z_sampling",):
        ok, msg = check_ema(source, rec["ema"])
        if not ok:
            issues.append(f"ema: {msg}")

    if issues:
        results.append((mid, config, "MISMATCH", "; ".join(issues)))
        fail_count += 1
    else:
        results.append((mid, config, "PASS", ""))
        pass_count += 1

# ============================================================================
# 5. Print results
# ============================================================================
print("=" * 100)
print(f"{'ID':>5}  {'Config':<40}  {'Status':<10}  Details")
print("=" * 100)

for mid, config, status, detail in results:
    if status == "PASS":
        print(f"{mid:>5}  {config:<40}  \033[92m{status:<10}\033[0m")
    elif status == "MISMATCH":
        print(f"{mid:>5}  {config:<40}  \033[91m{status:<10}\033[0m  {detail}")
    elif status == "MISSING":
        print(f"{mid:>5}  {config:<40}  \033[93m{status:<10}\033[0m  {detail}")

print("=" * 100)
print(f"\nSUMMARY: {pass_count}/{len(all_records)} PASS, "
      f"{fail_count} MISMATCH, {missing_count} MISSING")

if fail_count > 0:
    print(f"\n--- MISMATCHES ---")
    for mid, config, status, detail in results:
        if status == "MISMATCH":
            print(f"  {mid}: {config}")
            print(f"         {detail}")

if missing_count > 0:
    print(f"\n--- MISSING ---")
    for mid, config, status, detail in results:
        if status == "MISSING":
            print(f"  {mid}: {config} -- {detail}")

sys.exit(1 if fail_count > 0 or missing_count > 0 else 0)
