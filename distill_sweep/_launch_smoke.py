#!/usr/bin/env python3
"""Launch all 48 distill scripts as a 5-step-per-stage smoke test.

Runs in waves of `MAX_PARALLEL` scripts to avoid GPU OOM (each PyTorch process
incurs ~500 MB CUDA context overhead, so 48 in parallel exhausts a 24 GB GPU).
"""
import os, subprocess, time

HERE      = os.path.dirname(os.path.abspath(__file__))
FB2D      = os.path.dirname(HERE)
LOG_DIR   = os.path.join(HERE, "smoke_logs")
SAVE_ROOT = os.path.join(FB2D, "checkpoints_distill_smoke")
PY        = "/home/frl/anaconda3/envs/incr_learn/bin/python"
MAX_PARALLEL = 8

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)

scripts = sorted(f for f in os.listdir(HERE) if f.startswith("train_fb_distill_") and f.endswith(".py"))
assert len(scripts) == 48, f"expected 48 scripts, got {len(scripts)}"

print(f"smoke-launching {len(scripts)} scripts in waves of {MAX_PARALLEL}")

procs = []
def wait_until_under(limit):
    while True:
        alive = [p for p in procs if p.poll() is None]
        procs[:] = alive
        if len(alive) < limit:
            return
        time.sleep(2)

for fname in scripts:
    wait_until_under(MAX_PARALLEL)
    name = fname.replace("train_fb_distill_", "").replace(".py", "")
    save_dir = os.path.join(SAVE_ROOT, name)
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    cmd = [
        PY, "-u", os.path.join(HERE, fname),
        "--updates_per_stage", "5",
        "--seed", "0",
        "--save_dir", save_dir,
    ]
    logf = open(log_path, "wb")
    p = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=FB2D,
    )
    procs.append(p)
    print(f"  launched {name} (pid {p.pid})")

# wait for the final wave
print("waiting for last wave to finish...")
for p in procs:
    p.wait()
print("all 48 done")
