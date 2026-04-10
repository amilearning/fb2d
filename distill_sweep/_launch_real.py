#!/usr/bin/env python3
"""Launch the full B×C×D distillation sweep: 48 configs × 3 seeds = 144 runs.

Each subprocess gets:
  --updates_per_stage 60000     (the real schedule)
  --seed {1,2,3}
  --save_dir checkpoints_distill_real/{name}
  --wandb_project fb2d-distill
  --wandb_run_name {name}_seed{seed}

Runs in waves of MAX_PARALLEL = 8 to stay under GPU memory.

Writes completion log to ./real_runner.log; per-script logs in ./real_logs/.
"""
import os, subprocess, time

HERE      = os.path.dirname(os.path.abspath(__file__))
FB2D      = os.path.dirname(HERE)
LOG_DIR   = os.path.join(HERE, "real_logs")
SAVE_ROOT = os.path.join(FB2D, "checkpoints_distill_real")
RUNNER_LOG = os.path.join(HERE, "real_runner.log")
PY        = "/home/frl/anaconda3/envs/incr_learn/bin/python"
MAX_PARALLEL = 8
SEEDS = [1, 2, 3]
UPDATES_PER_STAGE = 60_000
WANDB_PROJECT = "fb2d-distill"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)

scripts = sorted(f for f in os.listdir(HERE) if f.startswith("train_fb_distill_") and f.endswith(".py"))
assert len(scripts) == 48, f"expected 48 scripts, got {len(scripts)}"

# Build the full job list (script, seed)
jobs = [(s, seed) for s in scripts for seed in SEEDS]
total = len(jobs)

# Inherit env so wandb credentials in ~/.netrc / WANDB_API_KEY work
env = os.environ.copy()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUNNER_LOG, "a") as f:
        f.write(line + "\n")

log(f"starting sweep: {total} jobs, MAX_PARALLEL={MAX_PARALLEL}")

procs = {}  # pid -> (name, start_time)
done = 0

def reap_finished():
    global done
    finished = []
    for pid, (name, t0) in procs.items():
        p = pid_to_proc[pid]
        if p.poll() is not None:
            finished.append(pid)
            done += 1
            elapsed = time.time() - t0
            status = "ok" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
            log(f"  finished {done}/{total}  {name}  ({elapsed:.0f}s)  {status}")
    for pid in finished:
        del procs[pid]
        del pid_to_proc[pid]

pid_to_proc = {}

job_iter = iter(jobs)
exhausted = False
while True:
    # spawn until we have MAX_PARALLEL alive (or job iterator is exhausted)
    while len(procs) < MAX_PARALLEL and not exhausted:
        try:
            fname, seed = next(job_iter)
        except StopIteration:
            exhausted = True
            break
        name = fname.replace("train_fb_distill_", "").replace(".py", "")
        run_name = f"{name}_seed{seed}"
        save_dir = os.path.join(SAVE_ROOT, name)
        log_path = os.path.join(LOG_DIR, f"{run_name}.log")
        cmd = [
            PY, "-u", os.path.join(HERE, fname),
            "--updates_per_stage", str(UPDATES_PER_STAGE),
            "--seed", str(seed),
            "--save_dir", save_dir,
            "--wandb_project", WANDB_PROJECT,
            "--wandb_run_name", run_name,
        ]
        logf = open(log_path, "wb")
        p = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=FB2D,
            env=env,
        )
        procs[p.pid] = (run_name, time.time())
        pid_to_proc[p.pid] = p
        log(f"  launched {run_name}  (pid {p.pid})  [{len(procs)}/{MAX_PARALLEL} alive]")

    if exhausted and not procs:
        break

    time.sleep(5)
    reap_finished()

log(f"all {total} jobs finished")
