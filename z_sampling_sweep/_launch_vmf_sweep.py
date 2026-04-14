#!/usr/bin/env python3
"""Launch vMF component-count sweep: 9 configs × 3 seeds = 27 runs."""
import os, subprocess, time

HERE       = os.path.dirname(os.path.abspath(__file__))
FB2D       = os.path.dirname(HERE)
LOG_DIR    = os.path.join(HERE, "vmf_sweep_logs")
SAVE_ROOT  = os.path.join(FB2D, "checkpoints_vmf_sweep_real")
RUNNER_LOG = os.path.join(HERE, "vmf_sweep_runner.log")
PY         = "/home/frl/anaconda3/envs/incr_learn/bin/python"
MAX_PARALLEL = 8
SEEDS = [1, 2, 3]
WANDB_PROJECT = "fb2d-distill"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)

scripts = sorted(f for f in os.listdir(HERE) if f.startswith("train_z_vmfMix") and f.endswith(".py"))
assert len(scripts) == 9, f"expected 9, got {len(scripts)}"

jobs = [(s, seed) for s in scripts for seed in SEEDS]
total = len(jobs)
env = os.environ.copy()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUNNER_LOG, "a") as f:
        f.write(line + "\n")

log(f"starting vMF sweep: {total} jobs, MAX_PARALLEL={MAX_PARALLEL}")

procs = {}
pid_to_proc = {}
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

job_iter = iter(jobs)
exhausted = False
while True:
    while len(procs) < MAX_PARALLEL and not exhausted:
        try:
            fname, seed = next(job_iter)
        except StopIteration:
            exhausted = True
            break
        name = fname.replace("train_z_", "").replace(".py", "")
        run_name = f"{name}_seed{seed}"
        save_dir = os.path.join(SAVE_ROOT, name)
        log_path = os.path.join(LOG_DIR, f"{run_name}.log")
        cmd = [
            PY, "-u", os.path.join(HERE, fname),
            "--updates_per_stage", "60000",
            "--seed", str(seed),
            "--save_dir", save_dir,
            "--wandb_project", WANDB_PROJECT,
            "--wandb_run_name", run_name,
        ]
        logf = open(log_path, "wb")
        p = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            cwd=FB2D, env=env,
        )
        procs[p.pid] = (run_name, time.time())
        pid_to_proc[p.pid] = p
        log(f"  launched {run_name}  (pid {p.pid})  [{len(procs)}/{MAX_PARALLEL} alive]")

    if exhausted and not procs:
        break
    time.sleep(5)
    reap_finished()

log(f"all {total} jobs finished")
