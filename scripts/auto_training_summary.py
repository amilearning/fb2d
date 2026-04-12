#!/usr/bin/env python3
"""Auto-generate a daily training progress summary.

Scans checkpoint dirs for completed runs, reads mean_d.npy / s10.npy,
and writes a dated markdown summary to the memory folder.

Usage:
  python auto_training_summary.py                    # today's date
  python auto_training_summary.py --date 20260412    # specific date

Can be run as a cron job:
  0 23 * * * /home/frl/anaconda3/envs/incr_learn/bin/python /home/frl/FB2D/scripts/auto_training_summary.py
"""
import argparse, glob, os, time
from datetime import datetime
import numpy as np

MEMORY_DIR = "/home/frl/.claude/projects/-home-frl-ContFB/memory"
FB2D = "/home/frl/FB2D"

SWEEP_DIRS = {
    "B×C×D sweep (standalone)": os.path.join(FB2D, "checkpoints_distill_real"),
    "Combined π+X sweep": os.path.join(FB2D, "checkpoints_distill_combined_real"),
}


def scan_results(root):
    """Scan a sweep root dir and return a dict of config -> list of (seed, mean_d, s10)."""
    results = {}
    if not os.path.isdir(root):
        return results
    for config_dir in sorted(os.listdir(root)):
        config_path = os.path.join(root, config_dir)
        if not os.path.isdir(config_path):
            continue
        seeds = []
        for seed_dir in sorted(glob.glob(os.path.join(config_path, "seed*"))):
            md_path = os.path.join(seed_dir, "mean_d.npy")
            s10_path = os.path.join(seed_dir, "s10.npy")
            if os.path.exists(md_path) and os.path.exists(s10_path):
                md = np.load(md_path).flatten()
                s10 = np.load(s10_path).flatten()
                seeds.append({"mean_d": md, "s10": s10, "dir": seed_dir})
        if seeds:
            results[config_dir] = seeds
    return results


def summarize_sweep(name, results):
    if not results:
        return f"### {name}\nNo completed runs found.\n"

    lines = [f"### {name}", f"Configs completed: {len(results)}", ""]
    lines.append("| config | seeds | mean_d_avg | s10_avg | s10_Q1 (memory) | s10_Q3 (plasticity) |")
    lines.append("|---|---|---|---|---|---|")

    rows = []
    for config, seeds in sorted(results.items()):
        n = len(seeds)
        md_all = np.stack([s["mean_d"] for s in seeds])
        s10_all = np.stack([s["s10"] for s in seeds])
        md_avg = md_all.mean(axis=1).mean()
        s10_avg = s10_all.mean(axis=1).mean()
        s10_q1 = s10_all[:, 0].mean() if s10_all.shape[1] >= 1 else 0
        s10_q3 = s10_all[:, 2].mean() if s10_all.shape[1] >= 3 else 0
        rows.append((md_avg, config, n, md_avg, s10_avg, s10_q1, s10_q3))

    rows.sort(key=lambda x: x[0])
    for _, config, n, md_avg, s10_avg, s10_q1, s10_q3 in rows:
        lines.append(f"| {config} | {n} | {md_avg:.3f} | {s10_avg:.3f} | {s10_q1:.3f} | {s10_q3:.3f} |")

    lines.append("")
    best = rows[0]
    lines.append(f"**Best:** {best[1]} (mean_d={best[3]:.3f}, s10_avg={best[4]:.3f})")
    lines.append("")
    return "\n".join(lines)


def check_runner_logs():
    """Check if any sweep runners are still active."""
    logs = []
    for name, path in [
        ("B×C×D", os.path.join(FB2D, "distill_sweep/real_runner.log")),
        ("Combined", os.path.join(FB2D, "distill_sweep_combined/real_runner.log")),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            if "all" in content.split("\n")[-1].lower() and "finished" in content.split("\n")[-1].lower():
                logs.append(f"- {name}: **completed**")
            else:
                last = content.strip().split("\n")[-1] if content.strip() else "empty"
                logs.append(f"- {name}: last log = `{last[:80]}`")
    return "\n".join(logs) if logs else "No runner logs found."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYYMMDD, default=today")
    args = p.parse_args()

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    fname = f"{date_str}_ResearchBot_memory.md"
    fpath = os.path.join(MEMORY_DIR, fname)

    lines = [
        "---",
        f"name: Auto Training Summary {date_str}",
        f"description: Automated daily summary of training progress and results",
        "type: project",
        "---",
        "",
        f"## Auto-generated Training Summary — {date_str}",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### Runner Status",
        check_runner_logs(),
        "",
    ]

    for name, root in SWEEP_DIRS.items():
        results = scan_results(root)
        lines.append(summarize_sweep(name, results))

    content = "\n".join(lines)

    # Append if file exists, otherwise create
    mode = "a" if os.path.exists(fpath) else "w"
    with open(fpath, mode) as f:
        if mode == "a":
            f.write("\n\n---\n\n")
        f.write(content)

    print(f"{'Appended to' if mode == 'a' else 'Created'}: {fpath}")
    print(content[:500] + "...")


if __name__ == "__main__":
    main()
