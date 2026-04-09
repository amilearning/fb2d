# Environment

The FB2D experiments were developed and run on:

| field | value |
|---|---|
| OS | Ubuntu Linux (kernel 6.17.0-19-generic) |
| GPU | NVIDIA RTX 4090 (24 GB) |
| CUDA | 13.0 |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu130 |
| numpy | 2.4.3 |
| matplotlib | 3.10.8 |

The conda environment used during development was named `incr_learn`.
The interpreter path used by the launchers is hard-coded in the helper scripts as:

```
/home/frl/anaconda3/envs/incr_learn/bin/python
```

If you reproduce on a different machine, edit `distill_sweep/_launch_smoke.py`
(and any other launcher) to point at your own Python interpreter, or simply
launch the training scripts directly with `python train_fb_distill_*.py`.

## Recreating the environment

The minimum dependency set is captured in `requirements.txt`.

```bash
conda create -n fb2d python=3.11 -y
conda activate fb2d
# pick the right torch wheel for your CUDA version, e.g. for CUDA 12.4:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

(If you use the cu130 wheel that was used here, replace `cu124` with `cu130`.)

## Why no full `pip freeze`?

The development machine's conda env also contains a ROS 2 installation
(rclpy, ament-*, ~80 packages). None of those are needed for FB2D, so
`requirements.txt` lists only the FB2D-relevant packages and their direct
transitive dependencies.

## Datasets

The FB experiments load
`checkpoints_offline_quaddyn/sub_perq.npz` and
`checkpoints_offline_quaddyn/freeroam_states_v2.npz`.

These are not included in the repo (they are produced by
`train_fb_offline_quaddyn.py` from the `Nav2DQuadDyn` environment).

## Hardware notes

- The full B×C×D distillation sweep (48 scripts × 60k updates × 3 stages) is
  GPU-bound. With waves of 8 parallel jobs, the full sweep takes ~1 h on an
  RTX 4090. Each PyTorch process needs ~500 MB of CUDA context, so 24 GB of
  VRAM cannot run all 48 in parallel — use the wave launcher in
  `distill_sweep/_launch_smoke.py` (`MAX_PARALLEL = 8`).
