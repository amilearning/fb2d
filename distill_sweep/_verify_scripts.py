#!/usr/bin/env python3
"""Programmatic checks for the 48 distill sweep scripts.

Each script is parsed and checked against its filename's (B, D, C) tuple.
We verify:
  1. The script imports the right helpers from distill_lib.
  2. The loss function name in the closure matches D.
  3. The closure uses the right target extractor for B.
  4. The closure's input source matches C (ctx vs sa_replay.sample).
  5. The right side (fb_extra_fn vs pi_extra_fn) is used and passed to joint_update.
  6. For C=replay, SAReplayBuffer is initialised and (s,a) is added per stage.
  7. The save_dir matches the (B, D, C) tuple.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

LOSS_NAME = {"l2": "loss_l2", "cosine": "loss_cosine",
             "contrastive": "loss_contrastive", "gram": "loss_gram"}


def parse_filename(fname):
    m = re.match(r"train_fb_distill_(F|B|FB|M|Q|pi)_(l2|cosine|contrastive|gram)_(current|replay)\.py$", fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


CHECKS_BY_TARGET_CURRENT = {
    "F":  ['student = (ctx["F1"], ctx["F2"])',         'get_F(teacher,'],
    "B":  ['student = ctx["B"]',                        'get_B(teacher, ctx["next_obs"])'],
    "FB": ['student = (ctx["F1"], ctx["F2"], ctx["B"])','get_F(teacher,', 'get_B(teacher, ctx["next_obs"])'],
    "M":  ['ctx["F1"] @ ctx["B"].T',                    'get_M(teacher,'],
    "Q":  ['(ctx["F1"] * ctx["z"]).sum(-1)',            'get_Q(teacher,'],
    "pi": ['student = ctx["action_pi"]',                'get_pi(teacher, ctx["obs"], ctx["z_actor"])'],
}

CHECKS_BY_TARGET_REPLAY = {
    "F":  ['get_F(agent,   s_r, a_r, z_r)',  'get_F(teacher, s_r, a_r, z_r)'],
    "B":  ['get_B(agent,   s_r)',            'get_B(teacher, s_r)'],
    "FB": ['get_F(agent,   s_r, a_r, z_r)',  'get_B(agent,   s_r)',
           'get_F(teacher, s_r, a_r, z_r)',  'get_B(teacher, s_r)'],
    "M":  ['Fs1 @ Bs.T',                     'get_M(teacher, s_r, a_r, z_r)'],
    "Q":  ['get_Q(agent,   s_r, a_r, z_r)',  'get_Q(teacher, s_r, a_r, z_r)'],
    "pi": ['get_pi(agent,   s_r, z_r)',      'get_pi(teacher, s_r, z_r)'],
}


def check_script(path, B, D, C):
    src = open(path).read()
    errs = []

    # 1. helper imports
    must_import = ["make_teacher", "sample_z", "joint_update",
                   "get_F", "get_B", "get_M", "get_Q", "get_pi",
                   LOSS_NAME[D]]
    for sym in must_import:
        if sym not in src:
            errs.append(f"missing import/use of `{sym}`")

    # 2. loss function name in the closure
    if f"{LOSS_NAME[D]}(student, target)" not in src:
        errs.append(f"closure doesn't call {LOSS_NAME[D]}(student, target)")

    # 3+4. closure substrings for B + C
    table = CHECKS_BY_TARGET_REPLAY if C == "replay" else CHECKS_BY_TARGET_CURRENT
    for snippet in table[B]:
        if snippet not in src:
            errs.append(f"closure missing snippet: `{snippet}`")

    # 5. side
    if B == "pi":
        if "def pi_extra_fn(ctx):" not in src:
            errs.append("missing `def pi_extra_fn(ctx):`")
        if "joint_update(agent, batch, pi_extra_fn=pi_extra_fn)" not in src:
            errs.append("joint_update not called with pi_extra_fn")
        if "fb_extra_fn" in src.split("def fill_buffer")[1].split("def main")[1]:
            errs.append("π script has stray fb_extra_fn references in main")
    else:
        if "def fb_extra_fn(ctx):" not in src:
            errs.append("missing `def fb_extra_fn(ctx):`")
        if "joint_update(agent, batch, fb_extra_fn=fb_extra_fn)" not in src:
            errs.append("joint_update not called with fb_extra_fn")

    # 6. replay buffer wiring
    if C == "replay":
        if "SAReplayBuffer(max_size=300_000" not in src:
            errs.append("missing SAReplayBuffer init")
        if "sa_replay.add_batch(" not in src:
            errs.append("missing sa_replay.add_batch(...) per stage")
        if "sa_replay.sample(args.batch_size)" not in src:
            errs.append("closure does not sample from sa_replay")
    else:
        if "SAReplayBuffer(" in src:
            errs.append("C=current script accidentally instantiates SAReplayBuffer")
        if "sa_replay.add_batch" in src:
            errs.append("C=current script accidentally adds to sa_replay")

    # 7. save_dir
    expected = f'default="checkpoints_distill/{B}_{D}_{C}"'
    if expected not in src:
        errs.append(f"save_dir default not set to `{expected}`")

    # 8. final-eval-only structure
    if "FINAL EVAL on Q1, Q2, Q3" not in src:
        errs.append("missing final-eval block (`FINAL EVAL on Q1, Q2, Q3`)")
    if "finals_md  = np.zeros(3)" not in src:
        errs.append("missing finals_md initialisation (1x3 shape)")
    if "finals_s10 = np.zeros(3)" not in src:
        errs.append("missing finals_s10 initialisation (1x3 shape)")
    # The per-stage eval block must NOT appear inside the stage loop.
    # Old version had `for j, eq in enumerate(QS):` AND `f = eval_in(agent, eq, seed=stage_idx`.
    if "seed=stage_idx * 100 + j" in src:
        errs.append("found old per-stage eval seed (`seed=stage_idx * 100 + j`)")

    return errs


def main():
    files = sorted(f for f in os.listdir(HERE) if f.startswith("train_fb_distill_"))
    print(f"checking {len(files)} scripts in {HERE}\n")
    fail = 0
    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            print(f"  SKIP  {f}  (filename doesn't parse)")
            continue
        B, D, C = parsed
        errs = check_script(os.path.join(HERE, f), B, D, C)
        if errs:
            fail += 1
            print(f"  FAIL  {f}")
            for e in errs:
                print(f"        - {e}")
        else:
            print(f"  ok    {f}")
    print(f"\n{len(files) - fail}/{len(files)} pass, {fail} fail")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
