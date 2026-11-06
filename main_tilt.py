"""TILT paper-protocol entry point.

The CLI retains the historical defaults while protocol values and run-path
resolution live in tilt_config.py. Set SCALE_KEEP_P before importing the
policy module because it is read when ScaleConcatBlock is constructed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SCALE_KEEP_P", "0.9")

from tilt_config import TiltProtocol
from hover_env import HoverEnv
from hover_train import run_entropic_dr_training
from hover_eval import final_robustness_eval


# Keep legacy module-level protocol names available to scripts importing them.
SEED = int(os.environ.get("TILT_SEED", "1"))
EXP_NAME, ALGO = "TILT", f"s{SEED}"
_PROTOCOL = TiltProtocol(seed=SEED).validate()
NUM_ENVS = _PROTOCOL.num_envs
NUM_EVAL_ENVS = _PROTOCOL.num_eval_envs
ITERATIONS_PER_ROUND = _PROTOCOL.iterations_per_round
TOTAL_ROUNDS = _PROTOCOL.total_rounds
EPISODES_PER_EVAL = _PROTOCOL.episodes_per_eval
EVAL_EPISODES = _PROTOCOL.final_eval_episodes
EVAL_SEED = _PROTOCOL.eval_seed
EVAL_MAX_STEPS = _PROTOCOL.eval_max_steps
LOG_DIR = str(_PROTOCOL.default_log_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate TILT.")
    parser.add_argument("--seed", type=int, default=SEED, help="Training seed.")
    parser.add_argument("--log-dir", default=None,
                        help="Run directory (default: logs/TILT_s<seed>).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate an existing final checkpoint without training.")
    parser.add_argument("--force", action="store_true",
                        help="Train even if the final checkpoint already exists.")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = TiltProtocol(seed=args.seed).validate()
    log_dir = protocol.resolve_log_dir(args.log_dir)
    final_ckpt = log_dir / f"model_{protocol.total_rounds * protocol.iterations_per_round}.pt"

    if args.eval_only:
        if not final_ckpt.is_file():
            raise FileNotFoundError(
                f"--eval-only requested but final checkpoint was not found: {final_ckpt}"
            )
        print(f"[eval-only] using checkpoint: {final_ckpt}")
    elif final_ckpt.exists() and not args.force:
        print(f"[skip training] final checkpoint exists: {final_ckpt}")
    else:
        run_entropic_dr_training(
            exp_name=EXP_NAME, algo=f"s{protocol.seed}",
            num_envs=protocol.num_envs, num_eval_envs=protocol.num_eval_envs,
            iterations_per_round=protocol.iterations_per_round,
            total_rounds=protocol.total_rounds,
            episodes_per_eval=protocol.episodes_per_eval,
            eval_sample_method="sobol", seed=protocol.seed,
            label=f"TILT (entropic DR + SSE p={os.environ['SCALE_KEEP_P']}) seed {protocol.seed}",
            build_env=lambda n, dr_cfg: HoverEnv(
                num_envs=n, show_viewer=False, dr_cfg=dr_cfg,
                include_log_arm_scale=True),
            policy_arch="scale",
        )

    stats = final_robustness_eval(
        str(log_dir), num_envs=protocol.num_eval_envs,
        episodes=protocol.final_eval_episodes,
        max_steps=protocol.eval_max_steps, seed=protocol.eval_seed,
        include_log_arm_scale=True,
    )
    print(f"\nTILT final robustness eval (Sobol {protocol.num_eval_envs} pts, "
          f"seed {protocol.eval_seed}, turbulence ON) -> {log_dir}/final_eval.json")
    for key in ("mean", "median", "std", "min", "p10", "cvar_10", "cvar_05"):
        print(f"  {key:>8}: {stats[key]:.4f}")


if __name__ == "__main__":
    main()
