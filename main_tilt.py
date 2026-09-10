"""TILT — minimal self-contained entry (Entropic DR + SSE, paper main method).

Reproduces the paper's main row (Table III row E) with hard-coded defaults,
no CLI required:
  1. Entropic-risk DR: each round re-weights the env distribution as
     q(theta) ∝ exp(-beta * R), with a dual-beta KL controller
     (target KL = 2.0, beta init 1.0, dual lr 0.2, ESS guard 5%).
  2. Scale-conditioned policy: log s = log(L/L0) concatenated to every
     layer input (ScaleConcatModel, swish [128, 128]).
  3. SSE (p = 0.9): during TRAINING each sample keeps its true log-s column
     only with probability 0.9 (else masked to 0); eval always sees the
     true log s. Implemented in ScaleConcatBlock via the SCALE_KEEP_P env
     var, set below BEFORE importing hover_train.

Protocol (identical to the paper): 1024 train envs, 20 rounds x 50 PPO
iterations, mid-training evals on the fixed 4096-pt Sobol grid (15 eps/pt,
seed 42, scale column zeroed), final robustness eval on the same grid
(10 eps/pt, turbulence ON, true scale column) -> logs/TILT_s{seed}/final_eval.json.

Training seed: env var TILT_SEED (default 1).
Run:  python main_tilt.py
"""
import os

os.environ.setdefault("SCALE_KEEP_P", "0.9")  # SSE scale-exposure probability

from hover_env import HoverEnv
from hover_train import run_entropic_dr_training
from hover_eval import final_robustness_eval

# --- Hard-coded protocol (paper standard) ---
SEED = int(os.environ.get("TILT_SEED", "1"))
EXP_NAME, ALGO = "TILT", f"s{SEED}"          # -> logs/TILT_s{SEED}
NUM_ENVS = 1024
NUM_EVAL_ENVS = 4096
ITERATIONS_PER_ROUND = 50
TOTAL_ROUNDS = 20
EPISODES_PER_EVAL = 15
EVAL_EPISODES = 10
EVAL_SEED = 42
EVAL_MAX_STEPS = 1500
LOG_DIR = f"logs/{EXP_NAME}_{ALGO}"


def main():
    # Skip-if-done guard: round 1 of run_entropic_dr_training wipes the log
    # dir, so the check MUST happen here (final ckpt = 20 rounds x 50 iters).
    final_ckpt = f"{LOG_DIR}/model_{TOTAL_ROUNDS * ITERATIONS_PER_ROUND}.pt"
    if os.path.exists(final_ckpt):
        print(f"[skip training] final checkpoint exists: {final_ckpt}")
    else:
        run_entropic_dr_training(
            exp_name=EXP_NAME, algo=ALGO,
            num_envs=NUM_ENVS, num_eval_envs=NUM_EVAL_ENVS,
            iterations_per_round=ITERATIONS_PER_ROUND,
            total_rounds=TOTAL_ROUNDS,
            episodes_per_eval=EPISODES_PER_EVAL,
            eval_sample_method="sobol", seed=SEED,
            label=f"TILT (entropic DR + SSE p=0.9) seed {SEED}",
            build_env=lambda n, dr_cfg: HoverEnv(
                num_envs=n, show_viewer=False, dr_cfg=dr_cfg,
                include_log_arm_scale=True),
            policy_arch="scale",
        )
    stats = final_robustness_eval(
        LOG_DIR, num_envs=NUM_EVAL_ENVS, episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS, seed=EVAL_SEED,
        include_log_arm_scale=True,
    )
    print(f"\nTILT final robustness eval (Sobol {NUM_EVAL_ENVS} pts, "
          f"seed {EVAL_SEED}, turbulence ON) -> {LOG_DIR}/final_eval.json")
    for k in ("mean", "median", "std", "min", "p10", "cvar_10", "cvar_05"):
        print(f"  {k:>8}: {stats[k]:.4f}")


if __name__ == "__main__":
    main()
