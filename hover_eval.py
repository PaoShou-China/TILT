"""
Robustness evaluation script
============================
1. Default: batch robustness test over a low-discrepancy Sobol grid of
   Scale-Conditioned DR parameters (no viewer).
2. --visualize: load trained policy and observe flight in the viewer.

Usage:
    python hover_eval.py --log-dir logs/drone-hovering_baseline \
                          --num-eval-envs 256
    python hover_eval.py --log-dir logs/drone-hovering_baseline --visualize
"""

import os, sys, time, copy, argparse, json

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

import torch
import numpy as np

_self_dir = os.path.dirname(os.path.abspath(__file__))
if _self_dir not in sys.path:
    sys.path.insert(0, _self_dir)

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from hover_env import HoverEnv
from hover_train import (
    BASE_DR,
    load_policy,
    evaluate_on_param_grid,
    _generate_sobol_dr_params,
    _inject_sobol_into_env,
)


class EvalEnv(HoverEnv):
    """Evaluation env with turbulence kept ON.

    The final robustness eval must match the training distribution: DR
    params from the Sobol grid AND the same turbulence (strength 0.08).
    Turbulence realization is identical across methods because every eval
    process gs.init(seed=42) and the eval loop consumes no global RNG.
    """

    def _build_scene(self, show_viewer):
        super()._build_scene(show_viewer, add_turbulence=True)


def run_sobol_test(log_dir, ckpt, num_eval_envs, episodes_per_eval,
                   max_steps=None, seed=None, sobol_params=None,
                   include_log_arm_scale=False):
    """Batch robustness test on a Sobol DR grid; reports aggregate & tail stats.

    seed: RNG seed. Fixes BOTH the Sobol grid (if sobol_params is None) and a
        deterministic lookup table for initial states & goal positions, so
        runs with the same seed see identical DR params, init states and
        goals (fair comparison).
    sobol_params: pre-generated grid. If given, overrides seed for the grid
        only; init states / goals still follow the deterministic table.
    """
    print(f"\n  [Sobol Test] Evaluating {log_dir} over {num_eval_envs} Sobol DR points ...", flush=True)
    result = load_policy(log_dir, ckpt)
    if result is None:
        print(f"Error: Cannot load policy from {log_dir}")
        sys.exit(1)
    train_cfg, model_path, used_ckpt = result
    print(f"  Policy loaded (checkpoint {used_ckpt})", flush=True)

    env = EvalEnv(num_envs=num_eval_envs, show_viewer=False,
                  dr_cfg={**BASE_DR, "sample_method": "sobol"},
                  randomize_init_state=True,
                  deterministic_eval_seed=seed,
                  include_log_arm_scale=include_log_arm_scale)

    if max_steps is not None:
        env.max_episode_length = min(int(max_steps), env.max_episode_length)

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.load(model_path, map_location="cpu")
    policy = runner.get_inference_policy(device=gs.device)

    if sobol_params is None:
        sobol_params = _generate_sobol_dr_params(num_eval_envs, seed=seed)
    if sobol_params:
        _inject_sobol_into_env(env, sobol_params)
        print(f"  [Sobol Test] Injected {num_eval_envs} low-discrepancy DR params", flush=True)
    else:
        print("  [Sobol Test] WARNING: scipy unavailable, falling back to uniform DR", flush=True)

    print(f"  Running {episodes_per_eval} episodes per param ...", flush=True)
    eval_rewards, _ = evaluate_on_param_grid(env, policy, episodes_per_eval)

    r = np.asarray(eval_rewards, dtype=np.float64)
    stats = {
        "mode": "RL only",
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "max": float(r.max()),
        "median": float(np.median(r)),
        "p10": float(np.percentile(r, 10)),
        "p25": float(np.percentile(r, 25)),
        "p75": float(np.percentile(r, 75)),
        "p90": float(np.percentile(r, 90)),
        "p95": float(np.percentile(r, 95)),
        "p99": float(np.percentile(r, 99)),
        "cvar_05": float(r[r <= np.percentile(r, 5)].mean()),
        "cvar_10": float(r[r <= np.percentile(r, 10)].mean()),
        "cvar_25": float(r[r <= np.percentile(r, 25)].mean()),
        "n": int(len(r)),
    }

    print(f"\n  Sobol grid robustness stats ({num_eval_envs} params x {episodes_per_eval} eps):")
    print(f"  {'='*50}")
    print(f"  {'Metric':<12} {'Value':>12}")
    print(f"  {'-'*24}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<12} {v:>12.4f}")
        else:
            print(f"  {k:<12} {v:>12}")
    print(f"  {'='*50}")

    del env, runner
    return stats


PROGRESS_FILE = "run_progress.log"


def final_robustness_eval(log_dir, num_envs=10000, episodes=10,
                          max_steps=1500, seed=42,
                          include_log_arm_scale=False):
    """Final fixed-grid robustness eval, callable right after training.

    Deterministic & fair A/B: seed fixes BOTH the Sobol DR grid and the
    init-state / goal lookup tables, so any two methods with the same seed
    see identical DR params, initial states and goal positions. Stats are
    also saved to {log_dir}/final_eval.json (skipped if already present).

    include_log_arm_scale: feed the true log(L/L0) scale column — required
        by scale-conditioned policies (D-SCALE / SSE); must match training.
    """
    import genesis as gs
    out_path = f"{log_dir}/final_eval.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            stats = json.load(f)
        print(f"[skip-eval] {out_path} already exists")
        return stats

    if not gs._initialized:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning",
                seed=seed, performance_mode=True)
    torch.set_float32_matmul_precision("high")

    stats = run_sobol_test(log_dir, None, num_envs, episodes,
                           max_steps=max_steps, seed=seed,
                           include_log_arm_scale=include_log_arm_scale)
    append_progress(log_dir, seed, stats)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Final eval stats saved -> {out_path}")
    return stats


def append_progress(log_dir, seed, stats):
    """Append a compact, machine-readable progress line after one method+seed block.

    Written to PROGRESS_FILE (plain text, overwrites nothing) so you can watch
    progress in near real time with:  tail -f run_progress.log
    """
    try:
        line = (f"[{time.strftime('%H:%M:%S')}] seed={seed} "
                f"{os.path.basename(log_dir)} mode={stats['mode']} "
                f"mean={stats['mean']:.4f} median={stats['median']:.4f} "
                f"std={stats['std']:.4f} min={stats['min']:.4f} "
                f"cvar_10={stats['cvar_10']:.4f}")
        with open(PROGRESS_FILE, "a") as f:
            f.write(line + "\n")
    except Exception as exc:
        print(f"  [WARN] write progress failed: {exc}")


def print_comparison(results):
    """Print a head-to-head comparison table across multiple methods.

    results: dict {log_dir: stats_dict}. stats_dict carries the same keys as
    run_sobol_test returns (mode, mean, std, min, max, median, p10..., cvar...).
    """
    methods = list(results.keys())
    if len(methods) == 0:
        return

    order = ["mode", "n", "mean", "median", "std", "min",
             "p10", "p25", "cvar_10", "cvar_05", "p90", "p95", "p99", "max"]
    cols = [m for m in order if m in results[methods[0]]]

    print(f"\n  {'='*76}")
    print(f"  {'Method / Metric':<28}" + "".join(f"{m.rsplit('/',1)[-1]:>16}" for m in methods))
    print(f"  {'='*76}")
    for c in cols:
        row = f"  {('mode' if c=='mode' else c):<28}"
        for m in methods:
            if c == "mode":
                row += f"{str(results[m][c]):>16}"
            else:
                row += f"{results[m][c]:>16.4f}"
        print(row)
    print(f"  {'='*76}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", nargs="+", default=["logs/drone-hovering_baseline"],
                        help="One or more log dirs (methods) to test head-to-head.")
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap episode length at this many steps (default 1500).")
    parser.add_argument("--visualize", action="store_true",
                        help="Run single-env viewer visualization instead of Sobol robustness test")
    parser.add_argument("--num-eval-envs", type=int, default=256,
                        help="Sobol grid size (number of DR parameter combos)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("genesis initializing ...", flush=True)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    torch.set_float32_matmul_precision('high')

    # ---- Standard Sobol MC eval mode ----
    if not args.visualize:
        results = {}
        for log_dir in args.log_dir:
            stats = run_sobol_test(
                log_dir, args.ckpt, args.num_eval_envs, args.episodes,
                max_steps=args.max_steps,
            )
            results[log_dir] = stats
            append_progress(log_dir, args.seed, stats)
        print_comparison(results)
        return

    log_dir = args.log_dir[0]
    result = load_policy(log_dir, args.ckpt)
    if result is None:
        print(f"Error: Cannot load policy from {log_dir}")
        sys.exit(1)
    train_cfg, model_path, used_ckpt = result

    print("Building environment ...", flush=True)
    env = EvalEnv(num_envs=1, show_viewer=True, dr_cfg=None)

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.load(model_path, map_location="cpu")
    policy = runner.get_inference_policy(device=gs.device)
    print(f"  Policy loaded (checkpoint {used_ckpt})", flush=True)

    print(f"Running {args.episodes} episodes (mode: RL only) ...", flush=True)

    for ep in range(1, args.episodes + 1):
        obs_dict = env.reset()
        cumulative = 0.0
        step = 0

        with torch.no_grad():
            max_steps = args.max_steps if args.max_steps is not None else 1500
            while step < max_steps:
                actions = policy(obs_dict)
                obs_dict, rews, dones, _ = env.step(actions)
                cumulative += rews[0].item()
                step += 1
                if dones[0]:
                    break
                time.sleep(0.001)

        print(f"  Episode {ep}: return={cumulative:.2f}  steps={step}", flush=True)


if __name__ == "__main__":
    main()