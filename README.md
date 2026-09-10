# TILT

**T**ail-Aware **I**terative **L**earning via **T**ilting — robust sim-to-real-oriented RL
for quadrotor hovering, built on the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics simulator and
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO implementation.

[Method](#1-method-overview) · [Installation](#3-requirements-and-installation) ·
[Quick start](#4-quick-start) · [Evaluation](#6-evaluation) ·
[Related repositories](#13-related-repositories-and-attribution)

This folder is a minimal, self-contained release of the paper's main method
(Table III, row "TILT (ours)"): entropic-risk domain randomization with a
dual-β KL controller, a scale-conditioned policy network, and Stochastic
Scale Exposure (SSE).

---

## 1. Method overview

Training alternates between policy optimization and environment-distribution
adaptation:

1. **Entropic-risk domain randomization (the core).**
   After each training round, the current policy is evaluated on a *fixed*
   4096-point Sobol grid that spans the whole DR manifold. The training
   environment distribution for the next round is re-sampled with
   exponential-tilted weights

   ```
   q(θ) ∝ exp(−β · R(θ))
   ```

   where `R(θ)` is the mean return at DR point `θ`. Hard parameters are thus
   oversampled and easy ones undersampled, concentrating training where the
   policy actually fails.

2. **Dual-β KL controller.** The tilt strength β is not hand-tuned. Each
   round measures `KL(q ‖ uniform)`; if the tilt is too sharp (KL above the
   target 2.0) β decreases, otherwise it increases (dual ascent, lr 0.2,
   β init 1.0). An ESS guard falls back to β ← 0.7·β if the effective
   sample size drops below 5% of the grid, preventing collapse onto a
   handful of points.

3. **Scale-conditioned policy (`ScaleConcatModel`).** The global dynamics
   scale `s` (arm length ratio `L/L0`) is appended — as `log s` — to the
   input of *every* layer of a swish [128, 128] MLP trunk:

   ```
   h1 = act(L1([x, x_s]));  h2 = act(L2([h1, x_s]));  y = head([h2, x_s])
   ```

   Each layer learns its own linear read-out of the scale (purely additive
   conditioning, no FiLM/modulation).

4. **SSE — Stochastic Scale Exposure (p = 0.9).** During *training*, each
   sample keeps its true `log s` column only with probability
   `SCALE_KEEP_P = 0.9` (per-sample Bernoulli mask; dropped samples see
   exactly the scale-blind input). At evaluation the true `log s` is always
   provided. Full exposure (p = 1.0) lets the policy specialize per scale and
   weakens its hedge against the residual uncertainties; p = 0.9 preserves
   the hedge while still teaching the network to use the scale when
   available.

## 2. Repository layout

```
TILT/
├── main_tilt.py    # One-command entry point: training + final robustness
│                   #   evaluation. All defaults hard-coded, no CLI needed.
├── hover_train.py  # Training core: BASE_DR config, entropic weight kernel +
│                   #   dual-β controller, ScaleConcatModel (SSE), the
│                   #   round-loop trainer, Sobol grid generation, ADR baseline.
├── hover_env.py    # Genesis quadrotor hovering environment: DR, turbulence,
│                   #   mirror-symmetric obs/actions, deterministic eval tables.
├── hover_eval.py   # Evaluation tools: Sobol-grid robustness test, single-env
│                   #   viewer, multi-method head-to-head comparison table.
├── requirements.txt # Pinned runtime dependencies (except PyTorch).
├── .gitignore      # Excludes checkpoints, logs, caches, and local environments.
└── README.md       # Documentation and reproduction guide.
```

The three `hover_*.py` files are copied unmodified from the paper's codebase,
so this release reproduces the reported protocol exactly.

## 3. Requirements and installation

- **GPU:** one NVIDIA GPU. Tested on RTX 3090 (24 GB), driver 580.173.02.
  Training (1024 envs) uses ~2 GB; the 4096-env evaluation scene is the
  peak allocation. ≥8 GB VRAM is recommended.
- **OS:** Linux.

Verified dependency versions:

| Package | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| genesis-world | 1.3.3 |
| rsl-rl-lib | 5.3.0 |
| numpy | 2.2.6 |
| tensorboard | 2.20.0 |

### Installation

```bash
# 1. Create a conda environment
conda create -n tilt python=3.10 -y
conda activate tilt

# 2. Install PyTorch (CUDA 12.8 build)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install the remaining pinned dependencies
pip install -r requirements.txt
```

## 4. Quick start

```bash
cd TILT
python main_tilt.py            # train seed 1 + final robustness eval
```

Use a different training seed:

```bash
TILT_SEED=2 python main_tilt.py
```

That is the entire workflow. `main_tilt.py` runs the full paper protocol
with hard-coded defaults and prints the final statistics at the end.

## 5. Training protocol

Identical to the paper's main-table protocol:

| Stage | Setting |
|---|---|
| Training environments | 1024 (parallel, GPU) |
| Rounds | 20 |
| PPO iterations per round | 50 (`num_steps_per_env` = 100) |
| Round-1 DR sampling | uniform over the Sobol grid |
| Rounds 2–20 DR sampling | `θ ~ q(θ)` from the previous round's entropic weights |
| Mid-training eval | fixed 4096-pt Sobol grid, 15 episodes/pt, eval seed 42, **scale column zeroed** (drives the weight kernel) |
| Final eval | same grid, 10 episodes/pt, turbulence ON, **true scale column** |

Per round the loop: (re)builds the training env with the current `q(θ)`,
runs PPO, saves a checkpoint, evaluates on the fixed grid, computes the
entropic weights + dual-β update, and snapshots `state.pkl`. Genesis is fully
torn down and re-initialized between rounds (`gs.destroy()`) because Genesis
1.3.3 leaks scene GPU memory across rounds otherwise (~1.15 GB/round on a
24 GB GPU).

## 6. Evaluation

### Final robustness eval (automatic)

After training, `main_tilt.py` calls `final_robustness_eval`: 4096-point
Sobol DR grid, 10 episodes per point, episode capped at 1500 steps,
turbulence ON, eval seed 42 (fixes the Sobol grid **and** the initial-state /
goal lookup tables, so any two methods/seeds are compared under identical
conditions). Results are written to `logs/TILT_s{seed}/final_eval.json`.

### Standalone evaluation

```bash
# Fast smoke check (256-pt grid, 3 eps/pt)
python hover_eval.py --log-dir logs/TILT_s1

# Full paper protocol
python hover_eval.py --log-dir logs/TILT_s1 --num-eval-envs 4096 --episodes 10

# Head-to-head comparison of several methods under the identical protocol
python hover_eval.py --log-dir logs/TILT_s1 logs/TILT_s2

# Single environment with the Genesis viewer
python hover_eval.py --log-dir logs/TILT_s1 --visualize

# Evaluate a specific checkpoint (default: latest)
python hover_eval.py --log-dir logs/TILT_s1 --ckpt 500
```

## 7. Checkpointing, resume, and skip-if-done

- **Resume after a crash.** Every completed round writes `state.pkl`
  (β, DR weights, eval history, round counter). Re-running
  `python main_tilt.py` resumes from the latest completed round.
- **Skip completed training.** If the final checkpoint
  `model_1000.pt` (= 20 rounds × 50 iters) exists, training is skipped and
  only the final evaluation runs.
- **Skip completed evaluation.** If `final_eval.json` already exists, the
  final eval is skipped (delete the file to force a re-evaluation).
- **Warning.** `run_entropic_dr_training` *wipes the log directory at the
  start of round 1*. The skip-if-done guard therefore lives in
  `main_tilt.py` (it checks `model_1000.pt` before calling the trainer).
  If you want a fresh run, delete `logs/TILT_s{seed}/` yourself.

## 8. Output files (`logs/TILT_s{seed}/`)

| File | Contents |
|---|---|
| `model_*.pt` | PPO checkpoints, one per round (`model_1000.pt` = final) |
| `final_eval.json` | Final robustness statistics (see §9) |
| `eval_history.json` / `.pkl` | Per-round fixed-grid eval history: mean/std/min/max/median, β, KL |
| `state.pkl` | Resume state: β, β/KL histories, DR weights, sampled params, completed rounds |
| `cfgs.pkl` | Snapshot of `(train_cfg, dr_cfg)` used for the run |
| `events.out.tfevents*` | TensorBoard logs — `tensorboard --logdir logs/TILT_s1` |

## 9. Metrics

`final_eval.json` contains, over the 4096 DR points (mean episode return per
point):

| Key | Meaning |
|---|---|
| `mean` / `median` / `std` | Average / median / spread of per-point returns |
| `p10` | 10th percentile of per-point returns |
| `cvar_10` | Mean of the worst 10% of points — **primary tail-risk metric** |
| `cvar_05` | Mean of the worst 5% of points — secondary |
| `min` | Worst single point — diagnostic only |

## 10. Domain randomization configuration

DR is specified in `BASE_DR` (`hover_train.py`) as residual ranges; the Sobol
grid spans the product space of the scale `s` and all residuals (8-D):

| Factor | Range |
|---|---|
| Scale `s` (arm length ratio `L/L0`) | [0.75, 1.25] |
| Mass residual `ξ_m = m/m_nom(s)` | [0.85, 1.15] |
| kf (thrust coefficient) residual `ξ_f` | [0.85, 1.15] |
| Inertia axis residuals (ixx, iyy) | [0.85, 1.15] |
| Center-of-mass shift | ±12 mm |
| Turbulence | strength 0.08 (eval ON) |

Effective mass scales as `s³·ξ_m` and kf as `s⁴·ξ_f`. The nominal hover RPM
scales as `s^(−1/2)`. The observation always contains a scale column (last
dim): the true `log s` when `include_log_arm_scale=True`, otherwise a
constant 0.

## 11. Key hyperparameters (all hard-coded in `main_tilt.py` / `hover_train.py`)

| Item | Value |
|---|---|
| PPO | lr 3e-4 (adaptive, desired_kl 0.01), γ 0.99, λ 0.95, clip 0.2, entropy 0.008, 4 mini-batches, 8 epochs |
| Network | swish, hidden [128, 128], Gaussian scalar std, same trunk for actor/critic |
| Entropic DR | β₀ = 1.0, dual lr 0.2, target KL = 2.0, ESS floor 5% (fallback β ← 0.7·β) |
| SSE | `SCALE_KEEP_P` = 0.9 (env var, read at model construction) |
| Budget | 20 rounds × 50 iters, `num_steps_per_env` = 100 |
| Eval | Sobol seed 42, final 10 eps/pt, `max_steps` 1500, turbulence ON |

## 12. Reference results

Final 4096-point Sobol evaluation protocol, mean ± std across training seeds:

| Method | Return (mean) | CVaR₁₀ |
|---|---|---|
| Conventional DR (paper, 3 seeds) | 1.859 ± 0.672 | 1.116 |
| **TILT (paper, 3 seeds)** | **3.045 ± 0.057** | **2.636** |
| Conventional DR (5 seeds) | 1.981 ± 0.510 | 1.238 ± 0.521 |
| **TILT (5 seeds)** | **3.044 ± 0.074** | **2.629 ± 0.064** |

TILT's cross-seed variance is an order of magnitude smaller, and its worst
10% of DR points retain a return above 2.5 on every seed, while
conventional DR's tail collapses toward zero on some seeds.

## 13. Related repositories and attribution

TILT relies on and/or adapts components from the following open-source
projects. Please follow each upstream project's license and citation
instructions when reusing their code.

| Project | Use in TILT | Original repository |
|---|---|---|
| Genesis | Physics simulation, Crazyflie model, and the basis of the hovering environment | [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) |
| Genesis hovering example | Upstream reference for the environment structure | [examples/drone/hover_train.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_train.py) |
| rsl_rl | PPO runner, `MLPModel`, and activation utilities | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) |
| PyTorch | Tensor operations, neural networks, CUDA execution, and Sobol sequences | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| NumPy | Numerical processing and evaluation statistics | [numpy/numpy](https://github.com/numpy/numpy) |
| TensorBoard | Training-log visualization | [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) |

The links above point to the original repositories rather than forks.
TILT-specific method code, experiment settings, and modifications are contained
in this repository; upstream projects retain their respective licenses.

## 14. Troubleshooting

- **CUDA OOM during the 4096-pt eval.** Close other GPU processes; the eval
  scene is the peak allocation. The round-loop teardown already returns all
  scene memory to the driver between rounds.
- **Want to re-run the final eval.** Delete `logs/TILT_s{seed}/final_eval.json`
  and re-run `python main_tilt.py` (training is skipped; only the eval reruns).
- **Want a fresh training run.** Delete the whole `logs/TILT_s{seed}/` folder
  (see §7 warning).
- **Viewer does not open.** `--visualize` requires a display; on a headless
  server use the Sobol eval modes instead.
