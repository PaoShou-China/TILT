import copy
import gc
import json
import os
import pickle
import shutil
import torch
import torch.nn as nn
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_nn_activation

import genesis as gs

from hover_env import HoverEnv


# ============================================================
# Scale-conditioned trunk (D-SCALE / SSE)
# ============================================================
class ScaleConcatBlock(nn.Module):
    """Plain-MLP trunk with log s CONCATENATED to every layer input (D-SCALE).

    No modulation: the scale scalar x_s = log(L/L0) (last obs column, real
    when include_log_arm_scale=True) is simply concatenated to the input of
    the first layer AND to the input of every subsequent layer:

        h1 = act(L1([x, x_s]));  h2 = act(L2([h1, x_s]));  y = head([h2, x_s])

    Each layer learns its own linear read-out of the scale — purely additive
    conditioning. Trunk = project standard swish [128, 128].

    STOCHASTIC SCALE EXPOSURE (SSE): with SCALE_KEEP_P < 1, each TRAINING
    forward pass keeps the true log s only with probability SCALE_KEEP_P
    (per-sample Bernoulli); on dropped samples the scale column AND all
    concat channels are zeroed, i.e. the network sees EXACTLY the blind-D
    input. Eval always uses the true s. Rationale: full exposure lets the
    policy re-optimize per-scale and weaken its hedge against the residual
    xi (observed: mean up, tail down, 3x); p=0.5 forces it to simultaneously
    solve the blind problem (hedging preserved) while still learning to use
    the scale when available.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, activation="swish"):
        super().__init__()
        self.activation = resolve_nn_activation(activation)
        self.linears = nn.ModuleList()
        prev = input_dim + 1                       # + concat scale column
        for h in hidden_dims:
            self.linears.append(nn.Linear(prev, h))
            prev = h + 1                           # hidden output + concat scale
        self.out = nn.Linear(prev, output_dim)
        self.scale_keep_p = float(os.environ.get("SCALE_KEEP_P", "1.0"))

    def forward(self, x):
        x_s = x[:, -1:]                            # scale column (last dim)
        if self.training and self.scale_keep_p < 1.0:
            keep = (torch.rand_like(x_s) < self.scale_keep_p).float()
            x_s = x_s * keep
            x = torch.cat([x[:, :-1], x_s], dim=-1)  # blind col-17 too
        h = x
        for lin in self.linears:
            h = self.activation(lin(torch.cat([h, x_s], dim=-1)))
        return self.out(torch.cat([h, x_s], dim=-1))


class ScaleConcatModel(MLPModel):
    """MLPModel with the trunk swapped for ScaleConcatBlock (log s concatenated
    to every layer input). Identical interface to MLPModel; requires the
    log-arm obs column (include_log_arm_scale=True) to be the LAST input dim.
    Selected via policy_arch="scale" in get_train_cfg.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(256, 256, 256), activation="elu",
                 obs_normalization=False, distribution_cfg=None):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, obs_normalization, distribution_cfg)
        out_dim = (self.distribution.input_dim
                   if self.distribution is not None else output_dim)
        self.mlp = ScaleConcatBlock(self.obs_dim, out_dim, hidden_dims,
                                    activation)


def build_algo_cfg():
    """PPO algorithm config."""
    return {
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.008,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.0003,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 8,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
        "class_name": "PPO",
    }


# ============================================================
# Train config
# ============================================================
def get_train_cfg(exp_name, policy_arch="mlp", alg_cfg_extra=None):
    """PPO train config.

    policy_arch: "mlp" (default) or "scale" (log-s concat trunk via
        ScaleConcatModel; requires include_log_arm_scale=True env).
    alg_cfg_extra: optional dict MERGED into the algorithm cfg, e.g. to
        inject RiskPPO as the RL objective (risk baseline):
            {"class_name": "risk_ppo:RiskPPO",
             "risk_cfg": {"beta": 2.0, ...}}

    PPO hyperparameters are IDENTICAL for fair comparison — only the
    network architecture differs."""
    model_class = ("hover_train:ScaleConcatModel" if policy_arch == "scale"
                   else "MLPModel")

    algo_cfg = build_algo_cfg()

    actor_cfg = {
        "class_name": model_class,
        "hidden_dims": [128, 128],
        "activation": "swish",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }
    critic_cfg = {
        "class_name": model_class,
        "hidden_dims": [128, 128],
        "activation": "swish",
    }

    if alg_cfg_extra:
        algo_cfg = {**algo_cfg, **alg_cfg_extra}

    return {
        "algorithm": algo_cfg,
        "actor": actor_cfg,
        "critic": critic_cfg,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 100,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


# ============================================================
# Base DR Configuration (Scale-Conditioned)
# ============================================================
# Effective physical params = s^p * xi, with s = L/L0 (arm scale).
# The ranges are the RESIDUAL uncertainty xi (adaptive DR learns xi);
# s (arm_range) sets the base dynamics scale.
BASE_DR = {
    "randomize_mass": True, "mass_range": [0.85, 1.15],
    "randomize_inertia_axis": True, "inertia_axis_range": [0.85, 1.15],
    "randomize_com": True, "com_shift_range": [-0.012, 0.012],
    "randomize_kf": True, "kf_range": [0.85, 1.15],
    "randomize_arm": True, "arm_range": [0.75, 1.25],
    "randomize_turbulence": True,
}

# Entropic-risk DR: q(θ) ∝ exp(−β·R) — exponential tilting / entropic risk.
# Dual-β target KL (kept fixed; drives the β feedback controller).
TARGET_KL = 2.0


def compute_dr_weights(rewards, beta):
    """q(θ) ∝ exp(−β·R) over the eval grid (rewards standardized, ±3σ clip)."""
    r = np.asarray(rewards, dtype=np.float64)
    r = (r - r.mean()) / (r.std() + 1e-8)
    r = np.clip(r, -3.0, 3.0)
    logits = -beta * r
    logits -= logits.max()   # numerical stability
    w = np.exp(logits)
    w /= w.sum()
    return w


def compute_kl(q, p):
    return float(np.sum(q * (np.log(q + 1e-12) - np.log(p + 1e-12))))


def update_beta(beta, kl, lr=0.1):
    # Negative feedback: KL > target → β down (too concentrated), else β up.
    beta = beta * np.exp(lr * (TARGET_KL - kl))
    return float(np.clip(beta, 1e-3, 10.0))


def compute_entropic_dr_weights(rewards, beta, beta_lr, dual_update=True):
    """Entropic-risk DR weights q(θ) = exp(−β·R) with dual-β update.

    Old-β weights → KL(q_old‖uniform); dual-ascent β update; then
    exp(−β·R) weights with an ESS guard (β×0.7 fallback if ESS < 5% N).
    dual_update=False freezes β (fixed-β ablation: KL is still recorded).
    Returns (weights, beta, kl, ess).
    """
    r = np.asarray(rewards, dtype=np.float64)

    q_old = compute_dr_weights(r, beta)
    kl = compute_kl(q_old, np.ones_like(q_old) / len(q_old))
    if dual_update:
        beta = update_beta(beta, kl, beta_lr)

    dr_weights = compute_dr_weights(r, beta)
    ess = 1.0 / (dr_weights ** 2).sum()
    if ess < 0.05 * len(dr_weights):
        beta *= 0.7
        dr_weights = compute_dr_weights(r, beta)
        ess = 1.0 / (dr_weights ** 2).sum()
        print(f"  [Warning] ESS too low, β → {beta:.4f} (implicit entropy regularization)")

    return dr_weights, beta, kl, ess


def evaluate_on_param_grid(env, action_fn, episodes_per_env=1, reset_fn=None):
    """Evaluate a policy/controller over a DR parameter grid.

    action_fn: callable(obs_dict) -> actions. reset_fn: optional callable
    (envs_idx=None) invoked once before the loop and after each done batch.
    """
    num_envs = env.num_envs
    env_reward_sums = [[] for _ in range(num_envs)]
    cumulative = torch.zeros(num_envs, device=gs.device, dtype=gs.tc_float)
    done_count = torch.zeros(num_envs, device=gs.device)

    obs_dict = env.reset()
    if reset_fn is not None:
        reset_fn()
    with torch.no_grad():
        while True:
            actions = action_fn(obs_dict)
            obs_dict, rews, dones, _ = env.step(actions)
            cumulative += rews

            done_envs = dones.nonzero(as_tuple=False).reshape(-1)
            for idx in done_envs:
                idx_i = idx.item()
                env_reward_sums[idx_i].append(cumulative[idx].item())
                cumulative[idx] = 0.0
                done_count[idx] += 1
            if reset_fn is not None and len(done_envs) > 0:
                reset_fn(done_envs)

            if (done_count >= episodes_per_env).all():
                break

    mean_rewards = np.array([np.mean(rs) if rs else -1e9 for rs in env_reward_sums])
    dr_params_raw = env.get_dr_params()
    dr_params_np = {k: v.cpu().numpy() for k, v in dr_params_raw.items()}
    return mean_rewards, dr_params_np


def sample_from_discrete_q(dr_params_np, dr_weights, n_sample):
    """Sample DR params via categorical sampling from q (no perturbation)."""
    n_pool = len(dr_weights)
    idx = np.random.choice(n_pool, size=n_sample, p=dr_weights)

    def _get(key, default_val):
        return dr_params_np[key] if key in dr_params_np else np.full(n_pool, default_val)

    mass_arr = _get("mass", 1.0)
    ixx_arr = _get("ixx", 1.0)
    iyy_arr = _get("iyy", 1.0)
    izz_arr = _get("izz", 1.0)
    com_x_arr = _get("com_x", 0.0)
    com_y_arr = _get("com_y", 0.0)
    com_z_arr = _get("com_z", 0.0)
    kf_arr = _get("kf", 1.0)
    arm_arr = _get("arm", 1.0)

    param_list = []
    for i in idx:
        param_list.append({
            "mass_scale": float(mass_arr[i]),
            "ixx_scale": float(ixx_arr[i]),
            "iyy_scale": float(iyy_arr[i]),
            "izz_scale": float(izz_arr[i]),
            "com_shift_x": float(com_x_arr[i]),
            "com_shift_y": float(com_y_arr[i]),
            "com_shift_z": float(com_z_arr[i]),
            "kf_scale": float(kf_arr[i]),
            "arm_scale": float(arm_arr[i]),
        })

    return param_list


# Sobol DR grid in (s, xi) space: s = arm scale, xi = residual uncertainty,
# effective scales = s^p * xi. Sobol gives uniform coverage (fair comparison).
def _generate_sobol_dr_params(n_samples, dr_ranges=None, seed=None):
    """Generate a Sobol-sequence DR parameter grid in (s, xi) space.

    seed makes the grid deterministic (identical across runs, fair
    comparison). Returns a list of per-env param dicts, or None if scipy
    is unavailable.
    """
    try:
        from scipy.stats import qmc
    except ImportError:
        print("  [WARNING] scipy not available, falling back to uniform DR sampling")
        return None

    if dr_ranges is None:
        dr_ranges = {
            "arm": (0.75, 1.25), "mass": (0.85, 1.15),
            "ixx": (0.85, 1.15), "iyy": (0.85, 1.15),
            "com_x": (-0.012, 0.012), "com_y": (-0.012, 0.012),
            "com_z": (-0.012, 0.012), "kf": (0.85, 1.15),
        }

    dim = len(dr_ranges)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    n_sobol = 1
    while n_sobol < n_samples:
        n_sobol *= 2
    sobol_points = sampler.random(n_sobol)[:n_samples]  # (n_samples, dim)

    keys = list(dr_ranges.keys())
    params_list = []
    for i in range(n_samples):
        param = {}
        for j, key in enumerate(keys):
            low, high = dr_ranges[key]
            param[key] = float(low + sobol_points[i, j] * (high - low))
        params_list.append(param)

    print(f"  [Sobol] Generated {n_samples} parameter points in (s, xi) space (dim={dim}, coverage=quasi-random)")
    return params_list


def _inject_sobol_into_env(env, sobol_params):
    """Inject (s, xi) grid as scale-conditioned DR (effective scales = s^p*xi)."""
    mapped_params = []
    for p in sobol_params:
        s = p["arm"]
        mapped = {
            "arm_scale": float(s),
            "mass_scale": float(s ** 3 * p["mass"]),
            "ixx_scale": float(s ** 5 * p["ixx"]),
            "iyy_scale": float(s ** 5 * p["iyy"]),
            "izz_scale": float(s ** 5),
            "com_shift_x": float(s * p["com_x"]),
            "com_shift_y": float(s * p["com_y"]),
            "com_shift_z": float(s * p["com_z"]),
            "kf_scale": float(s ** 4 * p["kf"]),
        }
        mapped_params.append(mapped)
    n = env.inject_hard_params(mapped_params)
    return n


# Iterative entropic-risk DR training loop (main method D and variants).
def run_entropic_dr_training(
    exp_name, algo, num_envs, num_eval_envs, iterations_per_round,
    total_rounds, episodes_per_eval, eval_sample_method, seed,
    label, build_env, use_tilt=True, fixed_beta=False,
    policy_arch="mlp", alg_cfg_extra=None, weights_fn=None,
):
    """Iterative entropic-risk DR loop: build env -> sample θ~q(θ) -> PPO ->
    eval on fixed grid -> entropic weights + dual-β update.

    build_env: algo-specific env factory.
    alg_cfg_extra: optional dict merged into the algorithm cfg (e.g. to
        inject RiskPPO as the RL objective — used by the risk baseline).

    use_tilt=False disables the entropic re-weighting — training envs keep
        their own (uniform) DR sampling every round; eval stats are still
        recorded on the fixed Sobol grid (A/B/C ablations, risk baseline).
    fixed_beta=True freezes β at its initial value 1.0 (no dual update) —
        the fixed-KL-dual ablation of D; everything else identical.
    weights_fn: optional replacement for compute_entropic_dr_weights with
        the same contract (rewards, beta, beta_lr, dual_update) ->
        (weights, beta, kl, ess) — used by the Famou proxy evaluator.
    """
    log_dir = f"logs/{exp_name}_{algo}"
    os.makedirs(log_dir, exist_ok=True)  # cfgs.pkl / state.pkl / models
    train_cfg = get_train_cfg(exp_name + f"_{algo}", policy_arch=policy_arch,
                              alg_cfg_extra=alg_cfg_extra)
    dr_cfg = {**BASE_DR, "sample_method": eval_sample_method}

    beta, beta_lr = 1.0, 0.2
    beta_history, kl_history = [beta], []
    eval_history = []
    dr_weights, dr_params_np = None, None
    EVAL_SEED = 42

    # Resume support: if a previous run crashed mid-way, continue from the
    # latest saved round state (beta, DR weights, eval history) instead of
    # restarting from scratch.
    state_path = f"{log_dir}/state.pkl"
    start_round = 0
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        beta = state["beta"]
        beta_history = state["beta_history"]
        kl_history = state["kl_history"]
        eval_history = state["eval_history"]
        dr_weights = state["dr_weights"]
        dr_params_np = state["dr_params_np"]
        start_round = state["round"]  # completed rounds
        print(f"  [Resume] log_dir has {start_round} completed rounds "
              f"(beta={beta:.4f}); continuing from round {start_round + 1}")

    for round_idx in range(start_round, total_rounds):
        # (Re)initialize Genesis at each round start: the previous round
        # ends with a full gs.destroy() teardown (see loop bottom), so
        # scene memory starts from a clean slate every round.
        if not gs._initialized:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning",
                    seed=seed, performance_mode=True)
        free_mb = torch.cuda.mem_get_info()[0] / 2**20
        print(f"  [VRAM] {free_mb:,.0f} MB free at round start")
        rnd, is_first = round_idx + 1, round_idx == 0
        print(f"\n{'#'*60}\n  {label} — Round {rnd}/{total_rounds}\n{'#'*60}")

        if is_first:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            saved = [train_cfg, dr_cfg]
            with open(f"{log_dir}/cfgs.pkl", "wb") as f:
                pickle.dump(saved, f)

        # Build env & sample θ ~ q(θ). The factory fully controls the
        # training env (it may ignore dr_cfg, e.g. the nominal-RL ablation).
        env = build_env(num_envs, dr_cfg)
        if use_tilt and not is_first and dr_weights is not None and dr_params_np is not None:
            q_params = sample_from_discrete_q(dr_params_np, dr_weights, num_envs)
            env.inject_hard_params(q_params)
        if use_tilt:
            print(f"  [DR] {'uniform DR (Round 1)' if is_first else f'{num_envs} params from q(theta) (beta={beta:.4f})'}")
        else:
            print("  [DR] uniform DR every round (entropic tilt DISABLED)")

        # PPO update.
        runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
        if is_first:
            runner.learn(num_learning_iterations=iterations_per_round, init_at_random_ep_len=True)
        else:
            prev_ckpt = f"{log_dir}/model_{round_idx * iterations_per_round}.pt"
            if os.path.exists(prev_ckpt):
                runner.load(prev_ckpt, map_location="cpu")
                runner.learn(num_learning_iterations=iterations_per_round)
            else:
                print(f"  WARNING: {prev_ckpt} not found, training from scratch")
                runner.learn(num_learning_iterations=iterations_per_round, init_at_random_ep_len=True)

        ckpt_path = f"{log_dir}/model_{rnd * iterations_per_round}.pt"
        runner.save(ckpt_path)
        print(f"  Checkpoint saved -> {ckpt_path}")

        policy = runner.get_inference_policy(device=gs.device)
        del env, runner
        # Return PyTorch's cached GPU blocks to the driver so the large
        # (num_eval_envs) eval scene can allocate via cuMemAllocAsync.
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate on the FIXED deterministic grid. deterministic_eval_seed
        # fixes init states & goal sequences per grid point so all methods
        # are evaluated under identical conditions each round.
        eval_env = HoverEnv(num_envs=num_eval_envs, show_viewer=False,
                            dr_cfg=dr_cfg, randomize_init_state=True,
                            deterministic_eval_seed=EVAL_SEED)
        if eval_sample_method == "sobol":
            sobol_params = _generate_sobol_dr_params(num_eval_envs, seed=EVAL_SEED)
            if sobol_params is not None:
                _inject_sobol_into_env(eval_env, sobol_params)
        eval_rewards, dr_params_np = evaluate_on_param_grid(eval_env, policy, episodes_per_eval)
        del eval_env
        gc.collect()
        torch.cuda.empty_cache()

        # Entropic-risk weights + dual-β update (tilt runs only; no-tilt
        # ablations keep uniform DR and only record the eval stats).
        if use_tilt:
            _weights_fn = weights_fn if weights_fn is not None else compute_entropic_dr_weights
            dr_weights, beta, kl, ess = _weights_fn(
                eval_rewards, beta, beta_lr, dual_update=not fixed_beta)
        else:
            dr_weights, kl, ess = None, float("nan"), float("nan")
        beta_history.append(beta)
        kl_history.append(kl)
        print(f"\n  DR stats: beta={beta:.4f}  KL={kl:.4f}  ESS={ess:.0f}/{num_eval_envs}")

        eval_history.append({
            "round": rnd,
            "mean": float(eval_rewards.mean()),
            "std": float(eval_rewards.std()),
            "min": float(eval_rewards.min()),
            "max": float(eval_rewards.max()),
            "median": float(np.median(eval_rewards)),
            "beta": beta,
            "kl": kl,
        })

        # Per-round state snapshot (enables resume after a crash).
        with open(state_path, "wb") as f:
            pickle.dump({
                "round": rnd,
                "beta": beta,
                "beta_history": beta_history,
                "kl_history": kl_history,
                "eval_history": eval_history,
                "dr_weights": dr_weights,
                "dr_params_np": dr_params_np,
            }, f)

        # Full teardown at end of round. Genesis 1.3.3 does NOT free scene
        # GPU memory when env/scene objects are deleted (measured: ~1.15GB
        # leaked per round -> cuMemAllocAsync OOM by round ~20 on a 24GB
        # GPU; torch.cuda.empty_cache() cannot reclaim it — the memory is
        # allocated via CUDA's async pool, not PyTorch's caching allocator).
        # gs.destroy() returns everything to the driver; the loop top
        # re-initializes Genesis in <1s. Costs ~1s/round, eliminates the
        # accumulation entirely.
        del policy
        gs.destroy()
        gc.collect()
        torch.cuda.empty_cache()

    # Save histories.
    with open(f"{log_dir}/eval_history.pkl", "wb") as f:
        pickle.dump(eval_history, f)
    with open(f"{log_dir}/eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2)
    print(f"\n  Evaluation history saved -> {log_dir}/eval_history.pkl")

    print(f"\n{'='*60}\n  {label} COMPLETE: {total_rounds} rounds")
    print(f"  Final model: {log_dir}/model_{total_rounds * iterations_per_round}.pt")
    print(f"  beta trajectory: {[round(b, 4) for b in beta_history]}\n{'='*60}")


# ============================================================
# Adaptive Domain Randomization (OpenAI-style ADR) — baseline
# ============================================================
# Unlike the entropic-risk DR (which adapts the SAMPLING distribution), ADR
# adapts the DR RANGE width: a scalar progress p drives how far each
# randomized factor can deviate from its nominal value. When the current
# policy succeeds on the current (scaled) range above SUCCESS_HIGH, p grows
# by ADR_STEP toward 1 (the full BASE_DR range); below SUCCESS_LOW it
# shrinks. No statistical re-weighting is performed (plain PPO + MLP).
#
# Per-round eval is DONE TWICE:
#   * current-range Sobol grid -> drive the ADR success → p update
#   * fixed FULL Sobol grid    -> identical tracking protocol to the other
#                                  methods (fair comparison).
ADR_PROG_INIT = 0.1      # start at 10% of full range
ADR_STEP = 0.1
ADR_SUCCESS_HIGH = 0.7   # grow range if success rate above this
ADR_SUCCESS_LOW = 0.3    # shrink range if success rate below this
ADR_SUCCESS_R_THRESH = 0.0   # an episode "succeeds" if its mean reward > 0


def _adr_dr_cfg(prog):
    """Scale-conditioned BASE_DR with all range widths scaled by prog∈[0,1]."""
    def mul(center, width):
        return [center - width * prog, center + width * prog]
    return {
        "randomize_mass": True,       "mass_range": mul(1.0, 0.15),
        "randomize_inertia_axis": True, "inertia_axis_range": mul(1.0, 0.15),
        "randomize_com": True,        "com_shift_range": mul(0.0, 0.012),
        "randomize_kf": True,         "kf_range": mul(1.0, 0.15),
        "randomize_arm": True,        "arm_range": mul(1.0, 0.25),
        "randomize_turbulence": True,
        "sample_method": "sobol",
    }


def _adr_sobol_ranges(prog):
    """(s, xi)-space Sobol ranges for the CURRENT ADR range (eval A)."""
    def mul(center, width):
        return (center - width * prog, center + width * prog)
    return {
        "arm": mul(1.0, 0.25),
        "mass": mul(1.0, 0.15),
        "ixx": mul(1.0, 0.15),
        "iyy": mul(1.0, 0.15),
        "com_x": mul(0.0, 0.012),
        "com_y": mul(0.0, 0.012),
        "com_z": mul(0.0, 0.012),
        "kf": mul(1.0, 0.15),
    }


def run_adr_training(
    exp_name, algo, num_envs, num_eval_envs, iterations_per_round,
    total_rounds, episodes_per_eval, seed, label, build_env,
):
    """Iterative OpenAI-style ADR: PPO(MLP) -> eval on current range -> grow/
    shrink range via success -> repeat. Tracks final-eval protocol identically."""
    from rsl_rl.runners import OnPolicyRunner

    log_dir = f"logs/{exp_name}_{algo}"
    train_cfg = get_train_cfg(exp_name + f"_{algo}", policy_arch="mlp")
    EVAL_SEED = 42

    prog = ADR_PROG_INIT
    prog_history = []
    eval_history = []
    eval_rewards_full = None

    # Resume support.
    state_path = f"{log_dir}/state.pkl"
    start_round = 0
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        prog = state["prog"]
        prog_history = state["prog_history"]
        eval_history = state["eval_history"]
        eval_rewards_full = state["eval_rewards_full"]
        start_round = state["round"]
        print(f"  [Resume] {start_round} completed rounds (prog={prog:.3f})")

    for round_idx in range(start_round, total_rounds):
        if not gs._initialized:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning",
                    seed=seed, performance_mode=True)
        free_mb = torch.cuda.mem_get_info()[0] / 2**20
        print(f"  [VRAM] {free_mb:,.0f} MB free at round start")
        rnd, is_first = round_idx + 1, round_idx == 0
        print(f"\n{'#'*60}\n  {label} — Round {rnd}/{total_rounds}\n{'#'*60}")

        if is_first:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/cfgs.pkl", "wb") as f:
                pickle.dump([train_cfg, _adr_dr_cfg(prog)], f)

        # --- Train PPO (MLP) under the CURRENT (scaled) DR range ---
        dr_cfg = _adr_dr_cfg(prog)
        print(f"  [ADR] training range progress = {prog:.3f}")
        env = build_env(num_envs, dr_cfg)
        runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
        if is_first:
            runner.learn(num_learning_iterations=iterations_per_round, init_at_random_ep_len=True)
        else:
            prev_ckpt = f"{log_dir}/model_{round_idx * iterations_per_round}.pt"
            if os.path.exists(prev_ckpt):
                runner.load(prev_ckpt, map_location="cpu")
                runner.learn(num_learning_iterations=iterations_per_round)
            else:
                print(f"  WARNING: {prev_ckpt} not found, training from scratch")
                runner.learn(num_learning_iterations=iterations_per_round, init_at_random_ep_len=True)
        ckpt_path = f"{log_dir}/model_{rnd * iterations_per_round}.pt"
        runner.save(ckpt_path)
        policy = runner.get_inference_policy(device=gs.device)
        del env, runner
        gc.collect()
        torch.cuda.empty_cache()

        # --- (A) Current-range eval -> ADR success -> update prog ---
        eval_env = HoverEnv(num_envs=num_eval_envs, show_viewer=False,
                            dr_cfg=dr_cfg, randomize_init_state=True,
                            deterministic_eval_seed=EVAL_SEED)
        sobol_cur = _generate_sobol_dr_params(num_eval_envs,
                                              dr_ranges=_adr_sobol_ranges(prog),
                                              seed=EVAL_SEED)
        if sobol_cur is not None:
            _inject_sobol_into_env(eval_env, sobol_cur)
        cur_rewards, _ = evaluate_on_param_grid(eval_env, policy, episodes_per_eval)
        del eval_env
        gc.collect()
        torch.cuda.empty_cache()
        cur_r = np.asarray(cur_rewards)
        success_rate = float((cur_r > ADR_SUCCESS_R_THRESH).mean())
        if success_rate >= ADR_SUCCESS_HIGH:
            prog = min(1.0, prog + ADR_STEP)
        elif success_rate <= ADR_SUCCESS_LOW:
            prog = max(0.0, prog - ADR_STEP)
        prog_history.append(prog)
        print(f"  [ADR] current-range success = {success_rate:.3f} -> prog = {prog:.3f}")

        # --- (B) Fixed FULL Sobol grid eval (identical tracking protocol) ---
        eval_env = HoverEnv(num_envs=num_eval_envs, show_viewer=False,
                            dr_cfg={**BASE_DR, "sample_method": "sobol"},
                            randomize_init_state=True,
                            deterministic_eval_seed=EVAL_SEED)
        sobol_full = _generate_sobol_dr_params(num_eval_envs, seed=EVAL_SEED)
        if sobol_full is not None:
            _inject_sobol_into_env(eval_env, sobol_full)
        eval_rewards_full, _ = evaluate_on_param_grid(eval_env, policy, episodes_per_eval)
        del eval_env
        gc.collect()
        torch.cuda.empty_cache()

        eval_history.append({
            "round": rnd,
            "mean": float(eval_rewards_full.mean()),
            "std": float(eval_rewards_full.std()),
            "min": float(eval_rewards_full.min()),
            "max": float(eval_rewards_full.max()),
            "median": float(np.median(eval_rewards_full)),
            "prog": prog,
        })

        with open(state_path, "wb") as f:
            pickle.dump({
                "round": rnd, "prog": prog,
                "prog_history": prog_history,
                "eval_history": eval_history,
                "eval_rewards_full": eval_rewards_full,
            }, f)

        del policy
        gs.destroy()
        gc.collect()
        torch.cuda.empty_cache()

    with open(f"{log_dir}/eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2)
    print(f"\n{'='*60}\n  {label} COMPLETE: {total_rounds} rounds")
    print(f"  Final model: {log_dir}/model_{total_rounds * iterations_per_round}.pt")
    print(f"  ADR progress trajectory: {[round(p, 3) for p in prog_history]}\n{'='*60}")


# Policy loading.
def _find_latest_checkpoint(log_dir):
    """Return (model_path, ckpt_index) for the latest model_*.pt, or
    (None, None) if none exist."""
    model_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")],
        key=lambda x: int(x.replace("model_", "").replace(".pt", "")),
    )
    if not model_files:
        return None, None
    ckpt_idx = int(model_files[-1].replace("model_", "").replace(".pt", ""))
    return os.path.join(log_dir, model_files[-1]), ckpt_idx


def load_policy(log_dir, ckpt=None):
    """Load trained policy from log directory."""
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        return None

    if ckpt is None:
        model_path, used_ckpt = _find_latest_checkpoint(log_dir)
        if model_path is None:
            return None
    else:
        model_path = f"{log_dir}/model_{ckpt}.pt"
        used_ckpt = ckpt
        if not os.path.exists(model_path):
            model_path, used_ckpt = _find_latest_checkpoint(log_dir)
            if model_path is None:
                return None

    with open(cfg_path, "rb") as f:
        loaded = pickle.load(f)
        train_cfg = loaded[0] if isinstance(loaded, list) else loaded

    alg = train_cfg.get("algorithm", {})
    if "class_name" not in alg:
        train_cfg.setdefault("algorithm", {})["class_name"] = "PPO"

    return train_cfg, model_path, used_ckpt