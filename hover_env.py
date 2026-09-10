import torch
import math
import copy
import logging
from tensordict import TensorDict

import genesis as gs
from genesis.engine.force_fields import Turbulence
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from genesis.utils.misc import qd_to_torch

# Filter benign Genesis URDF parsing warnings (harmless physics messages).
_GENESIS_BENIGN_WARNINGS = (
    "free joint has non-zero frictionloss, damping or armature parameters",
    "is not watertight. Falling back to convex hull",
)
_genesis_warning_filter_installed = False


class _BenignGenesisWarningFilter(logging.Filter):
    def filter(self, record):
        return not any(msg in record.getMessage() for msg in _GENESIS_BENIGN_WARNINGS)


def _install_genesis_warning_filter():
    global _genesis_warning_filter_installed
    if _genesis_warning_filter_installed:
        return
    logger = logging.getLogger("genesis")
    logger.addFilter(_BenignGenesisWarningFilter())
    _genesis_warning_filter_installed = True


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class HoverEnv:
    def __init__(self, num_envs, visualize_camera=False, show_viewer=False, dr_cfg=None,
                 randomize_init_state=False, include_log_arm_scale=False,
                 deterministic_eval_seed=None):
        self.num_envs = num_envs
        self.dt = 0.01
        self.dr_cfg = dr_cfg
        # Switch: append log(L/L0) to the observation. When ON it exposes the
        # real (per-env) arm-scale; when OFF the column is always 0 so the
        # policy sees a constant and learns to ignore it.
        self.include_log_arm_scale = include_log_arm_scale
        self.rendered_env_num = min(10, num_envs)
        self.device = gs.device
        self._randomize_init_state = randomize_init_state
        # Deterministic evaluation: when set, init-state noise and goal
        # (command) resampling are drawn from pre-generated lookup tables
        # indexed by (env_idx, reset/resample counter) instead of the global
        # RNG. Two runs with the same seed then see IDENTICAL initial states
        # and goal sequences regardless of when episodes terminate -- required
        # for fair head-to-head policy comparison.
        self._det_seed = deterministic_eval_seed
        # Nominal hover RPM; scales as s^(-1/2) under scale-conditioned DR.
        self.nominal_hover_rpm = 14468.429183500699

        self.env_cfg = {
            "visualize_target": True if show_viewer else False,
            "visualize_camera": visualize_camera,
            "max_visualize_FPS": 60,
            "termination_if_roll_greater_than": 180,
            "termination_if_pitch_greater_than": 180,
            "termination_if_close_to_ground": 0.1,
            "termination_if_x_greater_than": 3.0,
            "termination_if_y_greater_than": 3.0,
            "termination_if_z_greater_than": 2.0,
            "base_init_pos": [0.0, 0.0, 1.0],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 15.0,
            "at_target_threshold": 0.1,
            "resampling_time_s": 3.0,
            "simulate_action_latency": True,
            "clip_actions": 1.0,
            # RPM is capped so thrust stays bounded under any DR combo (no NaN).
            "max_rpm_ratio": 2.0,
        }
        # Actor output dim: 4 rotor RPMs.
        self.num_actions = 4
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)

        self.obs_scales = {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / math.pi,
        }
        self.reward_cfg = {
            "yaw_lambda": -10.0,
            "reward_scales": {
                "target": 10.0,
                "smooth": -1e-4,
                "yaw": 0.01,
                "angular": -2e-4,
                "crash": -10.0,
            },
        }
        self.command_cfg = {
            "num_commands": 3,
            "pos_x_range": [-1.0, 1.0],
            "pos_y_range": [-1.0, 1.0],
            "pos_z_range": [1.0, 1.0],
        }
        self.num_commands = self.command_cfg["num_commands"]
        self.reward_scales = copy.deepcopy(self.reward_cfg["reward_scales"])

        if self._det_seed is not None:
            self._build_deterministic_tables(self._det_seed)

        self._build_scene(show_viewer)
        self._init_buffers()

        if self.dr_cfg is not None:
            self._pre_sample_dr_params()
            self._randomize_dynamics(torch.arange(self.num_envs, device=gs.device))

        self._init_rewards()
        self.cfg = self.env_cfg
        self.reset()

    def _build_scene(self, show_viewer, add_turbulence=True):
        _install_genesis_warning_filter()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=self.env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_links_info=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)),
                ),
            )
        else:
            self.target = None

        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        if add_turbulence and self.dr_cfg is not None and self.dr_cfg.get("randomize_turbulence", False):
            turb_strength = self.dr_cfg.get("turbulence_strength", 0.08)
            self.scene.add_force_field(Turbulence(strength=turb_strength, frequency=1.0))

        self.scene.build(n_envs=self.num_envs)

        if self.dr_cfg is not None:
            self._base_link_idx = self.drone.get_link("base_link").idx_local
            self._nominal_mass = self.drone.get_link("base_link").get_mass()
            self._prev_inertia_scales = torch.ones((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
            self._base_link_global = self._base_link_idx + self.drone._link_start

            self._rotor_link_locals = [int(g - self.drone._link_start) for g in self.drone._propellers_link_idx]
            self._rotor_nominal_pos = torch.tensor(
                [[0.028, -0.028, 0.0], [-0.028, -0.028, 0.0], [-0.028, 0.028, 0.0], [0.028, 0.028, 0.0]],
                dtype=gs.tc_float, device=gs.device,
            )

        self._kf_scales = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self._arm_scales = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _init_buffers(self):
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = {}

    def _init_rewards(self):
        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _build_deterministic_tables(self, seed):
        """Pre-generate per-env random tables for init-state noise & goals.

        Indexed by (env_idx, counter % K): the k-th reset / k-th goal
        resample of env i always yields the same values, independent of the
        global RNG state and of when episodes terminate. Tables are built on
        a dedicated CPU generator so both policies in an A/B eval see
        identical initial states and goal sequences.
        """
        n = self.num_envs
        k_eps = 64     # init-state table depth (per env)
        k_cmd = 512    # goal table depth (per env)
        g = torch.Generator()
        g.manual_seed(seed)

        # device="cpu": the global default device may be CUDA (Genesis),
        # which would demand a CUDA generator.
        u_pos = torch.rand((n, k_eps, 3), generator=g, device="cpu")
        u_yaw = torch.rand((n, k_eps), generator=g, device="cpu")
        u_cmd = torch.rand((n, k_cmd, 3), generator=g, device="cpu")

        # Same distributions as gs_rand_float in reset_idx / _resample_commands.
        self._det_pos_noise = (u_pos * 2.0 - 1.0) * 0.05
        self._det_yaw = (u_yaw * 2.0 - 1.0) * (math.pi / 6.0)
        cmd = torch.empty_like(u_cmd)
        for j, key in enumerate(("pos_x_range", "pos_y_range", "pos_z_range")):
            lo, hi = self.command_cfg[key]
            cmd[..., j] = lo + u_cmd[..., j] * (hi - lo)
        self._det_cmds = cmd

        self._det_pos_noise = self._det_pos_noise.to(device=gs.device, dtype=gs.tc_float)
        self._det_yaw = self._det_yaw.to(device=gs.device, dtype=gs.tc_float)
        self._det_cmds = self._det_cmds.to(device=gs.device, dtype=gs.tc_float)

        self._det_k_eps = k_eps
        self._det_k_cmd = k_cmd
        self._det_ep_count = torch.zeros(n, dtype=torch.long, device=gs.device)
        self._det_cmd_count = torch.zeros(n, dtype=torch.long, device=gs.device)

    def _resample_commands(self, envs_idx):
        if self._det_seed is not None:
            # Deterministic goal sequence: k-th resample of env i is fixed.
            k = self._det_cmd_count[envs_idx] % self._det_k_cmd
            self.commands[envs_idx] = self._det_cmds[envs_idx, k]
            self._det_cmd_count[envs_idx] += 1
            return

        def sample_range(key):
            r = self.command_cfg[key]
            return gs_rand_float(r[0], r[1], (len(envs_idx),), gs.device)

        self.commands[envs_idx, 0] = sample_range("pos_x_range")
        self.commands[envs_idx, 1] = sample_range("pos_y_range")
        self.commands[envs_idx, 2] = sample_range("pos_z_range")

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # Scale-Conditioned DR: hover rpm scales as s^(-1/2) (m ~ s^3, k_f ~ s^4).
        # kf factor is applied in the dynamics kernel via kf_scales, not the RPM.
        scale_ratio = self._arm_scales
        hover_rpm = self.nominal_hover_rpm * scale_ratio.pow(-0.5)

        rotor_rpm = (1 + self.actions * 0.8) * hover_rpm.unsqueeze(-1)

        # Clamp RPM so thrust stays bounded under extreme DR points.
        max_rpm = self.env_cfg["max_rpm_ratio"] * hover_rpm.unsqueeze(-1)
        rotor_rpm = torch.clamp_min(rotor_rpm, 0.0)
        rotor_rpm = torch.clamp_max(rotor_rpm, max_rpm.to(dtype=rotor_rpm.dtype))
        self.drone.set_propellers_rpm(rotor_rpm, kf_scales=self._kf_scales)
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._update_observation()
        self.last_actions[:] = self.actions[:]

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf},
                          batch_size=[self.num_envs])

    def _update_observation(self):
        obs_parts = [
            torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
            self.base_quat,
            torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
            self.last_actions,
        ]
        # Log arm-scale column (switch-controlled). When enabled, the real
        # per-env log(L/L0) is exposed; otherwise the column stays 0.
        if self.include_log_arm_scale:
            s = getattr(self, "_dr_arm_scales", None)
            if s is None:  # no DR -> nominal scale s=1 -> log=0
                log_arm = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
            else:
                log_arm = torch.log(s.clamp(min=1e-6)).unsqueeze(-1)
            obs_parts.append(log_arm)
        else:
            obs_parts.append(torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float))
        self.obs_buf = torch.cat(obs_parts, axis=-1)

    def _pre_sample_dr_params(self):
        n = self.num_envs

        def sample_1d(lower, upper):
            return gs_rand_float(lower, upper, (n,), gs.device)

        def sample_if(key):
            if not self.dr_cfg.get(key, False):
                return torch.ones(n, device=gs.device)
            key_ = key.replace("randomize_", "")
            r = self.dr_cfg[key_ + "_range"]
            return sample_1d(r[0], r[1])

        # ---- Scale variable s = L / L0 (arm length) ----
        if self.dr_cfg.get("randomize_arm", False):
            r = self.dr_cfg.get("arm_range", [0.9, 1.1])
            s = sample_1d(r[0], r[1])
        else:
            s = torch.ones(n, device=gs.device)
        self._dr_arm_scales = s

        # Scale-Conditioned DR: effective = s^p * xi for mass/I/kf/com.
        xi_m = sample_if("randomize_mass")
        self._dr_mass_scales = (s.pow(3) * xi_m).unsqueeze(-1)  # (n, 1)

        # Inertia: per-axis (ixx, iyy) or legacy scalar.
        if self.dr_cfg.get("randomize_inertia_axis", False):
            r = self.dr_cfg.get("inertia_axis_range", [0.9, 1.1])
            ixx = sample_1d(r[0], r[1])   # (n,)
            iyy = sample_1d(r[0], r[1])   # (n,)
            izz = torch.ones(n, device=gs.device)
            xi_I = torch.stack([ixx, iyy, izz], dim=-1)  # (n, 3)
        elif self.dr_cfg.get("randomize_inertia", False):
            r = self.dr_cfg["inertia_range"]
            xi_I = sample_1d(r[0], r[1]).unsqueeze(-1).repeat(1, 3)  # (n, 3)
        else:
            xi_I = torch.ones((n, 3), device=gs.device)
        self._dr_inertia_scales = s.pow(5).unsqueeze(-1) * xi_I  # (n, 3)

        xi_kf = sample_if("randomize_kf")
        self._dr_kf_scales = s.pow(4) * xi_kf  # (n,)

        if self.dr_cfg.get("randomize_com", False):
            r = self.dr_cfg["com_shift_range"]
            com_x = sample_1d(r[0], r[1])
            com_y = sample_1d(r[0], r[1])
            com_z = sample_1d(r[0], r[1])
            dc = torch.stack([com_x, com_y, com_z], dim=-1)  # (n, 3)
            self._dr_com_shifts = (s.unsqueeze(-1) * dc).unsqueeze(-2)  # (n, 1, 3)

    def _apply_dynamics_to_envs(self, envs_idx):
        """Apply pre-sampled DR parameters to the physics engine for given envs."""
        if len(envs_idx) == 0:
            return

        if hasattr(self, "_dr_inertia_scales"):
            new_scale = self._dr_inertia_scales[envs_idx]
            self.drone._solver.set_links_inertia(
                1.0 / self._prev_inertia_scales[envs_idx, :1].clamp(min=1e-6),
                links_idx=[self._base_link_global],
                envs_idx=envs_idx,
            )
            self.drone._solver.set_links_inertia(
                new_scale[:, :1],
                links_idx=[self._base_link_global],
                envs_idx=envs_idx,
            )
            ixx = new_scale[:, 0].clamp(min=1e-6)
            corr_iyy = new_scale[:, 1] / ixx
            corr_izz = new_scale[:, 2] / ixx
            inertial_i_data = qd_to_torch(self.drone._solver.dyn_info.links.inertial_i, transpose=True, copy=False)
            inertial_i_data[envs_idx, self._base_link_global, 1, 1] *= corr_iyy
            inertial_i_data[envs_idx, self._base_link_global, 2, 2] *= corr_izz
            self._prev_inertia_scales[envs_idx] = new_scale

        if hasattr(self, "_dr_mass_scales"):
            self.drone.set_links_inertial_mass(
                self._nominal_mass * self._dr_mass_scales[envs_idx],
                links_idx_local=[self._base_link_idx],
                envs_idx=envs_idx,
            )

        if hasattr(self, "_dr_com_shifts"):
            self.drone.set_COM_shift(
                self._dr_com_shifts[envs_idx],
                links_idx_local=[self._base_link_idx],
                envs_idx=envs_idx,
            )

        if hasattr(self, "_dr_kf_scales"):
            self._kf_scales[envs_idx] = self._dr_kf_scales[envs_idx]

        if hasattr(self, "_dr_arm_scales"):
            self._arm_scales[envs_idx] = self._dr_arm_scales[envs_idx]
            lam = self._dr_arm_scales[envs_idx]
            com_shifts = lam[:, None, None] * self._rotor_nominal_pos[None, :, :]
            self.drone.set_COM_shift(
                com_shifts,
                links_idx_local=self._rotor_link_locals,
                envs_idx=envs_idx,
            )

    def _randomize_dynamics(self, envs_idx):
        if self.dr_cfg is None or len(envs_idx) == 0:
            return
        self._apply_dynamics_to_envs(envs_idx)

    def get_dr_params(self):
        params = {}
        if hasattr(self, "_dr_mass_scales"):
            params["mass"] = self._dr_mass_scales.squeeze(-1)
        if hasattr(self, "_dr_inertia_scales"):
            s = self._dr_inertia_scales  # (n, 3)
            params["ixx"] = s[:, 0]
            params["iyy"] = s[:, 1]
            params["izz"] = s[:, 2]
        if hasattr(self, "_dr_com_shifts"):
            shifts = self._dr_com_shifts.squeeze(-2)
            params["com_x"] = shifts[:, 0]
            params["com_y"] = shifts[:, 1]
            params["com_z"] = shifts[:, 2]
        if hasattr(self, "_dr_kf_scales"):
            params["kf"] = self._dr_kf_scales
        if hasattr(self, "_dr_arm_scales"):
            params["arm"] = self._dr_arm_scales
        return params

    def inject_hard_params(self, hard_params_list):
        n_hard = min(len(hard_params_list), self.num_envs)
        if n_hard == 0:
            return n_hard

        for i, hp in enumerate(hard_params_list[:n_hard]):
            if hasattr(self, "_dr_mass_scales"):
                self._dr_mass_scales[i] = hp["mass_scale"]
            if hasattr(self, "_dr_inertia_scales"):
                if "ixx_scale" in hp and "iyy_scale" in hp:
                    izz = hp.get("izz_scale", 1.0)
                    self._dr_inertia_scales[i] = torch.tensor(
                        [hp["ixx_scale"], hp["iyy_scale"], izz],
                        device=gs.device, dtype=gs.tc_float,
                    )
                else:
                    s = hp["inertia_scale"]
                    self._dr_inertia_scales[i] = torch.tensor([s, s, s], device=gs.device, dtype=gs.tc_float)
            if hasattr(self, "_dr_com_shifts"):
                self._dr_com_shifts[i, 0, 0] = hp["com_shift_x"]
                self._dr_com_shifts[i, 0, 1] = hp["com_shift_y"]
                self._dr_com_shifts[i, 0, 2] = hp["com_shift_z"]
            if hasattr(self, "_dr_kf_scales"):
                self._dr_kf_scales[i] = hp["kf_scale"]
            if hasattr(self, "_dr_arm_scales"):
                self._dr_arm_scales[i] = hp["arm_scale"]

        hard_envs = torch.arange(n_hard, device=gs.device)
        self._apply_dynamics_to_envs(hard_envs)
        return n_hard

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        if self._randomize_init_state:
            n = len(envs_idx)
            if self._det_seed is not None:
                # Deterministic init state: k-th reset of env i is fixed.
                k = self._det_ep_count[envs_idx] % self._det_k_eps
                pos_noise = self._det_pos_noise[envs_idx, k]
                yaw = self._det_yaw[envs_idx, k]
                self._det_ep_count[envs_idx] += 1
            else:
                pos_noise = gs_rand_float(-0.05, 0.05, (n, 3), gs.device)
                yaw = gs_rand_float(-math.pi / 6, math.pi / 6, (n,), gs.device)
            self.base_pos[envs_idx] = self.base_pos[envs_idx] + pos_noise
            half_yaw = yaw * 0.5
            yaw_quat = torch.stack([
                torch.cos(half_yaw),
                torch.zeros(n, device=gs.device),
                torch.zeros(n, device=gs.device),
                torch.sin(half_yaw),
            ], dim=-1)
            self.base_quat[envs_idx] = transform_quat_by_quat(
                self.base_quat[envs_idx], yaw_quat
            )

        self.last_base_pos[envs_idx] = self.base_pos[envs_idx]
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    def _reward_target(self):
        return torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw)
        yaw = torch.where(yaw < -180, yaw + 360, yaw)
        yaw = yaw / 180 * math.pi
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        return torch.norm(self.base_ang_vel / math.pi, dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

