"""Reproduction protocol and run-path configuration for TILT.

This module contains paper-protocol defaults and validates user-supplied run
metadata. It deliberately does not own algorithm hyperparameters or simulator
configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TiltProtocol:
    seed: int = 1
    num_envs: int = 1024
    num_eval_envs: int = 4096
    iterations_per_round: int = 50
    total_rounds: int = 20
    episodes_per_eval: int = 15
    final_eval_episodes: int = 10
    eval_seed: int = 42
    eval_max_steps: int = 1500

    def validate(self) -> "TiltProtocol":
        for name in (
            "num_envs", "num_eval_envs", "iterations_per_round", "total_rounds",
            "episodes_per_eval", "final_eval_episodes", "eval_max_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return self

    @property
    def run_name(self) -> str:
        return f"TILT_s{self.seed}"

    @property
    def default_log_dir(self) -> Path:
        return Path("logs") / self.run_name

    @property
    def final_checkpoint(self) -> Path:
        return self.default_log_dir / f"model_{self.total_rounds * self.iterations_per_round}.pt"

    def resolve_log_dir(self, override: str | None = None) -> Path:
        path = Path(override).expanduser() if override else self.default_log_dir
        return path
