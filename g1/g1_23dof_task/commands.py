"""Velocity command that also samples YAW-ONLY commands (vx = vy = 0, wz != 0).

With the stock sampling, vx is uniform in (0, 1], so "turn on the spot" almost never occurs. After stand-still fine-tuning the
policy then treated every command with vx ~ vy ~ 0 as "stand still" and stopped turning in place.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class SparseVelocityCommand(UniformVelocityCommand):
    cfg: SparseVelocityCommandCfg

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pick = torch.rand(len(ids), device=self.device) < self.cfg.rel_yaw_only_envs
        ids = ids[pick]
        if len(ids) == 0:
            return
        lo, hi = self.cfg.yaw_only_magnitude
        mag = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
        sign = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[ids, 0] = 0.0
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = mag * sign
        self.is_heading_env[ids] = False  # yaw rate is the command itself, not derived from a heading target
        self.is_standing_env[ids] = False


@configclass
class SparseVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SparseVelocityCommand
    rel_yaw_only_envs: float = 0.15
    """Fraction of resampled commands that are yaw only (vx = vy = 0)."""
    yaw_only_magnitude: tuple = (0.3, 1.0)
    """|wz| range of those commands (rad/s), kept away from 0 so they are not mistaken for standing."""
