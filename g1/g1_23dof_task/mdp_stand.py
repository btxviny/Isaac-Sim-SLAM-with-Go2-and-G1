"""Reward terms that only act while the velocity command is (almost) zero, to teach the G1 to stand still.

The stock tasks use `stand_still_joint_deviation_l1`, whose mask looks only at the xy command, so a pure yaw command
would count as "standing". These versions use all three command components (vx, vy, wz).
"""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _standing(env, command_name: str, threshold: float) -> torch.Tensor:
    """True for environments whose commanded (vx, vy, wz) is ~0."""
    cmd = env.command_manager.get_command(command_name)
    return torch.norm(cmd[:, :3], dim=1) < threshold


def stand_still_joint_deviation_l1(
    env, command_name: str, asset_cfg: SceneEntityCfg, threshold: float = 0.06
) -> torch.Tensor:
    """Penalize offsets from the default joint positions while the command is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(dev), dim=1) * _standing(env, command_name, threshold)


def stand_still_joint_vel_l1(
    env, command_name: str, asset_cfg: SceneEntityCfg, threshold: float = 0.06
) -> torch.Tensor:
    """Penalize joint velocities (marching in place) while the command is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * _standing(env, command_name, threshold)


def stand_still_base_motion(
    env, command_name: str, threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize base linear (xy) and yaw velocity while the command is zero (squared)."""
    asset: Articulation = env.scene[asset_cfg.name]
    motion = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) + torch.square(asset.data.root_ang_vel_b[:, 2])
    return motion * _standing(env, command_name, threshold)


def stand_still_feet_off_ground(
    env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.06, force_threshold: float = 1.0
) -> torch.Tensor:
    """Number of feet that are not touching the ground while the command is zero."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > force_threshold
    return torch.sum((~in_contact).float(), dim=1) * _standing(env, command_name, threshold)
