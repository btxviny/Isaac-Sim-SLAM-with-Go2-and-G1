"""Registers the 23-joint G1 velocity tasks with gymnasium (import this after the Isaac Sim app has started)."""

import gymnasium as gym

from . import cfg

gym.register(
    id="Isaac-Velocity-Flat-G1-23DoF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:G123DofFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.cfg:G123DofFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-23DoF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:G123DofFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.cfg:G123DofFlatPPORunnerCfg",
    },
)

# stand-still fine-tuning of the walking policy (resume from a walking checkpoint)
gym.register(
    id="Isaac-Velocity-Flat-G1-23DoF-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:G123DofStandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.cfg:G123DofStandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-23DoF-Stand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:G123DofStandFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.cfg:G123DofStandPPORunnerCfg",
    },
)
