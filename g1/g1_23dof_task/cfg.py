"""Isaac Lab flat-terrain velocity task for OUR G1 (scene/g1_23dof/g1_23dof.usd, 23 joints, no fingers).

The stock `Isaac-Velocity-Flat-G1-v0` task trains Isaac Lab's 37-joint `g1_minimal.usd`; a policy trained on it cannot drive
the 23-joint G1 in office.usd. This task is the stock one with the robot, joint-name regexes and gains swapped.
Everything not overridden here (observations, commands, events, terminations, PPO settings) is the stock G1 flat task.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY

USD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scene", "g1_23dof", "g1_23dof.usd")

# Standing pose and pelvis height solved by an earlier PD-gain sweep (stable standing with this USD).
PELVIS_Z = 0.758
POSE = {
    ".*_hip_pitch_joint": -0.324,
    ".*_knee_joint": 0.7,
    ".*_ankle_pitch_joint": -0.376,
    "left_shoulder_roll_joint": 0.3,
    "right_shoulder_roll_joint": -0.3,
    ".*_elbow_joint": 1.5708,
}

# Torque limits are the real ones stored in the USD (hip 88, knee 139, ankle 35, waist 88, arms 25 N*m).
G1_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, PELVIS_Z), joint_pos=POSE, joint_vel={".*": 0.0}),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "waist_yaw_joint"],
            effort_limit_sim={
                ".*_hip_.*": 88.0,
                ".*_knee_joint": 139.0,
                "waist_yaw_joint": 88.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=35.0,
            stiffness=60.0,
            damping=4.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=25.0,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll_joint"],
            effort_limit_sim=25.0,
            stiffness=20.0,
            damping=5.0,
            armature=0.001,
        ),
    },
)


@configclass
class G123DofFlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # joint-name regexes of the stock rewards that do not exist on this robot
        arms = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=arms)
        self.rewards.joint_deviation_fingers = None  # no fingers
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="waist_yaw_joint")


@configclass
class G123DofFlatEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        arms = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=arms)
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="waist_yaw_joint")


@configclass
class G123DofFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_23dof_flat"


# ---------------------------------------------------------------------------------------------------------------
# Stand-still fine-tuning: more zero-command episodes + penalties that only act when the command is zero.
# Same observations / actions / network as the walking task, so it can be resumed from a walking checkpoint.
# ---------------------------------------------------------------------------------------------------------------
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402

from . import mdp_stand  # noqa: E402
from .commands import SparseVelocityCommandCfg  # noqa: E402

LEG_JOINTS = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
FEET = ".*_ankle_roll_link"


def _add_stand_still_rewards(rewards):
    cmd = "base_velocity"
    rewards.stand_still_dev = RewTerm(
        func=mdp_stand.stand_still_joint_deviation_l1,
        weight=-0.4,
        params={"command_name": cmd, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    rewards.stand_still_joint_vel = RewTerm(
        func=mdp_stand.stand_still_joint_vel_l1,
        weight=-0.05,
        params={"command_name": cmd, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    rewards.stand_still_base = RewTerm(
        func=mdp_stand.stand_still_base_motion, weight=-1.0, params={"command_name": cmd}
    )
    rewards.stand_still_feet = RewTerm(
        func=mdp_stand.stand_still_feet_off_ground,
        weight=-1.0,
        params={"command_name": cmd, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
    )


@configclass
class G123DofStandFlatEnvCfg(G123DofFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.commands.base_velocity
        self.commands.base_velocity = SparseVelocityCommandCfg(
            asset_name=old.asset_name,
            resampling_time_range=old.resampling_time_range,
            rel_standing_envs=0.3,  # 30 % of the episodes have an all-zero command
            rel_yaw_only_envs=0.15,  # 15 % turn on the spot (vx = vy = 0)
            rel_heading_envs=old.rel_heading_envs,
            heading_command=old.heading_command,
            heading_control_stiffness=old.heading_control_stiffness,
            debug_vis=old.debug_vis,
            ranges=old.ranges,
        )
        _add_stand_still_rewards(self.rewards)


@configclass
class G123DofStandFlatEnvCfg_PLAY(G123DofFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.5  # half of the robots get a zero command, to watch them stand


@configclass
class G123DofStandPPORunnerCfg(G123DofFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 800
        self.algorithm.learning_rate = 3.0e-4  # gentle: do not forget how to walk
        self.algorithm.entropy_coef = 0.004
