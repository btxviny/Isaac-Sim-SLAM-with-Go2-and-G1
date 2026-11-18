"""Run a trained Isaac Lab velocity policy (Go2, G1, any robot with the stock observation layout) inside Isaac Sim (no Isaac Lab needed at runtime).

The controller subscribes to PhysX step events, so it runs at the physics rate (200 Hz) and queries the policy every
`decimation` steps (4 -> 50 Hz), exactly like training. Everything robot-specific (default pose, PD gains, effort limit,
action scale, decimation, command ranges) is read from the `env.yaml` that was saved with the policy.

Used directly from Python.

Command sources, in priority order:
  1. env var GO2_TEST_CMD="vx,vy,wz"   (headless testing)
  2. ROS 2 Subscribe Twist node (optional, see `twist_node_path`)  -> /cmd_vel
  3. keyboard: Up/Down = forward/back, Left/Right = sideways, Z/X = yaw left/right
"""

import os
import re

import numpy as np
import yaml


def _load_env_yaml(path):
    """env.yaml contains python-specific tags (tuples, slices); load them as plain lists/None."""

    class _Loader(yaml.SafeLoader):
        pass

    def _generic(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return None

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _generic)
    with open(path, "r") as f:
        return yaml.load(f, Loader=_Loader)


def _per_joint(values, joint_names, default=None):
    """Expand a float / {regex: value} entry to one value per joint name (regex full match)."""
    out = []
    for name in joint_names:
        v = default
        if isinstance(values, (int, float)):
            v = float(values)
        elif isinstance(values, dict):
            for pattern, val in values.items():
                if re.fullmatch(pattern, name):
                    v = float(val)
        out.append(v)
    return out


def _quat_to_rot(q):
    """Rotation matrix (body -> world) from a (w, x, y, z) quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class VelocityPolicyController:
    def __init__(
        self, robot_path, policy_path, env_path, twist_node_path=None, log=print, use_keyboard=True, fixed_cmd=None, name=None
    ):
        self.use_keyboard = use_keyboard  # False: this robot ignores the keyboard (e.g. the other robot in the scene)
        self.fixed_cmd = None if fixed_cmd is None else np.asarray(fixed_cmd, dtype=np.float32)
        self.name = name or os.path.basename(robot_path.rstrip("/"))
        self.robot_path = robot_path
        self.policy_path = policy_path
        self.env_path = env_path
        self.twist_node_path = twist_node_path
        self.log = log

        env = _load_env_yaml(env_path)
        self._env = env
        self.decimation = int(env["decimation"])
        self.dt = float(env["sim"]["dt"])
        act = env["actions"]["joint_pos"]
        self.action_scale = float(act["scale"])
        r = env["commands"]["base_velocity"]["ranges"]
        self.cmd_lo = np.array([r["lin_vel_x"][0], r["lin_vel_y"][0], r["ang_vel_z"][0]], dtype=np.float32)
        self.cmd_hi = np.array([r["lin_vel_x"][1], r["lin_vel_y"][1], r["ang_vel_z"][1]], dtype=np.float32)

        self.policy = None
        self.torch = None
        if policy_path is not None:  # None: observation / state side only (the policy runs in another process)
            import torch

            self.torch = torch
            self.policy = torch.jit.load(policy_path, map_location="cpu")
            self.policy.eval()

        self.robot = None
        self._inited = False
        self._counter = 0
        self._warned_dt = False
        self._phys_sub = None
        self._tl_sub = None
        self._kb_sub = None
        self._keys = set()
        self.status = "stopped"

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        import carb.input
        import omni.appwindow
        import omni.physx
        import omni.timeline

        self._phys_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_physics_step)
        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._tl_sub = stream.create_subscription_to_pop(self._on_timeline)
        try:
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._kb_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        except Exception as e:  # headless: no window / keyboard
            self.log(f"[velocity_policy] keyboard unavailable ({e})")
        self.status = "waiting for Play"
        self.log(f"[velocity_policy] ready: policy={os.path.basename(self.policy_path)} robot={self.robot_path}")

    def stop(self):
        self._phys_sub = None
        self._tl_sub = None
        if self._kb_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._kb_sub)
            except Exception:
                pass
            self._kb_sub = None
        self.status = "stopped"

    def _on_timeline(self, event):
        import omni.timeline

        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # the physics view is destroyed on stop; rebuild everything on the next play
            self._inited = False
            self.robot = None
            self._counter = 0
            self.status = "waiting for Play"

    # ------------------------------------------------------------------ setup (first physics step after Play)
    def _init_robot(self):
        from isaacsim.core.prims import SingleArticulation
        from omni.physx import get_physx_simulation_interface

        self.robot = SingleArticulation(prim_path=self.robot_path, name="velocity_policy_robot")
        # physics view: the one Isaac Sim's SimulationManager keeps (GUI / World), else create our own (bare headless app)
        sim_view = None
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            sim_view = SimulationManager.get_physics_sim_view()
        except Exception:
            pass
        if sim_view is None:
            import omni.physics.tensors as tensors

            sim_view = tensors.create_simulation_view("numpy")
        self.robot.initialize(physics_sim_view=sim_view)
        names = list(self.robot.dof_names)
        n = len(names)
        robot_cfg = self._env["scene"]["robot"]

        # default pose / velocity
        self.default_pos = np.array(_per_joint(robot_cfg["init_state"]["joint_pos"], names, 0.0), dtype=np.float32)
        # PD gains and torque limit (all actuator groups)
        kp = np.zeros(n)
        kd = np.zeros(n)
        eff = np.full(n, 1e6)
        arm = np.full(n, np.nan)
        for grp in robot_cfg["actuators"].values():
            sel = [any(re.fullmatch(p, nm) for p in grp["joint_names_expr"]) for nm in names]
            kps = _per_joint(grp["stiffness"], names, 0.0)
            kds = _per_joint(grp["damping"], names, 0.0)
            lim = grp.get("effort_limit_sim")
            effs = _per_joint(lim if lim is not None else grp.get("effort_limit"), names, None)
            arms = _per_joint(grp.get("armature"), names, None)
            for i, s in enumerate(sel):
                if s:
                    kp[i], kd[i] = kps[i], kds[i]
                    if effs[i] is not None:
                        eff[i] = effs[i]
                    if arms[i] is not None:
                        arm[i] = arms[i]
        view = self.robot._articulation_view
        ctrl = self.robot.get_articulation_controller()
        ctrl.set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        ctrl.switch_control_mode("position")
        view.set_gains(kp.reshape(1, -1), kd.reshape(1, -1))
        view.set_max_efforts(eff.reshape(1, -1))
        if not np.all(np.isnan(arm)):
            view.set_armatures(np.nan_to_num(arm, nan=0.0).reshape(1, -1))  # training used these armatures
        get_physx_simulation_interface().flush_changes()

        # articulation solver settings used in training (Isaac Lab overrides the USD's own values, e.g. 8/4 instead of 32/1)
        ap = (robot_cfg.get("spawn") or {}).get("articulation_props") or {}
        for key, setter in (
            ("solver_position_iteration_count", self.robot.set_solver_position_iteration_count),
            ("solver_velocity_iteration_count", self.robot.set_solver_velocity_iteration_count),
            ("stabilization_threshold", self.robot.set_stabilization_threshold),
            ("sleep_threshold", self.robot.set_sleep_threshold),
        ):
            v = ap.get(key)
            if v is not None and v != float("inf"):
                setter(v)
        if isinstance(ap.get("enabled_self_collisions"), bool):
            self.robot.set_enabled_self_collisions(ap["enabled_self_collisions"])

        # start from the pose the policy was trained from
        self.robot.set_joint_positions(self.default_pos)
        self.robot.set_joint_velocities(np.zeros(n))
        self.action = np.zeros(n, dtype=np.float32)
        self._counter = 0
        self._inited = True
        self.status = "running"
        self.log(f"[{self.name}] {n} joints: {names}")
        self.log(f"[{self.name}] kp={np.round(kp[[0, -1]], 1)} kd={np.round(kd[[0, -1]], 2)} max_effort={np.round(eff[[0, -1]], 1)} "
                 f"scale={self.action_scale} decimation={self.decimation} dt={self.dt} (first/last joint)")

    # ------------------------------------------------------------------ commands
    def _on_key(self, event, *args):
        import carb.input

        K = carb.input.KeyboardInput
        name = event.input
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            self._keys.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys.discard(name)
        return True  # do not swallow the event

    def _keyboard_command(self):
        import carb.input

        K = carb.input.KeyboardInput
        k = self._keys
        vx = (0.8 if K.UP in k else 0.0) - (0.8 if K.DOWN in k else 0.0)
        vy = (0.5 if K.LEFT in k else 0.0) - (0.5 if K.RIGHT in k else 0.0)
        wz = (1.0 if K.Z in k else 0.0) - (1.0 if K.X in k else 0.0)
        return np.array([vx, vy, wz], dtype=np.float32)

    CMD_DEADBAND = 0.05  # m/s or rad/s

    def command(self):
        test = os.environ.get("GO2_TEST_CMD")
        if self.fixed_cmd is not None:
            cmd = self.fixed_cmd.copy()
        elif test:
            cmd = np.array([float(v) for v in test.split(",")], dtype=np.float32)
        else:
            cmd = self._keyboard_command() if self.use_keyboard else np.zeros(3, dtype=np.float32)
            if self.twist_node_path:
                try:
                    import omni.graph.core as og

                    lin = og.Controller.attribute(self.twist_node_path + ".outputs:linearVelocity").get()
                    ang = og.Controller.attribute(self.twist_node_path + ".outputs:angularVelocity").get()
                    ros = np.array([lin[0], lin[1], ang[2]], dtype=np.float32)
                    if np.any(ros != 0.0):
                        cmd = ros
                except Exception:
                    pass
        cmd = np.clip(cmd, self.cmd_lo, self.cmd_hi)
        cmd[np.abs(cmd) < self.CMD_DEADBAND] = 0.0  # tiny commands become the exact zero the policy was trained to stand still on
        return cmd

    # ------------------------------------------------------------------ ROS-split helpers (no policy in this process)
    def ensure_init(self):
        """Create / re-create the articulation view (needed after every Stop -> Play) and set gains from env.yaml."""
        if not self._inited:
            self._init_robot()
        return self._inited

    def invalidate(self):
        self._inited = False
        self.robot = None

    def observe_partial(self):
        """Everything of the policy observation that comes from the simulator, in training order:
        base_lin_vel(3), base_ang_vel(3), projected_gravity(3), joint_pos_rel(n), joint_vel_rel(n).
        The command (3) and last action (n) are inserted by the policy process: obs = [lin, ang, grav, cmd, qrel, qd, last_action]."""
        pos, quat = self.robot.get_world_pose()
        R_bw = _quat_to_rot(np.asarray(quat, dtype=np.float64)).T
        lin_b = R_bw @ np.asarray(self.robot.get_linear_velocity())
        ang_b = R_bw @ np.asarray(self.robot.get_angular_velocity())
        grav_b = R_bw @ np.array([0.0, 0.0, -1.0])
        q = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        qd = np.asarray(self.robot.get_joint_velocities(), dtype=np.float64)
        return np.concatenate([lin_b, ang_b, grav_b, q - self.default_pos, qd])

    # ------------------------------------------------------------------ control loop (every physics step)
    def _observation(self, cmd):
        pos, quat = self.robot.get_world_pose()
        R_bw = _quat_to_rot(np.asarray(quat, dtype=np.float64)).T  # world -> body
        lin_b = R_bw @ np.asarray(self.robot.get_linear_velocity())
        ang_b = R_bw @ np.asarray(self.robot.get_angular_velocity())
        grav_b = R_bw @ np.array([0.0, 0.0, -1.0])
        q = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        qd = np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)
        # order matches training: base_lin_vel, base_ang_vel, projected_gravity, command, joint_pos_rel, joint_vel_rel, last_action
        return np.concatenate([lin_b, ang_b, grav_b, cmd, q - self.default_pos, qd, self.action]).astype(np.float32)

    def _on_physics_step(self, dt):
        try:
            if not self._inited:
                self._init_robot()
                return
            if not self._warned_dt and abs(dt - self.dt) > 1e-4:
                self._warned_dt = True
                self.log(f"[velocity_policy] WARNING: physics dt is {dt:.5f}, the policy was trained at {self.dt}")
            if self._counter % self.decimation == 0:
                obs = self._observation(self.command())
                with self.torch.no_grad():
                    self.action = self.policy(self.torch.from_numpy(obs).view(1, -1)).view(-1).numpy()
            from isaacsim.core.utils.types import ArticulationAction

            target = self.default_pos + self.action * self.action_scale
            self.robot.apply_action(ArticulationAction(joint_positions=target))
            self._counter += 1
        except Exception as e:  # never let an exception kill the physics callback silently
            self.status = f"error: {e}"
            self.log(f"[velocity_policy] ERROR in physics step: {e!r}")
            self._phys_sub = None


# the controller is robot-agnostic: it works for any Isaac Lab velocity task with the stock observation layout
Go2PolicyController = VelocityPolicyController  # old name
