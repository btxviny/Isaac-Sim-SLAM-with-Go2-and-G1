#!/usr/bin/python3
"""ROS 2 side of the split architecture: observation in, joint targets out (runs the TorchScript policy).

    Isaac Sim graph --/<ns>/obs (Float64MultiArray)--> this node --/<ns>/joint_commands (Float64MultiArray)--> Isaac Sim graph
    /cmd_vel (geometry_msgs/Twist) -> velocity command

/<ns>/obs layout (published by sim_obs_node.py every `decimation` physics steps = 50 Hz of SIMULATION time):
    [step, sim_time, base_lin_vel(3), base_ang_vel(3), projected_gravity(3), joint_pos_rel(n), joint_vel_rel(n)]
The policy input is assembled exactly like in training:
    obs = [lin(3), ang(3), gravity(3), command(3), joint_pos_rel(n), joint_vel_rel(n), last_action(n)]
Event driven: one inference per obs message (no wall-clock timer), reply published immediately.
Output: default_joint_pos + action_scale * action, in the articulation's joint order.

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 ~/robots_isaac/ros/policy_ros_node.py --robot go2

One terminal does everything: it runs the policy AND reads keyboard teleop from the terminal (no extra process):
    w / s  or Up / Down      forward / backward  (+-0.1 m/s per press, held until changed)
    a / d                    sideways left / right (+-0.1 m/s per press)
    q / e  or Left / Right   turn left / right    (+-0.2 rad/s per press)
    space                    stop (zero command)
    Ctrl+C                   quit
/cmd_vel (Twist) is still accepted; the most recent source wins. Use --no-keyboard to disable the keyboard.
"""

import argparse
import os
import re
import select
import sys
import termios
import threading
import time
import tty

import warnings

import numpy as np
import rclpy
import torch
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

warnings.filterwarnings("ignore", message=".*torch.jit.load.*")  # TorchScript load is what Isaac Lab exports

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# joint order of the articulation (must equal the training order; verified for the Go2)
JOINTS = {
    "go2": [f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")],
}
RUN_DIR = {"go2": os.path.join(ROOT, "go2", "policy"), "g1": os.path.join(ROOT, "g1", "policy")}


def load_env_yaml(path):
    class L(yaml.SafeLoader):
        pass

    def generic(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return None

    L.add_multi_constructor("tag:yaml.org,2002:python/", generic)
    with open(path) as f:
        return yaml.load(f, Loader=L)


def per_joint(values, names, default=0.0):
    out = []
    for n in names:
        v = default
        if isinstance(values, (int, float)):
            v = float(values)
        elif isinstance(values, dict):
            for pat, val in values.items():
                if re.fullmatch(pat, n):
                    v = float(val)
        out.append(v)
    return np.array(out, dtype=np.float32)


class PolicyNode(Node):
    def __init__(self, args):
        super().__init__("policy_ros_node")
        run = args.run or RUN_DIR[args.robot]
        env = load_env_yaml(run + "/params/env.yaml")
        names = args.joint_names.split(",") if args.joint_names else JOINTS[args.robot]
        self.n = len(names)
        self.default = per_joint(env["scene"]["robot"]["init_state"]["joint_pos"], names)
        self.scale = float(env["actions"]["joint_pos"]["scale"])
        r = env["commands"]["base_velocity"]["ranges"]
        self.lo = np.array([r["lin_vel_x"][0], r["lin_vel_y"][0], r["ang_vel_z"][0]], dtype=np.float32)
        self.hi = np.array([r["lin_vel_x"][1], r["lin_vel_y"][1], r["ang_vel_z"][1]], dtype=np.float32)
        self.decimation = int(env["decimation"])
        self.policy = torch.jit.load(run + "/exported/policy.pt", map_location="cpu").eval()
        torch.set_num_threads(1)

        self.cmd = np.zeros(3, dtype=np.float32)
        self.cmd_time = 0.0
        self.kb_enabled = args.keyboard and sys.stdin.isatty()
        self.kb_step = np.array([0.1, 0.1, 0.2], dtype=np.float32)  # m/s, m/s, rad/s per key press
        self.cmd_timeout = args.cmd_timeout
        self.last_action = np.zeros(self.n, dtype=np.float32)
        self.stats = {"n": 0, "lat": 0.0, "t0": time.time(), "last_step": None, "gaps": 0, "sim_t": 0.0}

        ns = "/" + args.ns.strip("/")
        self.pub = self.create_publisher(Float64MultiArray, ns + "/joint_commands", 10)
        self.create_subscription(Float64MultiArray, ns + "/obs", self.on_obs, 10)
        self.create_subscription(Twist, args.cmd_topic, self.on_cmd, 10)
        self.create_timer(1.0 if self.kb_enabled else 5.0, self.report)
        self.get_logger().info(
            f"policy {run}\n  {self.n} joints, scale {self.scale}, default {np.round(self.default, 2).tolist()}\n"
            f"  command ranges lo={self.lo.tolist()} hi={self.hi.tolist()}\n  obs <- {ns}/obs   targets -> {ns}/joint_commands   cmd <- {args.cmd_topic}"
        )

    # ---------------------------------------------------------------- keyboard teleop (terminal, cbreak mode)
    def start_keyboard(self):
        self._fd = sys.stdin.fileno()
        self._old_tty = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # keys arrive immediately, Ctrl+C still works
        self._kb_run = True
        threading.Thread(target=self._kb_loop, daemon=True).start()
        print(self.HELP, flush=True)

    def stop_keyboard(self):
        self._kb_run = False
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_tty)
        except Exception:
            pass

    HELP = (
        "\r\nkeyboard teleop:  w/s forward/back  a/d sideways  q/e turn (or arrows: up/down, left/right)  space stop  Ctrl+C quit\r\n"
    )
    KEYS = {  # key -> (index of [vx, vy, wz], direction)
        "w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "q": (2, 1), "e": (2, -1),
        "\x1b[A": (0, 1), "\x1b[B": (0, -1), "\x1b[D": (2, 1), "\x1b[C": (2, -1),
    }

    def _kb_loop(self):
        while self._kb_run:
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            data = os.read(self._fd, 8).decode(errors="ignore")
            i = 0
            while i < len(data):
                key = data[i : i + 3] if data[i] == "\x1b" else data[i]
                i += len(key)
                if key == " ":
                    self.cmd[:] = 0.0
                elif key.lower() in self.KEYS or key in self.KEYS:
                    idx, sign = self.KEYS.get(key, self.KEYS.get(key.lower()))
                    self.cmd[idx] += sign * self.kb_step[idx]
                    self.cmd[:] = np.clip(self.cmd, self.lo, self.hi)
                else:
                    continue
                self.cmd_time = float("inf")  # keyboard commands stay until changed

    def on_cmd(self, msg):
        self.cmd = np.clip(np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32), self.lo, self.hi)
        self.cmd_time = time.time()

    def on_obs(self, msg):
        t0 = time.perf_counter()
        d = np.asarray(msg.data, dtype=np.float32)
        step, sim_t, body = d[0], d[1], d[2:]
        if int(step) % self.decimation != 0:  # the sim side already decimates; this guards against a 200 Hz publisher
            return
        if body.size != 9 + 2 * self.n:
            self.get_logger().error(f"obs has {body.size} numbers, expected {9 + 2 * self.n}", throttle_duration_sec=2.0)
            return
        if self.stats["last_step"] is not None and step < self.stats["last_step"]:
            self.last_action[:] = 0.0  # simulation was stopped / restarted
        self.stats["last_step"] = step
        cmd = self.cmd if (time.time() - self.cmd_time) < self.cmd_timeout else np.zeros(3, dtype=np.float32)
        cmd = np.where(np.abs(cmd) < 0.05, 0.0, cmd).astype(np.float32)  # dead-band: tiny commands -> exact zero (= stand still)
        lin, ang, grav = body[0:3], body[3:6], body[6:9]
        qrel, qd = body[9 : 9 + self.n], body[9 + self.n :]
        obs = np.concatenate([lin, ang, grav, cmd, qrel, qd, self.last_action]).astype(np.float32)
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(obs).view(1, -1)).view(-1).numpy()
        self.last_action = action
        self.pub.publish(Float64MultiArray(data=(self.default + self.scale * action).astype(np.float64).tolist()))
        self.stats["n"] += 1
        self.stats["lat"] += time.perf_counter() - t0
        self.stats["sim_t"] = float(sim_t)

    def report(self):
        s = self.stats
        dt = time.time() - s["t0"]
        if self.kb_enabled:
            state = f"{s['n'] / dt:4.0f} Hz" if s["n"] else "waiting for sim (press Play)"
            print(f"\r\x1b[2K[{state}, sim_t={s['sim_t']:.1f}s] cmd vx={self.cmd[0]:+.1f} vy={self.cmd[1]:+.1f} wz={self.cmd[2]:+.1f}", end="", flush=True)
        elif s["n"]:
            self.get_logger().info(
                f"{s['n'] / dt:5.1f} Hz wall, inference {1000 * s['lat'] / s['n']:.2f} ms, sim_t={s['sim_t']:.1f}s, cmd={np.round(self.cmd, 2).tolist()}"
            )
        else:
            self.get_logger().info("waiting for observations (is the simulation playing?)")
        s.update(n=0, lat=0.0, t0=time.time())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="go2", choices=list(RUN_DIR))
    ap.add_argument("--run", default=None, help="policy folder (default: <robot>/policy)")
    ap.add_argument("--ns", default=None, help="topic namespace (default: the robot name)")
    ap.add_argument("--joint-names", default=None, help="comma list, articulation order (needed for g1)")
    ap.add_argument("--cmd-topic", default="/cmd_vel")
    ap.add_argument("--no-keyboard", dest="keyboard", action="store_false", help="do not read teleop keys from this terminal")
    ap.add_argument("--cmd-timeout", type=float, default=0.5, help="seconds without /cmd_vel before the command is zeroed")
    args, ros_args = ap.parse_known_args()
    args.ns = args.ns or args.robot
    if args.robot not in JOINTS and not args.joint_names:
        sys.exit("--joint-names is required for this robot")
    rclpy.init(args=ros_args)
    node = PolicyNode(args)
    if node.kb_enabled:
        node.start_keyboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.kb_enabled:
        node.stop_keyboard()
        print("")
    node.destroy_node()
    if rclpy.ok():  # Ctrl+C may already have shut the context down
        rclpy.shutdown()


if __name__ == "__main__":
    main()
