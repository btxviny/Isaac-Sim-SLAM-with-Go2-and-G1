"""Open a scene that contains the ROS policy graph (scene/office.usd) and just run the simulation.
The policy itself runs in policy_ros_node.py (another process); this script only plays the scene and reports the robot pose.

    source ~/isaac-sim-5.1.0/setup_ros_env.sh && cd ~/isaac-sim-5.1.0
    ./python.sh ~/robots_isaac/ros/run_scene.py                         # window
    ./python.sh ~/robots_isaac/ros/run_scene.py --headless --seconds 15 --realtime   # test

--realtime keeps simulation time <= wall time (headless runs would otherwise go much faster than real time, which makes
the policy process look slow in simulation time).
"""

import argparse
import math
import os
import time

ap = argparse.ArgumentParser()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--scene", default=os.path.join(ROOT, "scene", "office.usd"))
ap.add_argument("--robot-path", default="/World/go2/base")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seconds", type=float, default=0.0, help="stop after N simulated seconds (0 = until the window is closed)")
ap.add_argument("--realtime", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.prims import get_prim_at_path  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

for ext in ("omni.graph.action", "omni.graph.scriptnode", "isaacsim.core.nodes", "isaacsim.ros2.bridge"):
    enable_extension(ext)
simulation_app.update()
assert open_stage(args.scene), f"cannot open {args.scene}"
simulation_app.update()

world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
world.reset()


def pose():
    m = UsdGeom.Xformable(get_prim_at_path(args.robot_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    up = m.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
    return np.array(m.ExtractTranslation()), math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))


world.play()
p0, _ = pose()
print(f"[scene] start pos={np.round(p0, 3)}", flush=True)
wall0 = time.perf_counter()
next_report = 1.0
reset_needed = False
while simulation_app.is_running():
    world.step(render=not args.headless)
    if world.is_stopped():
        reset_needed = True
    elif reset_needed and world.is_playing():
        world.reset(True)
        reset_needed = False
    t = world.current_time
    if args.realtime and world.is_playing():
        ahead = t - (time.perf_counter() - wall0)
        if ahead > 0:
            time.sleep(min(ahead, 0.05))
    if args.seconds and world.is_playing():
        if t >= next_report:
            p, tilt = pose()
            print(f"[scene] t={t:5.1f}s pos={np.round(p, 3)} tilt={tilt:5.1f} deg", flush=True)
            next_report += 1.0
        if t >= args.seconds:
            break

if args.seconds:
    p1, tilt1 = pose()
    d = p1 - p0
    print(f"[scene] end displacement={np.round(d, 2)} tilt={tilt1:.1f} deg", flush=True)
    print(f"[scene] RESULT {'UPRIGHT' if tilt1 < 35 and d[2] > -0.20 else 'FELL'}", flush=True)
simulation_app.close()
