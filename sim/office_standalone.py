"""Run trained velocity policies (Go2 and/or our 23-joint G1) in scene/office.usd, modelled on Isaac Sim's h1_standalone.py.

Everything in ONE process: this script loads the TorchScript policy and steps it inside Isaac Sim (no ROS for the
policy). The scene's /World/PolicyGraph (the ROS-split policy, see ros/) is switched off so it does not fight over the
Go2's joints; the sensor graph /World/ActionGraph keeps publishing.

    source ~/isaac-sim-5.1.0/setup_ros_env.sh          # ROS 2 bridge libraries (do NOT source /opt/ros first)
    cd ~/isaac-sim-5.1.0
    ./python.sh ~/robots_isaac/sim/office_standalone.py --robots go2,g1 --drive g1     # window, keyboard drives the G1
    ./python.sh ~/robots_isaac/sim/office_standalone.py --robots g1 --headless --seconds 10 --cmd 0.5,0,0   # test

Keys (click the viewport first): Up/Down forward/back, Left/Right sideways, Z/X yaw left/right.
Robots that are not --drive'n get a zero command (they stand / march in place). The scene's lidar / RealSense ROS 2
graph keeps running (topics /point_cloud /rgb /depth ...).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# name -> (articulation root in office.usd, policy folder)
ROBOTS = {
    "go2": ("/World/go2/base", os.path.join(ROOT, "go2", "policy")),
    "g1": ("/World/g1_23dof/pelvis", os.path.join(ROOT, "g1", "policy")),
}


parser = argparse.ArgumentParser()
parser.add_argument("--scene", default=os.path.join(os.path.dirname(HERE), "scene", "office.usd"))
parser.add_argument("--robots", default="go2", help="comma list of: " + ", ".join(ROBOTS))
parser.add_argument("--drive", default=None, help="robot the keyboard / --cmd controls (default: the first one)")
parser.add_argument("--run", action="append", default=[], metavar="NAME=FOLDER", help="other policy folder, e.g. g1=/path/to/run")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after N simulated seconds (0 = run until closed)")
parser.add_argument("--cmd", default=None, help="fixed command 'vx,vy,wz' for the driven robot instead of the keyboard")
parser.add_argument("--no-ros", action="store_true", help="do not load the ROS 2 bridge")
parser.add_argument("--no-thick-floor", dest="thick_floor", action="store_false", help="do not add the thick box collider under the carpet (the carpet itself is a zero-thickness mesh and the G1 falls through its contacts)")
parser.add_argument("--ground-friction", type=float, default=1.0, help="friction of the office floor (training used 1.0; 0 = leave the scene's own)")
parser.add_argument("--robot-friction", type=float, default=0.8, help="friction of the robots' links (training used 0.8; 0 = leave the USD's own)")
args = parser.parse_args()

names = [n.strip() for n in args.robots.split(",") if n.strip()]
for n in names:
    if n not in ROBOTS:
        raise SystemExit(f"unknown robot {n!r}; choose from {list(ROBOTS)}")
drive = args.drive or names[0]
runs = dict(r.split("=", 1) for r in args.run)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import math  # noqa: E402

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.prims import get_prim_at_path  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

sys.path.insert(0, HERE)
from velocity_policy import VelocityPolicyController  # noqa: E402

enable_extension("omni.graph.action")
if not args.no_ros:
    enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

if not open_stage(args.scene):
    raise RuntimeError(f"could not open {args.scene}")
simulation_app.update()

_graph = get_prim_at_path("/World/PolicyGraph")
if _graph and _graph.IsValid():
    _graph.SetActive(False)  # the ROS-split policy graph; this script runs the policies itself
    simulation_app.update()


def bind_friction(prim_path, friction, name):
    """Give a prim (and its colliders below it) a physics material with this friction, combined by multiplication like in training."""
    from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    mat_path = f"/World/Physics_Materials/{name}"
    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr("multiply")
    px.CreateRestitutionCombineModeAttr("multiply")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants, "physics"
    )


import omni.usd  # noqa: E402

if args.thick_floor:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    _top = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(_stage.GetPrimAtPath("/World/floor_carpet")).ComputeAlignedRange().GetMax()[2]
    _box = UsdGeom.Cube.Define(_stage, "/World/thick_floor")
    _box.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(_box).SetScale((60.0, 60.0, 1.0))
    UsdGeom.XformCommonAPI(_box).SetTranslate((0.0, 0.0, _top - 0.5 - 0.001))  # top face 1 mm below the carpet
    UsdGeom.Imageable(_box).CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(_box.GetPrim())
    bind_friction("/World/thick_floor", args.ground_friction if args.ground_friction > 0 else 1.0, "thick_floor_friction")
    print(f"[office] thick floor collider, top at z={_top - 0.001:.3f}", flush=True)
if args.ground_friction > 0:
    bind_friction("/World/floor_carpet", args.ground_friction, "ground_friction")
if args.robot_friction > 0:
    for n in names:
        root = ROBOTS[n][0].rsplit("/", 1)[0]  # e.g. /World/g1_23dof (parent of the articulation root)
        bind_friction(root, args.robot_friction, f"{n}_friction")

# same timing as training: physics 200 Hz (policy every 4th step = 50 Hz); render every 8th step as in the H1 example
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world.reset()

fixed = [float(v) for v in args.cmd.split(",")] if args.cmd else None
controllers = {}
for n in names:
    path, folder = ROBOTS[n]
    run = runs.get(n) or folder
    c = VelocityPolicyController(
        path,
        run + "/exported/policy.pt",
        run + "/params/env.yaml",
        log=lambda m: print(m, flush=True),
        use_keyboard=(n == drive),
        fixed_cmd=fixed if n == drive else ([0.0, 0.0, 0.0] if fixed is not None else None),
        name=n,
    )
    c.start()
    controllers[n] = c
    print(f"[office] {n}: policy {run}", flush=True)


def base_pose(path):
    m = UsdGeom.Xformable(get_prim_at_path(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    up = m.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
    return np.array(m.ExtractTranslation()), math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))


start = {n: base_pose(ROBOTS[n][0])[0] for n in names}
print(f"[office] start {({n: np.round(p, 3).tolist() for n, p in start.items()})} drive={drive} cmd={args.cmd or 'keyboard'}", flush=True)

my_world.play()  # start straight away (press Stop / Play in the GUI to reset)
next_report = 1.0
reset_needed = False
qd_sum = {n: 0.0 for n in names}  # for the stand-still check: mean |joint velocity| and the length of the path the base walked
qd_cnt = {n: 0 for n in names}
path_len = {n: 0.0 for n in names}
last_xy = {n: start[n][:2].copy() for n in names}
while simulation_app.is_running():
    my_world.step(render=not args.headless)
    if args.seconds and my_world.is_playing():
        for n in names:
            c = controllers[n]
            if c._inited:
                qd_sum[n] += float(np.abs(np.asarray(c.robot.get_joint_velocities())).mean())
                qd_cnt[n] += 1
    if my_world.is_stopped():
        reset_needed = True
    elif reset_needed and my_world.is_playing():
        my_world.reset(True)  # back to the initial state; the controllers re-initialize themselves
        reset_needed = False
    if args.seconds and my_world.is_playing():
        t = my_world.current_time
        if t >= next_report:
            for n in names:
                p, tilt = base_pose(ROBOTS[n][0])
                path_len[n] += float(np.linalg.norm(p[:2] - last_xy[n]))
                last_xy[n] = p[:2].copy()
                print(f"[office] t={t:5.1f}s {n}: pos={np.round(p, 3)} tilt={tilt:5.1f} deg", flush=True)
            next_report += 1.0
        if t >= args.seconds:
            break

if args.seconds:
    for n in names:
        p, tilt = base_pose(ROBOTS[n][0])
        d = p - start[n]
        ok = tilt < 35.0 and d[2] > -0.20
        print(f"[office] end {n}: displacement={np.round(d, 2)} tilt={tilt:.1f} deg controller={controllers[n].status}", flush=True)
        print(f"[office] end {n}: base path length={path_len[n]:.2f} m, mean |joint velocity|={qd_sum[n] / max(qd_cnt[n], 1):.3f} rad/s "
              f"(stand-still check: path ~ 0 and joint velocity near 0)", flush=True)
        print(f"[office] RESULT {n} {'UPRIGHT' if ok else 'FELL'}", flush=True)

for c in controllers.values():
    c.stop()
simulation_app.close()
