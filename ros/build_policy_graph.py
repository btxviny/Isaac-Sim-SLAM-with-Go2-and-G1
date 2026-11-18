"""(Re)build the ROS-split policy graph /World/PolicyGraph in scene/office.usd (an existing graph is replaced).
Use --out to write a copy instead of saving in place.

    source ~/isaac-sim-5.1.0/setup_ros_env.sh && cd ~/isaac-sim-5.1.0
    ./python.sh ~/robots_isaac/ros/build_policy_graph.py [--robot-path /World/go2/base] [--ns go2]

Graph /World/PolicyGraph:
    On Physics Step --step--> Script Node (sim_obs_node.py) --execOut--> ROS2 Publisher   /<ns>/obs
    On Physics Step --step--> ROS2 Subscriber /<ns>/joint_commands --data--> Articulation Controller (position targets)
"""

import argparse
import os

ap = argparse.ArgumentParser()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap.add_argument("--src", default=os.path.join(ROOT, "scene", "office.usd"))
ap.add_argument("--out", default=os.path.join(ROOT, "scene", "office.usd"))
ap.add_argument("--robot-path", default="/World/go2/base")
ap.add_argument("--ns", default="go2")
ap.add_argument("--graph", default="/World/PolicyGraph")
ap.add_argument("--no-thick-floor", dest="thick_floor", action="store_false",
                help="do not add a thick box collider under the carpet")
ap.add_argument("--script", default=os.path.join(HERE, "sim_obs_node.py"))
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402
from pxr import Sdf  # noqa: E402

for ext in ("omni.graph.action", "omni.graph.scriptnode", "isaacsim.core.nodes", "isaacsim.ros2.bridge"):
    enable_extension(ext)


def pump(n=10):
    for _ in range(n):
        app.update()


pump()
import omni.graph.core as og  # noqa: E402
import omni.graph.tools.ogn as ogn  # noqa: E402

assert open_stage(args.src), f"cannot open {args.src}"
pump()
stage = omni.usd.get_context().get_stage()
if stage.GetPrimAtPath(args.graph).IsValid():
    print(f"[build] replacing the existing {args.graph}", flush=True)
    stage.RemovePrim(args.graph)
    pump()

G = args.graph
keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {
        "graph_path": G,
        "evaluator_name": "execution",
        # On Physics Step only fires in an on-demand graph
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("on_physics_step", "isaacsim.core.nodes.OnPhysicsStep"),
            ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
            ("obs_node", "omni.graph.scriptnode.ScriptNode"),
            ("obs_publisher", "isaacsim.ros2.bridge.ROS2Publisher"),
            ("cmd_subscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
            ("articulation_controller", "isaacsim.core.nodes.IsaacArticulationController"),
        ],
        keys.SET_VALUES: [
            ("obs_node.inputs:usePath", True),
            ("obs_node.inputs:scriptPath", args.script),
            ("obs_publisher.inputs:messagePackage", "std_msgs"),
            ("obs_publisher.inputs:messageName", "Float64MultiArray"),
            ("obs_publisher.inputs:topicName", f"{args.ns}/obs"),
            ("cmd_subscriber.inputs:messagePackage", "std_msgs"),
            ("cmd_subscriber.inputs:messageName", "Float64MultiArray"),
            ("cmd_subscriber.inputs:topicName", f"{args.ns}/joint_commands"),
            ("articulation_controller.inputs:robotPath", args.robot_path),
        ],
    },
)
pump()

# the Script Node's custom output (the observation) and its self-managed execution output
obs_node = og.get_node_by_path(f"{G}/obs_node")
og.Controller.create_attribute(
    obs_node, "outputs:obs", og.Type(og.BaseDataType.DOUBLE, 1, 1), og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT
)
obs_node.get_attribute("outputs:execOut").set_metadata(ogn.MetadataKeys.HIDDEN, "1")
pump()

og.Controller.edit(
    graph,
    {
        keys.CONNECT: [
            (f"{G}/on_physics_step.outputs:step", f"{G}/obs_node.inputs:execIn"),
            (f"{G}/obs_node.outputs:execOut", f"{G}/obs_publisher.inputs:execIn"),
            (f"{G}/obs_node.outputs:obs", f"{G}/obs_publisher.inputs:data"),
            (f"{G}/ros2_context.outputs:context", f"{G}/obs_publisher.inputs:context"),
            (f"{G}/on_physics_step.outputs:step", f"{G}/cmd_subscriber.inputs:execIn"),
            (f"{G}/ros2_context.outputs:context", f"{G}/cmd_subscriber.inputs:context"),
            (f"{G}/on_physics_step.outputs:step", f"{G}/articulation_controller.inputs:execIn"),
            (f"{G}/cmd_subscriber.outputs:data", f"{G}/articulation_controller.inputs:positionCommand"),
        ]
    },
)
pump()

if args.thick_floor and not stage.GetPrimAtPath("/World/thick_floor").IsValid():
    # The carpet collider is a zero-thickness mesh; humanoid feet get through its contacts and the G1 falls.
    # An invisible 1 m thick box with its top 1 mm below the carpet (friction 1.0, as in training) fixes that.
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Usd

    top = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
        stage.GetPrimAtPath("/World/floor_carpet")).ComputeAlignedRange().GetMax()[2]
    box = UsdGeom.Cube.Define(stage, "/World/thick_floor")
    box.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(box).SetScale((60.0, 60.0, 1.0))
    UsdGeom.XformCommonAPI(box).SetTranslate((0.0, 0.0, top - 0.5 - 0.001))
    UsdGeom.Imageable(box).CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    mat = UsdShade.Material.Define(stage, "/World/Physics_Materials/thick_floor_friction")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(1.0)
    api.CreateDynamicFrictionAttr(1.0)
    api.CreateRestitutionAttr(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr("multiply")
    px.CreateRestitutionCombineModeAttr("multiply")
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    print(f"[build] thick floor collider added, top at z={top - 0.001:.3f}", flush=True)

omni.usd.get_context().save_as_stage(args.out)
pump(5)
print(f"[build] saved {args.out} with graph {G}", flush=True)
app.close()
