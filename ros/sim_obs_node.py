"""OmniGraph Script Node script (Isaac Sim side of the split architecture): builds the policy observation.

Graph (see build_policy_graph.py):
    On Physics Step --step--> Script Node --execOut--> ROS2 Publisher  (/<ns>/obs)
    On Physics Step --step--> ROS2 Subscriber (/<ns>/joint_commands) -> Articulation Controller (position targets)

The node runs on every physics step (200 Hz) but only passes execution on every `decimation` steps (50 Hz), so the
ROS side sees one observation per policy step of SIMULATION time, whatever the wall-clock speed of the simulator.
It also sets the PD gains / torque limit / armature from the policy's env.yaml on the first step after Play.

Output attribute `outputs:obs` (double[]):  [step, sim_time, lin_vel_b(3), ang_vel_b(3), gravity_b(3), joint_pos_rel(n), joint_vel_rel(n)]
"""

import sys

# ---- edit these ------------------------------------------------------------------------------------------------
ROBOT_PATH = "/World/go2/base"  # articulation root of the robot in the scene
RUN = "/home/viny/robots_isaac/go2/policy"
ENV_YAML = RUN + "/params/env.yaml"
# ---------------------------------------------------------------------------------------------------------------
_CODE_DIR = "/home/viny/robots_isaac/sim"


def setup(db):
    if _CODE_DIR not in sys.path:
        sys.path.insert(0, _CODE_DIR)
    import importlib

    import velocity_policy

    importlib.reload(velocity_policy)
    # policy_path=None: no policy in this process, only the robot state / gains
    db.per_instance_state.ctl = velocity_policy.VelocityPolicyController(ROBOT_PATH, None, ENV_YAML, log=lambda m: print(m, flush=True), name="obs_node")
    db.per_instance_state.step = 0
    # take over the node's execution output: it is only enabled on policy steps (checked by the node after compute())
    import omni.graph.tools.ogn as ogn

    db.node.get_attribute("outputs:execOut").set_metadata(ogn.MetadataKeys.HIDDEN, "1")


def compute(db):
    import numpy as np
    import omni.graph.core as og

    st = db.per_instance_state
    ctl = st.ctl
    try:
        if not ctl.ensure_init():  # (re)create the articulation after Stop -> Play; sets gains from env.yaml
            return True
        obs = ctl.observe_partial()
    except Exception as e:  # physics view not ready yet / destroyed by Stop
        ctl.invalidate()
        st.step = 0
        db.log_warning(f"waiting for articulation: {e}")
        return True

    step = st.step
    st.step += 1
    if step % ctl.decimation == 0:
        db.outputs.obs = np.concatenate([[float(step), step * ctl.dt], obs])
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED  # hidden on the node: we decide when to trigger the publisher
    return True


def cleanup(db):
    ctl = getattr(db.per_instance_state, "ctl", None)
    if ctl is not None:
        ctl.invalidate()
