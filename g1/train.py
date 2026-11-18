"""Run Isaac Lab's RSL-RL train.py / play.py with the G1-23DoF tasks registered (Isaac Lab itself is not modified).

    cd ~/IsaacLab
    ./isaaclab.sh -p ~/robots_isaac/g1/train.py train --task Isaac-Velocity-Flat-G1-23DoF-v0 --headless --num_envs 4096
    ./isaaclab.sh -p ~/robots_isaac/g1/train.py play  --task Isaac-Velocity-Flat-G1-23DoF-Play-v0 --num_envs 16

All further arguments are passed to Isaac Lab's script unchanged. `play` also writes exported/policy.pt and policy.onnx.
"""

import os
import sys

RSL = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl")
HERE = os.path.dirname(os.path.abspath(__file__))

mode = sys.argv[1]
if mode not in ("train", "play"):
    raise SystemExit("usage: train.py {train|play} [Isaac Lab args]")
script = os.path.join(RSL, mode + ".py")

src = open(script).read()
marker = "import isaaclab_tasks  # noqa: F401"
assert marker in src, f"{script} no longer contains {marker!r}"
src = src.replace(marker, marker + "\nimport g1_23dof_task  # noqa: F401  (registers the 23-joint G1 tasks)", 1)

sys.path[:0] = [RSL, HERE]  # cli_args lives next to train.py; g1_23dof_task is a package next to this file
sys.argv = [script] + sys.argv[2:]
exec(compile(src, script, "exec"), {"__name__": "__main__", "__file__": script})
