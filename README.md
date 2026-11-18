# robots_isaac

Unitree **Go2** and our 23-joint **G1** in NVIDIA Isaac Sim 5.1 / Isaac Lab: walking policies trained with RSL-RL (PPO),
an office scene with simulated sensors (Hesai lidar, RealSense) published to ROS 2, and the policies running in that
scene either inside Isaac Sim or as a separate ROS 2 node.

Machine: Ubuntu 24.04, RTX 5070 Ti, Isaac Sim at `~/isaac-sim-5.1.0`, Isaac Lab at `~/IsaacLab`, ROS 2 Jazzy.

## The two graphs in `scene/office.usd`

`scene/office.usd` is the only world file. It contains the office, the Go2 (`/World/go2`), the G1 (`/World/g1_23dof`),
an invisible thick floor under the carpet (`/World/thick_floor`, see "Carpet collider" below) and two OmniGraphs.
Both start when you press Play.

### 1. `/World/ActionGraph`: sensors -> ROS 2

Runs on every rendered frame (`On Playback Tick`). It renders the Go2's sensors and publishes them. No policy is involved.

```
On Playback Tick
 ├─> Create Render Product (lidar PandarXT_32_10hz)   ─> ROS2 RTX Lidar Helper   /point_cloud        frame hesai
 ├─> Create Render Product (RealSense colour, 1280x800) ─> ROS2 Camera Helper    /rgb                frame realsense_color
 │                                                      └> ROS2 Camera Info      /rgb_camera_info
 └─> Create Render Product (RealSense depth, 1024x540)  ─> ROS2 Camera Helper    /depth              frame realsense_depth
                                                        └> ROS2 Camera Info      /depth_camera_info
```

Sensor prims: lidar `/World/go2/base/radar/World/XT32_SD10/PandarXT_32_10hz`, RealSense
`/World/go2/base/front_camera/Realsense_11/RSD455`. No IMU topic, no `/tf` and no G1 sensors yet.

### 2. `/World/PolicyGraph`: Go2 policy over ROS 2

Runs on every physics step (`On Physics Step`, 200 Hz). The graph does not run the neural network itself: it publishes what
the policy needs to see and applies the joint targets that come back. The network runs in `ros/policy_ros_node.py`, a
normal ROS 2 process.

```
Isaac Sim  /World/PolicyGraph                                           ROS 2 (system Python)
  On Physics Step (200 Hz)
   ├─> Script Node  ros/sim_obs_node.py ──(every 4th step = 50 Hz)──> ROS2 Publisher /go2/obs ──> ros/policy_ros_node.py
   └─> ROS2 Subscriber /go2/joint_commands ─> Articulation Controller /World/go2/base <──────────── (TorchScript policy)
                                                                               /cmd_vel or keyboard ─> velocity command
```

- `sim_obs_node.py` reads the Go2's state through `sim/velocity_policy.py`. On the first step after Play it also sets the PD
  gains, torque limit and armature from `go2/policy/params/env.yaml`.
- `/go2/obs` = `[step, sim_time, base_lin_vel(3), base_ang_vel(3), gravity(3), joint_pos_rel(12), joint_vel_rel(12)]`, all in
  the body frame, as in training. The ROS node adds the velocity command and the last action, runs the policy (about 0.1 ms)
  and replies right away (event driven, no timer).
- If `policy_ros_node.py` is not running, nothing commands the Go2's joints and it only holds the PD gains.
- The graph stores the full path `/home/viny/robots_isaac/ros/sim_obs_node.py` (Script Node `scriptPath`). If
  the repo moves, rebuild the graph (`ros/build_policy_graph.py`, below).

The two graphs are independent: you can use the sensors without the policy node, and the policy without looking at the
sensors.

## Layout

```
robots_isaac/
├── go2/                   everything for the Go2 policy: training code next to the trained weights   -> ~/IsaacLab/isaaclab.sh
│   ├── train.sh               train / play (stock Isaac Lab task Isaac-Velocity-Flat-Unitree-Go2-v0)
│   └── policy/                model_1999.pt, exported/policy.pt + policy.onnx, params/env.yaml + agent.yaml
├── g1/                    the same for our 23-joint G1
│   ├── train.py               train / play (registers our task, then runs Isaac Lab's train.py / play.py)
│   ├── g1_23dof_task/         the task definition: Isaac-Velocity-Flat-G1-23DoF-v0 (cfg.py)
│   └── policy/                model_3599.pt, exported/policy.pt + policy.onnx, params/env.yaml + agent.yaml
├── scene/                 the world (open in Isaac Sim)
│   ├── office.usd             office + Go2 + G1 + ActionGraph (sensors) + PolicyGraph (Go2 policy over ROS)
│   ├── go2_description/       Go2 robot USD, referenced by office.usd
│   ├── g1_23dof/              G1 23-joint robot USD, referenced by office.usd and by the G1 training task
│   └── textures/
├── sim/                   policies in the office, one process, no ROS for the policy -> ~/isaac-sim-5.1.0/python.sh ...
│   ├── office_standalone.py   open office.usd, run the Go2 and/or G1 policy, keyboard control
│   └── velocity_policy.py     the controller (VelocityPolicyController); also used by ros/sim_obs_node.py
├── ros/                   the PolicyGraph's two ends, and RViz
│   ├── sim_obs_node.py        the Script Node file inside /World/PolicyGraph (runs in Isaac Sim)
│   ├── policy_ros_node.py     the policy as a ROS 2 node, with keyboard teleop       -> /usr/bin/python3 ...
│   ├── build_policy_graph.py  (re)build /World/PolicyGraph in office.usd             -> ~/isaac-sim-5.1.0/python.sh ...
│   ├── run_scene.py           play office.usd headless and report the Go2's pose (tests)
│   └── go2_sensors.rviz       RViz config: lidar, RGB, depth
└── third_party/unitree_ros/   upstream Unitree repo (robot descriptions), not modified
```

Three launchers, do not mix them:

| Folder | Runs with | Why |
|---|---|---|
| `go2/`, `g1/` (training) | `~/IsaacLab/isaaclab.sh -p <script>` (`go2/train.sh` does it for you) | needs Isaac Lab (gym tasks, RSL-RL) |
| `sim/`, `ros/` except `policy_ros_node.py` | `~/isaac-sim-5.1.0/python.sh <script>` (after `source setup_ros_env.sh`) | plain Isaac Sim with its own ROS 2 bridge libraries |
| `ros/policy_ros_node.py` | `/usr/bin/python3` (after `source /opt/ros/jazzy/setup.bash`) | normal ROS 2 node |

Do not source `/opt/ros/jazzy/setup.bash` in the terminal that starts Isaac Sim: `isaac-sim.sh` / `setup_ros_env.sh` only
set up the bridge's own libraries when `ROS_DISTRO` is empty.

## Trained policies (`go2/policy/`, `g1/policy/`)

Both are **velocity-tracking** policies. You give a body-frame command (forward speed, sideways speed, yaw rate) and the
policy outputs joint position targets that make the robot walk at that velocity on flat ground. Trained with Isaac Lab and
RSL-RL's PPO, 4096 parallel environments on one GPU.

| | `go2/policy` | `g1/policy` |
|---|---|---|
| Task | `Isaac-Velocity-Flat-Unitree-Go2-v0` (stock) | `Isaac-Velocity-Flat-G1-23DoF-v0` (ours, `g1/g1_23dof_task/`) |
| Robot asset in training | Isaac Lab Nucleus `Go2/go2.usd` | `scene/g1_23dof/g1_23dof.usd` (the same G1 as in the office) |
| Checkpoint | `model_1999.pt` (2000 iterations) | `model_3599.pt` (2000 iterations, then resumed to 3600) |
| Network (actor and critic) | MLP 128-128-128, ELU | MLP 256-128-128, ELU |
| Trained command ranges | x ±1.0 m/s, y ±1.0 m/s, yaw ±1.0 rad/s | x **0 to 1.0** m/s, y ±0.5 m/s, yaw ±1.0 rad/s |
| Result | tracking error 0.14 m/s, 99.4% of episodes time out (do not fall) | walks upright in the office (6 s at 0.5 m/s: 2.8 m, tilt under 5 degrees) |
| Isaac Lab log (original run) | `~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-11-17_18-47-46` | `~/IsaacLab/logs/rsl_rl/g1_23dof_flat/2026-11-18_15-42-08` |

The G1 task is the stock G1 flat task with our robot swapped in. The stock one trains Isaac Lab's 37-joint `g1_minimal.usd`, which
cannot drive our 23-joint G1. Our task also uses the real torque limits 88 / 139 / 35 / 25 N*m and stiffer ankles, and its
rewards point at joints that exist.

Shared training setup (each policy's `params/agent.yaml` and `env.yaml` hold the exact values):

- **Algorithm:** PPO, 24 steps per env per update, 5 epochs, 4 mini-batches, learning rate 1e-3 (adaptive, desired KL 0.01),
  gamma 0.99, lambda 0.95, clip 0.2, initial action noise std 1.0, seed 42, a checkpoint every 50 iterations.
- **Simulation:** physics at 200 Hz (`dt = 0.005`), policy at 50 Hz (decimation 4), 20 s episodes, flat plane terrain.
- **Commands:** resampled every 10 s, 2% of environments are told to stand still.
- **Reward:** exponential tracking of linear xy velocity and yaw rate, minus penalties for vertical velocity, roll/pitch rate,
  joint torque, joint acceleration, action rate and similar. The G1 adds feet air time, ankle/hip deviation and joint limits.
- **Actions:** joint position targets added to the default pose, tracked by PD actuators.

Limitations:

- **Simulation only.** The observation includes the true base linear velocity, which a real robot does not have. Deploying
  would need a state estimator or retraining without that term.
- **They march in place at zero command** instead of standing still.
- **The G1 cannot walk backwards**, because its trained forward range starts at 0.
- Joint names, gains and default pose come from the policy's `env.yaml`, so a policy only fits a robot USD with the same joint names.

### Train (from `~/IsaacLab`)

```bash
~/robots_isaac/go2/train.sh train            # Go2, flat, 4096 envs, headless  (also: train rough, --gui, --num_envs N)
~/robots_isaac/go2/train.sh play             # watch the newest Go2 run in the viewport
./isaaclab.sh -p ~/robots_isaac/g1/train.py train --task Isaac-Velocity-Flat-G1-23DoF-v0 --headless --num_envs 4096
./isaaclab.sh -p ~/robots_isaac/g1/train.py play  --task Isaac-Velocity-Flat-G1-23DoF-Play-v0 --num_envs 16   # also exports policy.pt / .onnx
```

New runs land in `~/IsaacLab/logs/rsl_rl/<experiment>/<date>/` (Go2: `unitree_go2_flat`, G1: `g1_23dof_flat`). To deploy one,
copy its last `model_*.pt`, `exported/` and `params/` into `go2/policy/` or `g1/policy/`. To try a policy by hand, use
`sim/office_standalone.py` below (it uses these folders).

## Run policies in the office: two ways

| | `sim/office_standalone.py` | `/World/PolicyGraph` + `ros/policy_ros_node.py` |
|---|---|---|
| Where the network runs | inside Isaac Sim | separate ROS 2 process |
| Robots | Go2 and/or G1 | Go2 only (G1 not ported yet) |
| Command | Isaac Sim viewport keyboard, or `--cmd` | terminal keyboard, or `/cmd_vel` |
| PolicyGraph | switched off by the script | does the work |
| Sensor ActionGraph | runs | runs |

### A. Standalone (`sim/office_standalone.py`)

One process, like Isaac Sim's `h1_standalone.py`. It opens `scene/office.usd`, deactivates `/World/PolicyGraph` so it does
not fight over the Go2's joints, sets physics to 200 Hz and attaches a `VelocityPolicyController` per robot. Each controller
reads default pose, PD gains, torque limit, action scale, decimation and command ranges from the policy's `env.yaml` and
runs the TorchScript policy every 4th physics step. Friction is set as in training (floor 1.0, robots 0.8).

```bash
source ~/isaac-sim-5.1.0/setup_ros_env.sh && cd ~/isaac-sim-5.1.0
./python.sh ~/robots_isaac/sim/office_standalone.py --robots go2,g1 --drive g1      # window, keyboard drives the G1
./python.sh ~/robots_isaac/sim/office_standalone.py --robots g1 --headless --seconds 10 --cmd 0.5,0,0   # test
```

Keys (click the viewport first): Up / Down forward / back, Left / Right sideways, Z / X yaw. Robots that are not `--drive`n
get a zero command. Flags: `--cmd vx,vy,wz`, `--seconds N`, `--headless`, `--no-ros`, `--run g1=<policy folder>`, `--scene <usd>`.

Checked headless (6 s, both robots, G1 driven at 0.5 m/s): G1 walked 2.8 m, Go2 stayed in place, both under 5 degrees of tilt.

### B. Over ROS 2 (PolicyGraph)

Start Isaac Sim (`cd ~/isaac-sim-5.1.0 && ./isaac-sim.sh`), open `scene/office.usd` and press Play. Then, in another terminal:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 ~/robots_isaac/ros/policy_ros_node.py --robot go2
```

| Key (in that terminal) | Command (stays until changed) |
|---|---|
| `w` / `s` or Up / Down | forward / backward, ±0.1 m/s per press |
| `a` / `d` | sideways left / right, ±0.1 m/s per press |
| `q` / `e` or Left / Right | turn left / right, ±0.2 rad/s per press |
| space | stop |
| Ctrl+C | quit |

Commands are limited to the trained ranges. `/cmd_vel` (Twist) is accepted too, and the latest source wins. With `--no-keyboard`,
a `/cmd_vel` older than 0.5 s counts as zero.

Headless test instead of the window: `./python.sh ~/robots_isaac/ros/run_scene.py --headless --seconds 16 --realtime`
(from `~/isaac-sim-5.1.0` after `source setup_ros_env.sh`), with the policy node running. Checked: observations at 50.0 Hz,
inference 0.1 ms, the Go2 walked 6.2 m at 0.5 m/s and stayed upright. The simulator does not wait for the policy, so with
real-time simulation the command is at most one policy step late.

Rebuild the graph (after moving the repo, or to change robot / topics). This replaces `/World/PolicyGraph` in `office.usd` in place:

```bash
source ~/isaac-sim-5.1.0/setup_ros_env.sh && cd ~/isaac-sim-5.1.0
./python.sh ~/robots_isaac/ros/build_policy_graph.py [--robot-path /World/go2/base] [--ns go2] [--out <copy.usd>]
```

## Sensors in RViz

```bash
# terminal 1: Isaac Sim (ROS not sourced), open scene/office.usd, press Play
cd ~/isaac-sim-5.1.0 && ./isaac-sim.sh

# terminal 2
source /opt/ros/jazzy/setup.bash
ros2 topic hz /point_cloud
rviz2 -d ~/robots_isaac/ros/go2_sensors.rviz     # /point_cloud, /rgb, /depth (Best Effort QoS), Fixed Frame = hesai
```

RViz's default Fixed Frame is `map`. Without a `/tf` publisher you get "could not transform from hesai to map", so keep the
Fixed Frame at `hesai` or add a **ROS2 Publish Transform Tree** node to the ActionGraph.

## Things we ran into

- **Carpet collider:** the office carpet is a zero-thickness mesh, and the G1 fell through its contacts within seconds.
  `/World/thick_floor` is an invisible 1 m box with its top 1 mm below the carpet (friction 1.0), and the G1 walks upright on it.
  `build_policy_graph.py` and `office_standalone.py` add it if it is missing (`--no-thick-floor` turns that off).
- The RealSense asset carries its own rigid body, which sits inside the Go2's `base` rigid body and causes PhysX errors.
  Remove the Rigid Body from `RSD455` (right-click > Remove > Physics > Rigid Body). A leftover harmless message at startup:
  "Pattern '.../RSD455' did not match any rigid bodies".
- Add the lidar asset once only. A duplicate `World/XT32_SD10` can end up directly under `base`; delete the extra copy.
- The lidar render product must point at the `OmniLidar` prim (`PandarXT_32_10hz`), not at the outer Xform.
- Isaac Sim warns that the RealSense aperture does not match the viewport aspect and that depth texture sizes differ. These are
  preview-window warnings and do not affect published data.

## Not done yet

- G1 over ROS 2 (PolicyGraph for `/World/g1_23dof/pelvis`, G1 joint order in `policy_ros_node.py`, G1 constants in `sim_obs_node.py`).
- IMU topic, `/tf`, sensors on the G1.
- Check `/point_cloud`, `/rgb`, `/depth` in RViz while a policy walks.
