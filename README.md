# robots_isaac

Unitree **Go2** and our 23-joint **G1** in NVIDIA Isaac Sim 5.1 / Isaac Lab: walking policies trained with RSL-RL (PPO),
an office scene with simulated sensors (lidar, RealSense) published to ROS 2, and the policies running in that scene as
separate ROS 2 nodes.

Machine: Ubuntu 24.04, RTX 5070 Ti, Isaac Sim at `~/isaac-sim-5.1.0`, Isaac Lab at `~/IsaacLab`, ROS 2 Jazzy.

## The graphs in `scene/office.usd`

`scene/office.usd` is the only world file. It contains the office, the Go2 (`/World/go2`), the G1 (`/World/g1_23dof`),
an invisible thick floor under the carpet (`/World/thick_floor`, see "Carpet collider" below) and four OmniGraphs, a sensor
graph and a policy graph per robot. All of them start when you press Play.

| Graph | Robot | Does | ROS topics |
|---|---|---|---|
| `/World/Go2SensorGraph` | Go2 | lidar + RealSense -> ROS 2 | `/go2/point_cloud`, `/go2/rgb`, `/go2/depth`, ... |
| `/World/G1SensorGraph` | G1 | lidar + RealSense -> ROS 2 | `/g1/point_cloud`, `/g1/rgb`, `/g1/depth`, ... |
| `/World/PolicyGraph` | Go2 | observations out, joint targets in | `/go2/obs`, `/go2/joint_commands` |
| `/World/G1PolicyGraph` | G1 | observations out, joint targets in | `/g1/obs`, `/g1/joint_commands` |

### 1. `/World/Go2SensorGraph`: Go2 sensors -> ROS 2

Runs on every rendered frame (`On Playback Tick`). It renders the Go2's sensors and publishes them. No policy is involved.

```
On Playback Tick
 ├─> Create Render Product (lidar PandarXT_32_10hz)   ─> ROS2 RTX Lidar Helper   /go2/point_cloud    frame hesai
 ├─> Create Render Product (RealSense colour, 1280x800) ─> ROS2 Camera Helper    /go2/rgb            frame realsense_color
 │                                                      └> ROS2 Camera Info      /go2/rgb_camera_info
 └─> Create Render Product (RealSense depth, 1024x540)  ─> ROS2 Camera Helper    /go2/depth          frame realsense_depth
                                                        └> ROS2 Camera Info      /go2/depth_camera_info
```

Sensor prims: lidar `/World/go2/base/radar/World/XT32_SD10/PandarXT_32_10hz`, RealSense
`/World/go2/base/front_camera/Realsense_11/RSD455`. No IMU topic and no `/tf`.

### 1b. `/World/G1SensorGraph`: G1 sensors -> ROS 2

Same structure as the Go2 graph, for the G1. Each graph has its own ROS namespace (`/go2/...`, `/g1/...`), so the topics do not collide:
lidar `/g1/point_cloud` (frame `g1_mid360`, prim `/World/g1_23dof/torso_link/mid360_link/Example_Rotary`, a 40-channel
approximation of the Livox Mid-360), RealSense colour `/g1/rgb` + `/g1/rgb_camera_info` (frame `g1_d435_color`) and depth
`/g1/depth` + `/g1/depth_camera_info` (frame `g1_d435_depth`), from `/World/g1_23dof/torso_link/d435_link/Realsense_01/RSD455`.
An RTX lidar has no mesh, so it is invisible in the viewport; its points only appear if you attach a debug-draw writer to it.

### 2. `/World/PolicyGraph` and `/World/G1PolicyGraph`: policies over ROS 2

Each runs on every physics step (`On Physics Step`, 200 Hz). A graph does not run the neural network itself: it publishes what
the policy needs to see and applies the joint targets that come back. The network runs in `ros/policy_ros_node.py`, a
normal ROS 2 process. The two policy graphs have the same structure; the Go2 one is drawn here, the G1 one uses `/g1/...` topics
and the articulation `/World/g1_23dof/pelvis`.

```
Isaac Sim  /World/PolicyGraph                                           ROS 2 (system Python)
  On Physics Step (200 Hz)
   ├─> Script Node  ros/sim_obs_node.py ──(every 4th step = 50 Hz)──> ROS2 Publisher /go2/obs ──> ros/policy_ros_node.py
   └─> ROS2 Subscriber /go2/joint_commands ─> Articulation Controller /World/go2/base <──────────── (TorchScript policy)
                                                                               /cmd_vel or keyboard ─> velocity command
```

- `sim_obs_node.py` reads the robot's state through `sim/velocity_policy.py`. It picks the robot and policy folder from the
  graph's name (`PolicyGraph` = Go2, `G1PolicyGraph` = G1). On the first step after Play it also sets the PD gains, torque
  limit and armature from that policy's `params/env.yaml`.
- `/go2/obs` = `[step, sim_time, base_lin_vel(3), base_ang_vel(3), gravity(3), joint_pos_rel(12), joint_vel_rel(12)]`, all in
  the body frame, as in training (the G1 has 23 joints instead of 12). The ROS node adds the velocity command and the last
  action, runs the policy (about 0.1 ms) and replies right away (event driven, no timer).
- If `policy_ros_node.py` is not running for a robot, nothing commands its joints and it only holds the PD gains.
- The graph stores the full path `/home/viny/robots_isaac/ros/sim_obs_node.py` (Script Node `scriptPath`). If
  the repo moves, rebuild the graph (`ros/build_policy_graph.py`, below).

The sensor graphs and the policy graphs are independent: you can use the sensors without a policy node, and a policy
without looking at the sensors.

## Layout

```
robots_isaac/
├── go2/                   everything for the Go2 policy: training code next to the trained weights   -> ~/IsaacLab/isaaclab.sh
│   ├── train.sh               train / play (stock Isaac Lab task Isaac-Velocity-Flat-Unitree-Go2-v0)
│   └── policy/                model_1999.pt, exported/policy.pt + policy.onnx, params/env.yaml + agent.yaml
├── g1/                    the same for our 23-joint G1
│   ├── train.py               train / play (registers our task, then runs Isaac Lab's train.py / play.py)
│   ├── g1_23dof_task/         the task definition: Isaac-Velocity-Flat-G1-23DoF-v0 (cfg.py, plus stand-still / yaw-only command code)
│   ├── policy/                model_3599.pt, exported/policy.pt + policy.onnx, params/env.yaml + agent.yaml  (deployed)
│   └── policy_stand/          model_4398.pt, a stand-still fine-tuned checkpoint (not used by the scene)
├── scene/                 the world (open in Isaac Sim)
│   ├── office.usd             office + Go2 + G1 + the four graphs above
│   ├── go2_description/       Go2 robot USD, referenced by office.usd
│   ├── g1_23dof/              G1 23-joint robot USD, referenced by office.usd and by the G1 training task
│   └── textures/
├── sim/
│   └── velocity_policy.py     robot state / gains / observation helper (VelocityPolicyController), used by ros/sim_obs_node.py
├── ros/                   both ends of the policy graphs, and RViz
│   ├── sim_obs_node.py        the Script Node file inside the policy graphs (runs in Isaac Sim)
│   ├── policy_ros_node.py     the policy as a ROS 2 node, with keyboard teleop       -> /usr/bin/python3 ...
│   ├── build_policy_graph.py  (re)build a policy graph in office.usd (Go2 or G1)    -> ~/isaac-sim-5.1.0/python.sh ...
│   ├── run_scene.py           play office.usd (window or headless) and report a robot's pose (tests)
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
| Result | tracking error 0.14 m/s, 99.4% of episodes time out (do not fall) | walks upright in the office over ROS 2 (about 0.4 m/s at a 0.5 m/s command, tilt under 7 degrees) |
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
copy its last `model_*.pt`, `exported/` and `params/` into `go2/policy/` or `g1/policy/`. To try a policy by hand, use the ROS 2 route below.

## Run the policies in the office (over ROS 2)

The policy runs in a separate ROS 2 process and talks to a policy graph in the scene: `/World/PolicyGraph` for the Go2 and
`/World/G1PolicyGraph` for the G1. Start Isaac Sim (`cd ~/isaac-sim-5.1.0 && ./isaac-sim.sh`), open `scene/office.usd` and
press Play. Then start one node per robot, each in its own terminal:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 ~/robots_isaac/ros/policy_ros_node.py --robot go2      # Go2, /cmd_vel
/usr/bin/python3 ~/robots_isaac/ros/policy_ros_node.py --robot g1       # G1, /g1/cmd_vel
```

The keys are read from the terminal where the node runs, so keep that terminal focused. Both nodes can run at once.

| Key (in that terminal) | Command |
|---|---|
| `w` / `s` or Up / Down | forward / backward |
| `a` / `d` | sideways left / right |
| `q` / `e` or Left / Right | turn left / right |
| space | stop |
| Ctrl+C | quit |

A key sends its command only for a short moment and then the robot stops again: a tap is a short burst, holding the key keeps it
moving (a terminal cannot see key release, so key repeat renews the command). Tune with `--key-hold 0.4` (seconds a press stays
active; raise it to 0.7 if holding a key stutters before the repeat starts) and `--key-speed 0.5,0.3,1.0` (vx, vy, wz).

Commands are clipped to the trained ranges, so `s` (backward) does nothing for the G1 (its forward range starts at 0).
`/cmd_vel` (Go2) or `/g1/cmd_vel` (G1), a Twist, is accepted too; an active key press overrides it. With `--no-keyboard`, a
Twist older than 0.5 s counts as zero.

Headless test instead of the window: `./python.sh ~/robots_isaac/ros/run_scene.py --headless --seconds 16 --realtime`
(add `--robot-path /World/g1_23dof/pelvis` for the G1; from `~/isaac-sim-5.1.0` after `source setup_ros_env.sh`), with the
policy node running. Checked: observations at 50.0 Hz, inference 0.1 ms, the Go2 walked 6.2 m at 0.5 m/s and stayed upright,
and the G1 walked about 0.4 m/s and stayed upright. The simulator does not wait for the policy, so with real-time simulation
the command is at most one policy step late.

Rebuild a policy graph (after moving the repo, or to change robot / topics). This replaces that graph in `office.usd` in place
(`--out <copy.usd>` writes a copy instead). `sim_obs_node.py` picks the robot from the graph's name, so keep the names
`PolicyGraph` (Go2) and `G1PolicyGraph` (G1):

```bash
source ~/isaac-sim-5.1.0/setup_ros_env.sh && cd ~/isaac-sim-5.1.0
./python.sh ~/robots_isaac/ros/build_policy_graph.py                                   # Go2: /World/PolicyGraph
./python.sh ~/robots_isaac/ros/build_policy_graph.py --robot-path /World/g1_23dof/pelvis --ns g1 --graph /World/G1PolicyGraph
```

## Sensors in RViz (Go2 config; the G1 topics `/g1/...` work the same, there is no RViz config for them yet)

```bash
# terminal 1: Isaac Sim (ROS not sourced), open scene/office.usd, press Play
cd ~/isaac-sim-5.1.0 && ./isaac-sim.sh

# terminal 2
source /opt/ros/jazzy/setup.bash
ros2 topic hz /go2/point_cloud
rviz2 -d ~/robots_isaac/ros/go2_sensors.rviz     # /go2/point_cloud, /go2/rgb, /go2/depth (Best Effort QoS), Fixed Frame = hesai
```

RViz's default Fixed Frame is `map`. Without a `/tf` publisher you get "could not transform from hesai to map", so keep the
Fixed Frame at `hesai` or add a **ROS2 Publish Transform Tree** node to the Go2SensorGraph.

## Things we ran into

- **Carpet collider:** the office carpet is a zero-thickness mesh, and the G1 fell through its contacts within seconds.
  `/World/thick_floor` is an invisible 1 m box with its top 1 mm below the carpet (friction 1.0), and the G1 walks upright on it.
  `build_policy_graph.py` adds it if it is missing (`--no-thick-floor` turns that off).
- The RealSense asset carries its own rigid body, which sits inside the robot's rigid body (Go2 `base`, G1 `d435_link`) and
  causes PhysX errors; it also makes the prim read-only in the GUI. Remove the Rigid Body from `RSD455` (right-click > Remove >
  Physics > Rigid Body), and deactivate its `Imu_Sensor`, which needs a rigid-body parent. Done for the G1; the Go2's
  `Imu_Sensor` is still active and logs "Pattern '.../RSD455' did not match any rigid bodies" at startup (harmless).
- RTX lidars and cameras have no mesh, so they are invisible in the viewport; only the RealSense body mesh shows. The G1 lidar
  (`mid360_link/Example_Rotary`) is an approximation of the Livox Mid-360: 40 channels from -7 to +52 degrees, 360 degree
  azimuth, 10 Hz, 200k points/s, 0.1 to 70 m. Its profile is stored as USD attributes on the prim, not a JSON file.
- Do not reload `office.usd` inside a running Isaac Sim session after the graphs changed: stale render products produce
  hundreds of thousands of "Invalid USD RenderProduct Prim" errors and the sensor topics stop. Quit and start Isaac Sim again.
- `ros/policy_ros_node.py` holds the G1's joint order (the articulation order of the scene USD, 23 joints). If the robot USD
  changes, check it against `SingleArticulation(...).dof_names`.
- Add the lidar asset once only. A duplicate `World/XT32_SD10` can end up directly under `base`; delete the extra copy.
- The lidar render product must point at the `OmniLidar` prim (`PandarXT_32_10hz`), not at the outer Xform.
- Isaac Sim warns that the RealSense aperture does not match the viewport aspect and that depth texture sizes differ. These are
  preview-window warnings and do not affect published data.

## Not done yet

- IMU topic and `/tf` for either robot.
- An RViz config for the G1 sensors.
- The G1 drifts slowly (about 0.05 m/s) at zero command; the stand-still fine-tuned `g1/policy_stand/` is not wired in.
- Check `/go2/point_cloud`, `/go2/rgb`, `/go2/depth` and the `/g1/...` topics in RViz while a policy walks.
