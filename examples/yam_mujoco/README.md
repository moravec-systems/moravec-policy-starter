# YAM MuJoCo closed-loop check

A small bimanual YAM tabletop scene for running a YAM policy server closed loop before real evaluation, with joint or end-effector actions. It is a wiring and behaviour check, not a calibrated simulator or a benchmark.

The scene uses i2rt's MIT-licensed `yam_station_linear_4310_d405` model (two YAM arms with linear 4310 grippers, D405 wrist cameras, and an overhead D405). `yam_scene.py` adds:

- MuJoCo cameras at the three D405 optical frames: `top`, `left`, `right` (640×480 RGB, 58° vertical FOV).
- Position actuators for joints 1–6 and the gripper on each arm, with gravity compensation.
- A table, a 4 cm red block, and a green plate.

`run_episode.py` connects to a policy server with the shared client, sends `update_obs`/`get_action`, executes each returned chunk at 30 Hz, and writes a video (top view above both wrist views) plus a JSON summary.

## Setup

```sh
python -m pip install -e . mujoco "imageio[ffmpeg]"
git clone https://github.com/i2rt-robotics/i2rt examples/yam_mujoco/i2rt
git -C examples/yam_mujoco/i2rt checkout 120c3c81400171174604e503943f8d1ebc891058
```

Set `I2RT_ROOT` instead if you already have an i2rt checkout. Only its robot model files are used.

## Run

Start a YAM policy server, for example the [MolmoAct2 container](../../policy/MolmoAct2/README.md#container), then:

```sh
python examples/yam_mujoco/run_episode.py --url ws://127.0.0.1:6000 --chunks 20 --out episode.mp4
```

For an end-effector policy, add `--action-type ee`. For example, the Astra LLM agent with this scene's frame described in its prompt:

```sh
export OPENAI_API_KEY=...
python setup_policy_server.py --config_path examples/yam_mujoco/astra_yam.yml &
python examples/yam_mujoco/run_episode.py --action-type ee --chunks 15 --out astra.mp4
```

The simulation runs on CPU, so it can run on a laptop while the policy runs on a GPU host behind the SSH tunnel from the [integration guide](../../INTEGRATION.md#remote-serving).

## Interface

- Observation: `instruction`, `vision.{top,left,right}.color`, and `state.{left,right}_arm_joint_state` (6 joint angles, radians) plus `state.{left,right}_ee_joint_state` (gripper, 0 closed to 1 open, i2rt's convention, mapped linearly onto the finger travel).
- Action: chunks of absolute targets with the same four keys.
- With `--action-type ee`, state and actions use `{left,right}_ee_pose` instead of joint angles: the fingertip centre (i2rt's `tcp_left`/`tcp_right` sites) as `[x, y, z, qw, qx, qy, qz]` in the world frame (+z up, table at z = 0, left arm base at the origin, right arm base at y = -0.61). Targets are converted to joints with damped least-squares IK (a few millimetres of error in the central workspace). Episodes start from a ready pose with both grippers open, pointing down, 20 cm above the table.
- Episode ends after `--chunks` policy calls or when the block rests on the plate (`success: true`).

## What to expect

With MolmoAct2-BimanualYAM (30-step chunks, about 0.4 s per call on an L40S), two 45-second episodes on the pinned i2rt model behaved alike: the left arm reaches the block, repeatedly attempts a grasp, and nudges it 2–3 cm, once lifting it about 1.6 cm before losing it; neither episode placed it on the plate. The checkpoint was trained on real YAM data, and this scene's textures, lighting, camera intrinsics, start pose and contact model are not calibrated to Ai2's setup; the 30 Hz control rate is also an assumption. Treat task success here as a smoke signal only.

With the Astra agent (`gpt-6-astra`, about 5 s per call, block position not given in the prompt), one 15-call episode located the block from the images and brought the left gripper to it within a few calls, then hovered and nudged it 7 cm without closing on it.
