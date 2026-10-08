# GPT-6 Astra

LLM agent example using OpenAI model `gpt-6-astra` through the Responses API, following the design of the RoboProbe inspect agent. It runs in the shared policy server like every other example: the evaluator sends observations and executes the returned end-effector action chunks.

## How it works

Each turn sends the goal, every camera image (labelled with its name and step), and both arms' grasp-point state named exactly as the tool takes it: `left_x/y/z` in metres, `left_pitch_deg/roll_deg/yaw_deg` measured from a straight-down grasp (so `yaw_deg` turns the jaws), and `left_gripper` (0 closed, 1 open), likewise for `right_`. After a move it also reports how far the arm landed from the target ("left 10 mm and 0.0 deg away") and, if the evaluator sends `remaining_steps`, the env steps left.

The model must answer with exactly one tool call:

- `move_eef(targets, note)`: name only the dimensions to change; the rest hold. Both arms move together along time-aligned paths at a fixed speed (`max_step_m`, `max_step_deg` per env step); a move longer than `max_duration_s` is refused and must be split. A gripper value moves the jaw only after the arm arrives, then holds for `gripper_dwell_steps`. Targets outside `bounds` are clamped and the model is told.
- `give_up(reason, hindsight)`: ends the episode. There is deliberately no "done" tool; the evaluator ends a successful episode.

Malformed calls get up to three repair prompts. The conversation continues across calls (one per environment) until `reset` or `max_llm_calls` (100).

## Configuration

`deploy.yml` holds everything robot-specific. The defaults reproduce the original agent on the ARX X5 simulation embodiment:

- `down_quat_wxyz`: the reported `ee_pose` orientation of a straight-down grasp.
- `grasp_offset_m`: how far the grasp point sits beyond the reported pose along the tool axis (0.1501 m from the ARX flange; 0 when the evaluator already reports the fingertip centre).
- `bounds`: per-dimension limits, e.g. `left_x: [-1.1, 0.5]`.
- `embodiment_notes`: the frame, geometry, cameras and grasping advice the model reads in its instructions.
- `model`, `reasoning_effort` (default `medium`), `max_llm_calls`, motion speed and limits, optional `task_recipe` text appended to the first goal, and `base_url` for an OpenAI-compatible endpoint.

[`examples/yam_mujoco/astra_yam.yml`](../../examples/yam_mujoco/astra_yam.yml) configures it for the YAM MuJoCo scene.

## Setup

```sh
python -m pip install -e . openai pillow
export OPENAI_API_KEY=...   # never put keys in deploy.yml or Git
python setup_policy_server.py --config_path policy/GPT_6_Astra_Direct_EEF/deploy.yml
```

Observations need `instruction`, RGB cameras under `vision`, and `left_ee_pose`, `right_ee_pose` (`[x, y, z, qw, qx, qy, qz]`) plus `left_ee_joint_state`, `right_ee_joint_state` under `state`; actions use the same keys with `action_type: ee`, and the evaluator performs inverse kinematics. Each turn is one or more model requests, so latency is seconds, and output is not deterministic.

## Container

A CPU-only image of the same policy server. From the repository root:

```sh
docker build -f policy/GPT_6_Astra_Direct_EEF/Dockerfile -t moravec-astra:demo .
# OPENAI_API_KEY must already be exported in your shell; it is passed at runtime only.
docker run -d --rm --name moravec-astra -p 127.0.0.1:6000:6000 -e OPENAI_API_KEY moravec-astra:demo
moravec-validate policy/GPT_6_Astra_Direct_EEF --url ws://127.0.0.1:6000
docker stop moravec-astra
```

The validator sends synthetic camera images and end-effector state through the shared protocol, which makes a few paid `gpt-6-astra` requests, and checks the returned end-effector chunks, batching, and reset. It executes no actions and does not establish task success. Override `embodiment_notes` and the geometry settings for your evaluator by mounting a deployment YAML and overriding the default command with `python setup_policy_server.py --config_path /config/deploy.yml --host 0.0.0.0`.
