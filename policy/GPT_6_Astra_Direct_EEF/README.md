# GPT-6 Astra

LLM agent example using OpenAI model `gpt-6-astra` through the Responses API. It runs in the shared policy server like every other example: the evaluator sends observations and executes the returned action chunks.

Each `get_action` call sends the instruction, every camera image, and both arms' end-effector state. The model answers with one or more `move_eef` tool calls, each naming an arm, an absolute world-frame position, and a gripper opening (0 closed, 1 open). `model.py` turns those goals into an end-effector action chunk: positions are interpolated at most `max_step_m` per env step, orientation is held, and the gripper changes once the arm arrives and is held for `gripper_dwell_steps`. The conversation continues across calls with `previous_response_id`, one conversation per environment, until `reset`.

## Setup

```sh
python -m pip install -e . openai
export OPENAI_API_KEY=...   # never put keys in deploy.yml or Git
python setup_policy_server.py --config_path policy/GPT_6_Astra_Direct_EEF/deploy.yml
```

`deploy.yml` selects the model, reasoning effort, image size, interpolation limits, and `scene_notes`. Edit `scene_notes` to describe the evaluator's world frame and workspace (table height, where each arm is mounted); the model can only name useful positions if it knows the frame. Set `base_url` to use an OpenAI-compatible endpoint.

## Interface

- Observations: `instruction`, any RGB cameras under `vision`, and `left_ee_pose`, `right_ee_pose` (`[x, y, z, qw, qx, qy, qz]`), `left_ee_joint_state`, `right_ee_joint_state` (gripper, 0 closed to 1 open) under `state`.
- Actions: `action_type: ee` chunks with the same four keys. The evaluator performs inverse kinematics.

Each call to `get_action` is one model request, so latency is seconds, not milliseconds. Model output is not deterministic.
