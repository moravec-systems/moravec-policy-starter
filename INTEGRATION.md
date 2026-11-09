# Policy integration

## Submitting a policy

Send Moravec two things:

1. **A container image pinned by digest** (`registry/repository@sha256:…`, linux/amd64) that starts the policy server on port 6000. Build it like the examples in `policy/<name>/Dockerfile`. The simplest package includes the model weights in the image, so one digest identifies everything that runs. Moravec pulls it from your registry using read-only access you grant separately; never put credentials in the image, the manifest, or Git.
2. **A manifest, `moravec-policy.yaml`**, describing how to start the image and what it expects. Each example folder has one.

```yaml
schema: moravec.policy/v1
name: my-policy                      # your name for the policy
version: "1.0.0"                     # your version or checkpoint label
image: registry.example.com/team/my-policy@sha256:<digest>
checkpoint: /app/checkpoints/my-checkpoint   # where the weights are inside the image
serve:
  command: [python, setup_policy_server.py, --config_path, policy/my_policy/deploy.yml, --host, 0.0.0.0]
  port: 6000
  startup_timeout_s: 600             # time to load the model and open the port
resources:
  gpu_count: 1
  min_gpu_memory_gb: 24
secrets: []                          # names of environment variables Moravec must provide, if any
interface:
  robot: yam                         # a robot in utils/robot/_robot_info.json
  action_type: joint                 # joint or ee
  cameras: [top, left_wrist, right_wrist]
  chunk_steps: 30                    # actions returned per get_action call
  batching: false                    # whether get_action_batch is supported
```

| Field | Meaning |
| --- | --- |
| `image` | Image reference pinned by `sha256` digest. A tag alone is not accepted. |
| `checkpoint` | Path of the weights inside the image. |
| `serve` | Command that starts the policy server, its port, and how long startup may take. |
| `resources` | GPUs and minimum GPU memory needed to load and run the model. |
| `secrets` | Names (never values) of environment variables the policy needs at run time, such as an API key. |
| `interface` | The observation and action format described below: robot, `joint` or `ee` actions, camera names, chunk length, and batching support. A policy with its own native format (like the GR00T example) sets `format: native` and describes its keys in `notes`. |

Before submitting, validate the policy with `moravec-validate`, installed by `pip install -e .`:

```sh
moravec-validate policy/my_policy                                # manifest and files only
moravec-validate policy/my_policy --run --image my-policy:local  # start the image with Docker and test it
moravec-validate policy/my_policy --url ws://127.0.0.1:6000      # or test a server you already started
```

It checks the manifest against [`schemas/moravec-policy.v1.json`](schemas/moravec-policy.v1.json), looks for credentials in the policy folder, then starts the policy (with `--run`) and tests the contract: readiness, reset, and synthetic observations built from `interface`, checking the returned actions' count, keys, shapes, finite values, gripper range, unit quaternions, batching and latency. Nothing is executed on a robot. Add `--submit` to require a digest-pinned image, `--docker-arg` to pass mounts or other `docker run` options, and `--json` for a machine-readable report in which every failure carries a stable `code` and a `fix`. Moravec runs the same checks before any evaluation, then connects its evaluator to the policy over a private network.

## Interface

Each `policy/<name>/model.py` exposes `Model(model_cfg)` with:

- `reset()`: clear episode state.
- `update_obs(obs)`: receive the current observation.
- `get_action()`: return an ordered list of action dictionaries.
- Optional `update_obs_batch(obs_list)` and `get_action_batch(env_idx_list=None)` for adapters that support batching.

Observations carry an instruction, named cameras under `vision`, and robot state under `state`. The demo roundtrip shows three RGB cameras and joint state. That is an example, not a universal model requirement: use the selected policy's documented camera and state mapping. Images are decoded at the server boundary by the existing image utilities. GR00T instead takes its checkpoint's native `video`, `state`, and `language` dictionaries; see its example for the LIBERO mapping.

For our bimanual joint examples, actions use `left_arm_joint_state`, `right_arm_joint_state`, `left_ee_joint_state`, and `right_ee_joint_state`. End-effector examples replace arm joint arrays with `left_ee_pose` and `right_ee_pose`; the demo uses `[x, y, z, qw, qx, qy, qz]`. A one-arm robot such as `droid` uses the same keys without the prefix: `arm_joint_state` or `ee_pose`, plus `ee_joint_state`. Grippers (`ee_joint_state`) run from 0 closed to 1 open. Moravec's DROID stations report `ee_pose` as the `panda_hand` frame (where the Robotiq gripper mounts) in the arm's base frame; the point between the fingertips is 0.1628 m further along its z axis. Coordinate frames and absolute/delta interpretation must agree with the checkpoint and evaluator. The evaluator executes action chunks and supplies subsequent observations.

## Debug values

A policy can attach intermediate values to a chunk, and Moravec records them in the episode's trace next to what the robot did: a value estimate, attention weights, a planner's subgoal, the number of diffusion steps. Put a `debug` dict on the **first** action of the chunk:

```python
chunk[0]["debug"] = {
    "value": 0.83,                      # a scalar
    "vla.attn": attn,                   # an array, e.g. float32 (heads, tokens)
    "subgoal": np.array([x, y, z]),     # one value for the whole chunk
    "row_value": per_row_values,        # or one per row: leading dimension = chunk length
}
```

- The map describes the whole chunk (one inference). For per-row data use a leading dimension equal to the number of actions.
- Names are lowercase words of `[a-z0-9_]`, optionally grouped with dots (`vla.attn`), at most 48 characters.
- Values are numbers or numeric arrays (booleans become 0/1), at most rank 4. A chunk carries at most 32 names, 4096 elements per value and 16384 in total.
- Only the first action may carry `debug`; any other key is still rejected. Debug values never change what the robot does, and omitting them changes nothing.
- A value whose shape differs from that name's first appearance in the episode, or that breaks the caps, is dropped and counted rather than failing the episode. `moravec-validate` reports the same rules as `ACTION_DEBUG` and `ACTION_DEBUG_STEP` failures.

The demo policy reports `inference` (calls since reset) and `action_norm` (one per action).

## Configuration

`deploy.yml` selects the policy, host, port, action type, environment configuration, and model-specific checkpoint settings. Install the core with `python -m pip install -e .`; follow the model README for its separate runtime and weights.

The installed Python namespace is `moravec_policy`; the checkout folder can have any name. Install it with `python -m pip install -e .` in each policy/evaluator environment. External evaluators should import `moravec_policy.policy.<name>.deploy`. `env_cfg_type` names a robot in `utils/robot/_robot_info.json`, which supplies its arm and gripper dimensions; add an entry there for a new robot.

## Remote serving

On the policy machine, with its runtime activated and checkpoint configured:

```sh
python setup_policy_server.py --config_path policy/GR00T_N17/deploy.yml \
  --overrides host=127.0.0.1 port=6000
```

For a quick remote test, tunnel the endpoint from the evaluator machine:

```sh
ssh -N -L 6000:127.0.0.1:6000 user@policy-host
```

The evaluator connects using the existing client:

```python
from client_server.model_client import WsModelClient

with WsModelClient(url="ws://127.0.0.1:6000",
                   evaluation_id="example", trial_id="trial-1") as client:
    client.call("reset")
    client.call("update_obs", obs=observation)  # supplied by the evaluator
    actions = client.call("get_action")
```

The checkpoint and evaluator configuration must already exist; this command does not provision them. The server has no built-in authentication/TLS, so the example binds loopback and uses SSH. A custom HTTP/gRPC provider requires a policy adapter. The LLM example (Astra) calls the OpenAI API from inside the policy server, so only that machine needs `OPENAI_API_KEY`.

## Quick validation

```sh
python scripts/demo_roundtrip.py
```

This creates a temporary evaluator layout and local WebSocket server. Two cases per registered robot (joint and end-effector) check action shapes, observation transport, batching, and reset for the registered robot dimensions. It uses synthetic images and the zero-action demo; it neither runs model inference nor establishes simulation or physical task success. Never send the demo's zero actions to physical hardware.

## Editing an adapter

Copy `policy/demo_policy/` and set `policy_name` in `deploy.yml` to the new folder name. Implement preprocessing, inference, and action conversion in `model.py`. Document required cameras, state, units, frames, checkpoint, and trained embodiment. The evaluator owns simulation, physical control, and scoring.

Use the Python server plus YAML as the primary serving entry point. Training and shell launch wrappers are not included. Each model README installs the model from its original repository at a pinned commit; clones land in ignored directories and weights are separate. Keep weights, credentials, and run outputs outside Git.

After upgrading an existing checkout, reinstall the editable package in each runtime.
