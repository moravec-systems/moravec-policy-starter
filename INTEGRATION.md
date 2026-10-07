# Policy integration

## Policy handoff

The starter defines the policy interface and container examples. The web demo validates registration fields locally and stops at “Configuration ready.” Image retrieval, GPU deployment, runtime validation, simulation scheduling, and results delivery are planned backend work, not implemented by the demo.

### What the customer provides

- Policy name/version and a private registry image reference pinned by digest (`registry/repository@sha256:…`). Local example tags are for development, not published artifacts.
- Checkpoint identity/version and packaging: included in the image, or supplied separately through an agreed delivery method. State any gated dependencies. The GR00T demo allows downloads; it is not an offline deployment package.
- Supported Linux architecture, GPU count/type or minimum VRAM, and expected model-loading time.
- Container startup command, configuration, WebSocket port (6000 in these examples), and readiness behavior.
- Required cameras/state, image layout, action dimensions/units/frames, trained embodiment, and any task-specific assumptions. Matching dimensions alone do not qualify a checkpoint for a robot.

Use the [GR00T example](policy/GR00T_N17/README.md#container-quickstart) as a starting point and the [synthetic demo](policy/demo_policy/README.md#container-quickstart) to check transport without weights or a GPU. The supplied GR00T checkpoint targets LIBERO simulation, not YAM. Use the supplied container check client to verify inference before connecting an evaluator.

### How Moravec will pull and run it

1. The customer publishes the image to its own registry and registers the image reference/configuration. The browser does not receive the image or weights.
2. The customer and Moravec arrange scoped, read-only registry access through an agreed secure channel. Do not include tokens in JSON, Git, Dockerfiles, or browser demo uploads. This credential onboarding is not implemented yet.
3. Moravec's deployment service authenticates to the registry and pulls the specified digest onto approved infrastructure. Registry credentials stay with the deployment service, outside the policy container.
4. Moravec checks startup/readiness, observation/action compatibility, reset, and real inference before running task trials. A successful JSON check is not runtime validation; an inference check is not task success.
5. An evaluation worker connects to the policy over the private network, runs the selected simulator/tasks, and records scores, video, actions, and version metadata. Physical execution requires separate station qualification.

The NVIDIA pilot draft calls for dedicated facility inference machines without open-internet access. That deployment requires weights and dependencies to be available before startup; it is a later deployment target, not a property of today's downloadable-weight example. Customer-hosted remote policy serving is an alternative by agreement, using the same observation/action interface and separately arranged authenticated connectivity.

Registry access can be revoked to stop future pulls; it does not remotely disable an already-downloaded image. Retention/deletion and results-storage access are agreed separately. No registry or results-storage credentials are needed to try the current web demo.

## Interface

Each `policy/<name>/model.py` exposes `Model(model_cfg)` with:

- `reset()`: clear episode state.
- `update_obs(obs)`: receive the current observation.
- `get_action()`: return an ordered list of action dictionaries.
- Optional `update_obs_batch(obs_list)` and `get_action_batch(env_idx_list=None)` for adapters that support batching.

Observations carry an instruction, named cameras under `vision`, and robot state under `state`. The demo roundtrip shows three RGB cameras and joint state. That is an example, not a universal model requirement: use the selected policy's documented camera and state mapping. Images are decoded at the server boundary by the existing image utilities. GR00T's `io_format: native` deliberately preserves its checkpoint-specific `video`, `state`, and `language` dictionaries; see its example for the LIBERO mapping.

For our bimanual joint examples, actions use `left_arm_joint_state`, `right_arm_joint_state`, `left_ee_joint_state`, and `right_ee_joint_state`. End-effector examples replace arm joint arrays with `left_ee_pose` and `right_ee_pose`; the demo uses `[x, y, z, qw, qx, qy, qz]`. Coordinate frames and absolute/delta interpretation must agree with the checkpoint and evaluator. The evaluator executes action chunks and supplies subsequent observations.

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

This creates a temporary evaluator layout and local WebSocket server. Six cases check joint/end-effector action shapes, observation transport, batching, and reset for the registered robot dimensions. It uses synthetic images and the zero-action demo; it neither runs model inference nor establishes simulation or physical task success. Never send the demo's zero actions to physical hardware.

## Editing an adapter

Copy `policy/demo_policy/` and set `policy_name` in `deploy.yml` to the new folder name. Implement preprocessing, inference, and action conversion in `model.py`. Document required cameras, state, units, frames, checkpoint, and trained embodiment. The evaluator owns simulation, physical control, and scoring.

Use the Python server plus YAML as the primary serving entry point. Training and shell launch wrappers are not included. Each model README installs the model from its original repository at a pinned commit; clones land in ignored directories and weights are separate. Keep weights, credentials, and run outputs outside Git.

After upgrading an existing checkout, reinstall the editable package in each runtime.
