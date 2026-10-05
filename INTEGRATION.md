# Policy integration

## Interface

Each `policy/<name>/model.py` exposes `Model(model_cfg)` with:

- `reset()`: clear episode state.
- `update_obs(obs)`: receive the current observation.
- `get_action()`: return an ordered list of action dictionaries.
- Optional `update_obs_batch(obs_list)` and `get_action_batch(env_idx_list=None)` for adapters that support batching.

Observations carry an instruction, named cameras under `vision`, and robot state under `state`. The demo roundtrip shows three RGB cameras and joint state. That is an example, not a universal model requirement: use the selected policy's documented camera and state mapping. Images are decoded at the server boundary by the existing image utilities.

For our bimanual joint examples, actions use `left_arm_joint_state`, `right_arm_joint_state`, `left_ee_joint_state`, and `right_ee_joint_state`. End-effector examples replace arm joint arrays with `left_ee_pose` and `right_ee_pose`; the demo uses `[x, y, z, qw, qx, qy, qz]`. Coordinate frames and absolute/delta interpretation must agree with the checkpoint and evaluator. The evaluator executes action chunks and supplies subsequent observations.

## Configuration

`deploy.yml` selects the policy, host, port, action type, environment configuration, and model-specific checkpoint settings. Install the core with `python -m pip install -e .`; follow the model README for its separate runtime and weights.

The Python import namespace is still `XPolicyLab`. Some evaluator scripts also require a workspace with this checkout named `XPolicyLab/` beside `env_cfg/`. These names are retained compatibility requirements, not a second package to install. Runtime helpers load robot dimensions and simulator batch size from that neighboring configuration. For YAM, copy the local `utils/robot/_robot_info.json` entry into the evaluator registry and supply its environment configuration.

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
from client_server.ws.model_client import WsModelClient

with WsModelClient(url="ws://127.0.0.1:6000",
                   evaluation_id="example", trial_id="trial-1") as client:
    client.call("reset")
    client.call("update_obs", obs=observation)  # supplied by the evaluator
    actions = client.call("get_action")
```

The checkpoint and evaluator configuration must already exist; this command does not provision them. The server has no built-in authentication/TLS, so the example binds loopback and uses SSH. A custom HTTP/gRPC provider requires a policy adapter. Astra is currently different: its model-server adapter is a placeholder, while the agent and Cartesian execution live in the evaluator client.

## Quick validation

```sh
python scripts/demo_roundtrip.py
```

This creates a temporary evaluator layout and local WebSocket server. Six cases check joint/end-effector action shapes, observation transport, batching, and reset for the registered robot dimensions. It uses synthetic images and the zero-action demo; it neither runs model inference nor establishes simulation or physical task success. Never send the demo's zero actions to physical hardware.

## Editing an adapter

Copy `policy/demo_policy/` and set `policy_name` in `deploy.yml` to the new folder name. Implement preprocessing, inference, and action conversion in `model.py`. Document required cameras, state, units, frames, checkpoint, and trained embodiment. The evaluator owns simulation, physical control, and scoring.

Use the Python server plus YAML as the primary serving entry point. Existing shell scripts mainly install dependencies, activate separate environments, and orchestrate evaluator processes; they are optional for serving an already-configured model directly. Their benchmark-specific paths remain compatibility details. Keep weights, credentials, and run outputs outside Git.
