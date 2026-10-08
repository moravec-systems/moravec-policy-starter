# Moravec policy examples

Serve your robot policy to Moravec's evaluator. Each example wraps one model behind the same small interface and a shared WebSocket/MessagePack server; the evaluator owns the simulator or robot, task setup, and scoring.

## Quickstart

```sh
python -m pip install -e .
python scripts/demo_roundtrip.py
```

The demo starts a local policy server, sends synthetic camera images and joint state, and checks action shapes, batching, and reset for every registered robot in joint and end-effector modes. It runs no learned model. Its zero actions can command motion under absolute control, so never send them to hardware.

## Interface

A policy is a `policy/<name>/model.py` exposing `Model(deploy_cfg)` with `reset()`, `update_obs(obs)`, and `get_action()` (plus optional batch versions). `deploy.yml` beside it selects the checkpoint and serving settings, and `setup_policy_server.py` serves it. See the [integration guide](INTEGRATION.md) for the observation and action format, remote serving, and how to write your own adapter.

## Examples

| Folder | Model | Container |
| --- | --- | --- |
| [demo_policy](policy/demo_policy) | Synthetic actions for wiring checks | CPU |
| [GR00T_N17](policy/GR00T_N17) | NVIDIA GR00T N1.7 VLA (LIBERO checkpoint) | GPU |
| [Pi_05](policy/Pi_05) | Physical Intelligence π0.5 VLA via openpi | GPU |
| [OpenWAM](policy/OpenWAM) | OpenWAM world-action model | GPU |
| [MolmoAct2](policy/MolmoAct2) | Ai2 MolmoAct2-BimanualYAM VLA | GPU |
| [GPT_6_Astra_Direct_EEF](policy/GPT_6_Astra_Direct_EEF) | LLM agent choosing end-effector goals via the OpenAI API | CPU |

Each README covers setup, weights, the container, and a `check_client.py` that requests real inference without executing actions. Model code is installed from its original repository at a pinned commit (or, for MolmoAct2, its pinned Hugging Face snapshot); weights are downloaded separately and never stored in Git or images.

To run a YAM policy closed loop before real evaluation, use the [YAM MuJoCo check](examples/yam_mujoco/README.md): a bimanual YAM tabletop scene with top and wrist cameras for joint or end-effector policy servers.

## Robots

`env_cfg_type` in `deploy.yml` names a robot in `utils/robot/_robot_info.json`, which supplies its arm and gripper dimensions. All setups are bimanual:

| Robot | Key | Status |
| --- | --- | --- |
| YAM | `yam` | Primary target: six arm joints and one gripper per side |
| ARX X5 | `arx_x5` | Used by the π0.5 and OpenWAM simulation checkpoints |
| Franka FR3 Duo | `franka` | Planned; generic Franka dimensions, FR3 setup not yet qualified |

Matching dimensions do not make a checkpoint compatible: joint order, units, gripper convention, and cameras must also agree with the evaluator, and a pretrained checkpoint may need embodiment-specific training.

## Submitting a policy

Package the policy as a container image like the examples and describe it with a `moravec-policy.yaml` manifest. See [Submitting a policy](INTEGRATION.md#submitting-a-policy) for the format; every example folder includes a sample manifest.

Source attribution and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md).
