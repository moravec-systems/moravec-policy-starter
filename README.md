# Moravec policy examples

Examples for integrating customer policies with Moravec evaluation. Each policy has its own model configuration and runtime, with a shared WebSocket/MessagePack client/server. See the [integration guide](INTEGRATION.md) for the interface and remote serving.

## Examples

| Folder | Purpose |
| --- | --- |
| [demo_policy](policy/demo_policy) | Synthetic actions for testing client/server wiring; not a learned policy or a physical robot demo |
| [GPT_6_Astra_Direct_EEF](policy/GPT_6_Astra_Direct_EEF) | LLM agent; upstream inference and Cartesian execution currently run in the evaluator client |
| [GR00T_N17](policy/GR00T_N17) | NVIDIA GR00T VLA |
| [Pi_05](policy/Pi_05) | Physical Intelligence π0.5 VLA |
| [OpenWAM](policy/OpenWAM) | World-action model |
| [MolmoAct2](policy/MolmoAct2) | Ai2 BimanualYAM VLA; absolute joint actions |

Use each example's README and `deploy.yml` for its dependencies, checkpoint, and launch settings. GPU runtimes and checkpoints are installed separately. Model and agent implementations are fetched on demand with `scripts/fetch_policy.py` or, for MolmoAct2, as part of its pinned Hugging Face snapshot; they are not stored in Git. Each policy README gives its runtime setup commands.

## Submitting a policy

1. Build and test your policy container. Start with the [CPU-only demo](policy/demo_policy/README.md#container-quickstart) for transport, or [GR00T](policy/GR00T_N17/README.md#container-quickstart) for the learned-model integration.
2. Publish the image to your own private registry and identify its immutable digest.
3. Register the image reference and configuration with Moravec; arrange scoped read-only registry access separately. Never put credentials in the configuration file.
4. Moravec will pull and validate the image, then run selected evaluations through the upcoming backend.

The [web demo](https://moravec-evaluation-demo.vercel.app/demo) currently checks registration configuration only. It does not pull images, load weights, start evaluations, or verify model compatibility. The JSON sample is a UI fixture, not an adopted backend schema. See [Policy handoff](INTEGRATION.md#policy-handoff) for what to provide and how deployment will work.

GR00T's included checkpoint is a LIBERO simulation example, not a YAM-qualified policy. Use the container check client to verify inference with your checkpoint. Other learned-model containers will follow.

## Hardware scope

All Moravec setups in this scope are bimanual. The names below identify hardware, not validation claims.

| Name | Configuration key | Status |
| --- | --- | --- |
| YAM | `yam` | Primary target; six arm joints and one gripper coordinate per side registered |
| ARX X5 | `arx_x5` | Retained for reference benchmark integration |
| Franka FR3 Duo | `franka` | Planned; upstream generic Franka dimensions retained, exact FR3 setup not yet qualified |

`utils/robot/_robot_info.json` contains dimensions only. It is not a checkpoint compatibility guarantee or an enforced runtime allowlist. The evaluator's robot configuration must agree on dimensions, joint order, units, gripper conventions, and camera inputs. A pretrained checkpoint may require embodiment-specific training.

## Integration

Install the shared transport in a dedicated Python environment:

```sh
python -m pip install -e .
```

Run the hardware-free example:

```sh
python scripts/demo_roundtrip.py
```

It starts a local WebSocket server, sends three synthetic RGB camera images and joint state, and checks action shapes, batching, and reset for YAM, ARX X5, and generic Franka dimensions in joint and end-effector modes. Temporary evaluator configuration is deleted on exit. It does not run learned inference or simulation, and the demo ignores observation content.

The evaluator sends observations through the existing client; the policy implements `update_obs`, `get_action`, and `reset` (and batching where supported). The evaluator owns simulator or hardware control, task setup, and scoring.

Keep the evaluator's `env_cfg/` beside this checkout; the checkout folder can have any name. Runtime helpers read robot metadata from `env_cfg/robot/_robot_info.json`; when using YAM, copy the `yam` entry from this repository into that evaluator registry. Matching environment and simulator YAML files are also required. This repository alone does not supply a runnable YAM simulator.

The demo emits zero actions, which can command motion under absolute control. Use it only for offline or simulation wiring checks.

## Development

See [INTEGRATION.md](INTEGRATION.md) for implementing and serving a policy, and [THIRD_PARTY.md](THIRD_PARTY.md) for source attribution and licenses.
