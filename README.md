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

Use each example's README and `deploy.yml` for its dependencies, checkpoint, and launch settings. GPU runtimes and checkpoints are installed separately. Model and agent implementations are fetched on demand with `scripts/fetch_policy.py`; they are not stored in Git. Each policy README gives its runtime setup commands.

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

For now, preserve the upstream workspace layout: this checkout is `XPolicyLab/`, alongside the evaluator's `env_cfg/`. Runtime helpers read robot metadata from `env_cfg/robot/_robot_info.json`; when using YAM, copy the `yam` entry from this repository into that evaluator registry. Matching environment and simulator YAML files are also required. This repository alone does not supply a runnable YAM simulator.

The demo emits zero actions, which can command motion under absolute control. Use it only for offline or simulation wiring checks.

## Validation and scope

Validation on 2026-10-05 used an NVIDIA L40S for learned policies and simulation, and the direct OpenAI API for Astra. GR00T ran in LIBERO/MuJoCo; the other policies use an external Isaac Sim evaluator. These different task fixtures are integration checks, not comparable model rankings.

- The hardware-free demo passes six transport checks.
- GR00T: 4/4 successful episodes across two LIBERO tasks, plus a successful repeat using the included `scripts/validate_gr00t_libero.py`. A Mac client also retrieved real inference over an SSH tunnel.
- π0.5: Stack Bowls 2/2 successes; Stack Blocks 0/2 successes (15% and 0% partial credit).
- OpenWAM: Stack Bowls 2/2 successes; Stack Blocks 0/2 successes (15% partial credit each).
- Astra: direct OpenAI API, Stack Bowls 0/2 successes; Stack Blocks 1/2 successes, with a 40-call limit per trial.

GR00T, π0.5, and OpenWAM adapter outputs matched native inference on captured simulator observations. This checks conversion consistency; it does not imply every task succeeds. See each policy README for checkpoint revisions and settings. Raw traces, videos, weights, and credentials remain outside Git.

YAM and FR3 physical operation remains unqualified. Next steps are matching trained checkpoints to those embodiments and validating their cameras, frames, joint order, and control behavior. Deployment packaging and any broader compatibility schema can be refined with NVIDIA. Training and full benchmark orchestration are outside this starter's scope.

## Development

See [INTEGRATION.md](INTEGRATION.md) for implementing and serving a policy, and [THIRD_PARTY.md](THIRD_PARTY.md) for source attribution and licenses.
