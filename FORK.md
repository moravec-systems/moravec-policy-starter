# Moravec restart baseline

This branch restarts from XPolicyLab source at `45ae74c1a427206f62d55f08f48f00295d2dc1a3` (https://github.com/XPolicyLab/XPolicyLab). The previous Moravec starter is preserved on `main` at `dc7cc5f372ef1066e0fe07c2e6651cb6058328dc`.

The initial import preserves upstream policy folders, ModelTemplate, transports, deployment configuration, robot metadata, licenses, and attribution. In particular:

- `policy/demo_policy`: original minimal wiring example.
- `policy/GPT_6_Astra_Direct_EEF`: existing Astra example; its LLM and Cartesian executor run on the environment-client side. It is not the previous Moravec HTTP policy adapter.
- `policy/GR00T_N17` and `policy/OpenWAM`: upstream model implementations and configuration.
- `client_server`, `utils/robot`, and `model_template.py`: unchanged upstream interfaces.

Initial deviations are limited to this note, local credential/artifact ignore rules, and removal of root AGENTS.md/CLAUDE.md as previously requested. The import retains the full upstream policy set before targeted cleanup. One upstream Git LFS figure, `policy/Spirit_v15/spirit_v15/assets/rc_results.png`, was unavailable; its original pointer remains. It is not required by the demo, Astra, GR00T, or OpenWAM runtime.

## Verification

Installed upstream core dependencies in a separate local virtual environment. The unmodified demo and shared WebSocket implementation passed reset/update/get-action checks for single-arm and dual-arm configurations, in both joint and end-effector modes. Temporary test workspace configuration followed upstream's parent `env_cfg/` layout. These are four wiring checks, not model inference or simulation episodes.

No old validation claim automatically transfers to this branch. Astra API calls, GPU checkpoints, Isaac Sim integration, and physical robots have not been requalified here. The previous custom protocol, manifest schema, and LIBERO wrapper examples were not imported.

## Focused Moravec customization

Retained demo_policy, GPT_6_Astra_Direct_EEF, GR00T_N17, Pi_05, and OpenWAM; removed other tracked adapters. The unavailable Spirit illustration was removed with that unused adapter. Original root documentation is archived as UPSTREAM_README.md and UPSTREAM_README_zh.md; README.md describes this fork's actual scope. Model README links were adjusted to the archived documentation.

The dimension registry now includes YAM and retains ARX X5 and generic Franka. Other robot entries were removed. All intended Moravec setups are bimanual; use the display name YAM. Franka dimensions do not establish FR3 Duo compatibility. Runtime evaluator metadata still needs a matching YAM entry.

No lifecycle, transport, model inference, or evaluator lookup behavior changed. Configuration unification, expanded startup validation, and independent container packaging are deferred until integration evidence and NVIDIA collaboration justify them. Keep upstream model names and configuration conventions.
