# Astra: LLM plus RoboProbe reference harness

Use `gpt-6-astra`, not GPT-Sol or an alias. This example pins RoboProbe at the revision used for the prior corrected Opus integration. The successful Opus episode is evidence for that harness integration only, not a tested Astra run from this repo.

## Inspect the launch without inference

```bash
uv run moravec-policy astra-plan examples/astra/policy.json \
  --root /path/to/RoboProbe --task stack_bowls --layout 0 --eval-env uv
```

## Prepare and execute on a GPU host

1. Clone the pinned source from `sources.lock.json`; keep a clean checkout. Install its dependencies and follow upstream `docs/setup.md` to arrange the sibling RoboDojo environment and `env_cfg`.
2. Supply access to the exact Astra model. `ASTRA_API_KEY` and `ASTRA_BASE_URL` refer to an authorized provider endpoint, not a Codex subscription. Upstream's Astra preset uses the Responses API. A generic OpenRouter chat URL is **not** a drop-in replacement; use an independently validated provider transport if needed and disclose its revision. No provider keys from the current workspace are copied into this sample.
3. Ensure the native runtime is supported. The upstream setup includes Isaac 5.1-era assumptions. Our earlier Isaac 6.1 migration required bootstrap/camera fixes; this sample does not silently apply them. Record and validate such a migration separately.
4. Set `ROBODOJO_ROOT` to the prepared simulator checkout. Review the command, then run it explicitly:

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo-eval
export ASTRA_BASE_URL='<authorized Responses-compatible provider base>'
# Inject ASTRA_API_KEY from your secret manager, not source control.
uv run moravec-policy astra-plan examples/astra/policy.json \
  --root /path/to/RoboProbe --task stack_bowls --layout 0 --eval-env uv --execute
```

The wrapper pins model/preset, medium reasoning, 170 maximum calls, two-image history setting, RGB-only input and disables automatic shell retries. The reference Responses path may retain images differently from chat; consult the pinned harness implementation. It invokes `RoboDojo_Agent_L3_Inspect_EEF/run_fixed_layout.sh`. The reference planner converts grasp-point targets to planned joint paths and reports execution feedback. Scoring stays in RoboDojo.

The model must not receive reward state, hidden object poses, or layout metadata. The runner's fixed layout is orchestration data, not a model input. A single-layout run is a smoke evaluation, not the official 2,100-trial score. Set external time/spend/shutdown guards before paid execution; this wrapper's call cap is not a dollar cap or GPU shutdown service.
