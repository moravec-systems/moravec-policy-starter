# GPT-6 Astra

Moravec's retained LLM agent example uses model id `gpt-6-astra`. It receives RGB observations and calls `move_eef` to request world-frame grasp poses. The evaluator's Cartesian planner converts those goals into joint paths.

This is not currently a self-contained remote policy server: `model.py` is a placeholder, while agent inference runs in the evaluator client. `action_type` remains `joint` at the evaluator boundary. Batch evaluation is not supported.

The server currently uses the Pi_05 `openpi` Python environment through `policy_uv_env_path`. The agent runtime needs `openai==3.8.0` and `pillow==12.3.0`. The retained provider configuration uses `ARK_API_KEY`; do not assume this fork has already been configured or tested against a direct OpenAI endpoint.

A full episode requires the external evaluator, robot assets, and Cartesian planner expected by `inspect/`. The adapter still uses legacy evaluator configuration identifiers internally; these are functional dependencies, not a standalone Moravec simulator. Model-server startup alone does not validate LLM execution.

See the [integration guide](../../INTEGRATION.md) for the shared interface. This example needs evaluator integration before it can serve as a runnable customer LLM demo.
