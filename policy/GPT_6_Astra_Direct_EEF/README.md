# GPT-6 Astra

Moravec's retained LLM agent example uses model id `gpt-6-astra`. It receives RGB observations and calls `move_eef` to request world-frame grasp poses. The evaluator's Cartesian planner converts those goals into joint paths.

This is not currently a self-contained remote policy server: `model.py` is a placeholder, while agent inference runs in the evaluator client. `action_type` remains `joint` at the evaluator boundary. Batch evaluation is not supported.

Fetch its agent runtime with `python scripts/fetch_policy.py GPT_6_Astra_Direct_EEF` from the repository root (Python 3.12+). The downloaded `inspect/` tree is ignored by Git. Install the core and agent dependencies in a dedicated environment; the shared server can then be launched directly. The agent runtime needs `openai==3.8.0` and `pillow==12.3.0`. The adapter defaults to the direct OpenAI Responses API and reads `OPENAI_API_KEY`. Explicit `L4_INSPECT_BASE_URL` and `L4_INSPECT_API_KEY_ENV` overrides remain available. Do not put keys in deployment YAML or source control.

A full episode requires the external evaluator, robot assets, and Cartesian planner expected by `inspect/`. The adapter still uses legacy evaluator configuration identifiers internally; these are functional dependencies, not a standalone Moravec simulator. Model-server startup alone does not validate LLM execution.

See the [integration guide](../../INTEGRATION.md) for the shared interface. The simulation checks use that external evaluator; this remains an evaluator-integrated example, not a self-contained remote LLM service.

## Validation

On 2026-10-05, direct OpenAI `gpt-6-astra` completed four Isaac Sim trials with the existing Cartesian agent: Stack Bowls 0/2 successes (15% and 0% partial credit); Stack Blocks 1/2 successes (100% and 0%). Trials used layouts 0 and 1, medium reasoning effort, and a 40-call limit. They used 38, 39, 32, and 40 model calls respectively. All reached scored outcomes without provider failures; these small trials validate integration, not reliable task completion.
