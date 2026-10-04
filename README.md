# Moravec policy starter

A proposed, Moravec-owned container contract for running models against simulation and physical stations. Version 0.2 is an executable prototype, not a qualified robot deployment.

The station sends observations to `POST /v1/policy/step`. The policy container returns a typed decision. The station validates that decision, executes it through its controller, and returns measured feedback with the next observation. Scoring belongs to an independent evaluator; a model saying “done” does not establish success.

Two HTTP interfaces have different jobs:

```text
Station → Moravec policy endpoint → model backend
        ← typed robot decision   ← model output
Station validates / plans / executes → next observation + measured feedback
```

- An LLM backend receives images, state, instructions, tool definitions, and history over HTTP. Our harness turns its tool call into a decision.
- A learned robot policy receives checkpoint-specific image/state tensors. Our adapter maps its action output into the same decision contract.
- A world-action model can use that action path. A video-only world model needs a planner or action-selection policy; predicted video alone cannot command a robot. That planner is not implemented here.

## Examples

| Exact normalized model name | Backend | Current status |
| --- | --- | --- |
| `gpt-6-astra` | Configurable HTTP chat-completions endpoint | Harness implemented; endpoint is an explicit placeholder, no real inference validated |
| `gr00t-n1.7-3b` | NVIDIA model SDK, local checkpoint | Driver supplied; requires checkpoint/embodiment qualification and GPU image |
| `openwam-alpha-pretrain-foundation-model` | OpenWAM SDK, local checkpoint | Driver supplied; base pretrained checkpoint requires adaptation and qualification |
| `contract-fixture` | Deterministic test policy | Runnable; never a leaderboard model |

Names preserve version dots and otherwise use lowercase dashes. `provider_model_id` separately preserves the exact provider identifier, including namespace/capitalization when required. Never silently replace a requested model with another model.

## Run locally

```sh
uv sync --extra sim
uv run moravec-policy validate examples/gpt-6-astra/policy.json
uv run pytest -q
uv run python sim/validate.py
```

Simulation writes a report, decision/feedback events, and video to `runs/conformance/`. This is a two-joint MuJoCo interface test with deterministic models, not an Isaac Sim benchmark or a real model evaluation.

Serve the fixture:

```sh
export MORAVEC_POLICY_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run moravec-policy serve examples/contract-fixture/policy.json
```

For a container, keep that token set and run `docker compose up --build`. The default container runs the fixture on an internal network, as a nonroot user, with a read-only root filesystem. Model images require their own pinned CUDA/runtime dependencies; the generic image does not include model weights or SDKs. The API-connected profile needs a separately reviewed egress configuration.

## Read next

- [HTTP contract and harness inputs/outputs](docs/harnesses.md)
- [Simulation qualification plan](docs/simulation-validation.md)
- [Updated customer requirements and deployment boundaries](docs/customer-requirements.md)
- [Validation evidence](VALIDATION.md)
- [Generated JSON schemas](schemas/)

Model access, deployment, and results delivery are separate contracts. Team, billing, and budget management are outside this starter.
