# Moravec policy starter

Wrap your model in a container that accepts robot observations and returns actions. Moravec runs that container against its simulator or physical station and scores the resulting episode independently.

**Status:** executable interface prototype. MuJoCo tests use deterministic fixtures by default. GR00T and OpenWAM inference are not qualified. See [validation status](VALIDATION.md).

## Try it

```sh
uv sync --extra sim
uv run pytest -q
uv run python sim/validate.py
```

The test sends RGB images and robot state over HTTP, executes returned actions in a small MuJoCo scene, and saves video, events, and a report in `runs/conformance/`. Numeric targets are supplied explicitly: this checks integration, not perception or benchmark performance.

## Use Astra through OpenAI

Set `OPENAI_API_KEY` securely in your environment, then:

```sh
uv run python sim/validate.py --live-astra --output runs/astra-live
```

This makes **at most two paid calls** for one simple simulation episode. The [Astra manifest](examples/gpt-6-astra/policy.json) uses `gpt-6-astra` directly at `https://api.openai.com/v1/responses`, with no gateway or model substitution. The request contains image inputs, robot state, instruction and strict function tools. We return measured execution feedback as `function_call_output`, retain response reasoning items between calls, and use `store: false`. API data-handling terms still apply; this is not a claim of zero retention.

The harness converts a model tool call into a validated station decision. It never gives the LLM direct hardware access. OpenAI references: [Responses quickstart](https://developers.openai.com/api/docs/quickstart), [function calling](https://developers.openai.com/api/docs/guides/function-calling), [reasoning state](https://developers.openai.com/api/docs/guides/reasoning).

## Implement a policy

Implement two methods: `reset()` and `predict(observation, robot_profile) -> Decision`. The [fixture policy](src/moravec_policy/policies.py) is the smallest working example; [native.py](src/moravec_policy/native.py) shows learned-model adaptation. Use [create_app](src/moravec_policy/service.py) to expose your implementation without rewriting the HTTP service.

| Boundary | Input | Output |
| --- | --- | --- |
| Station → policy container | RGB cameras, named joint state, TCP poses, instruction, timestamps and prior execution feedback | Named-joint trajectory, Cartesian goal, or stop |
| Harness → OpenAI | Images, state, instruction, tool schemas and conversation state | Structured function call |
| Adapter → local model | Checkpoint-specific image/state tensors and language | Checkpoint-specific actions translated into the station's units and joint/frame names |

The [Python contracts](src/moravec_policy/contracts.py) and generated [OpenAPI](schemas/policy.openapi.json) are authoritative. Units: meters/radians, seconds, WXYZ quaternions. Robot profile IDs/revisions bind joint limits, frames, cameras and timing. A model's stop request is not a successful score.

The HTTP lifecycle is `POST /v1/episode/start`, repeated `POST /v1/policy/step`, then `POST /v1/episode/end`, authenticated by `MORAVEC_POLICY_TOKEN`. One container replica handles one active episode. The station checks expiry against its own current clock and owns planning, execution, safety and scoring. The included planar executor is only a test fixture, not a production controller.

## Package for evaluation

1. Agree on the task, robot profile, observation channels and action representation with Moravec.
2. Adapt the model without changing the contract, evaluator, scene, seeds or scoring rules.
3. Run contract tests and the local simulator. Qualify the actual checkpoint in the agreed Isaac Sim task before claiming readiness.
4. Build a model-specific container with pinned runtime/SDK versions. Submit its immutable image digest, policy manifest, checkpoint/binding hashes, resource requirements and validation report. Supply registry access through a separate secret channel.

`docker compose up --build` runs the deterministic fixture once `MORAVEC_POLICY_TOKEN` is set. The default image contains neither GPU SDKs nor model weights. For an isolated customer deployment, all inference assets must be local and no internet is allowed; an external OpenAI API deployment is a separate, explicitly API-connected profile. Results export is outside the inference container.

## Model examples

- [gpt-6-astra](examples/gpt-6-astra): direct OpenAI Responses harness.
- [gr00t-n1.7-3b](examples/gr00t-n1.7-3b): NVIDIA SDK adapter template; needs a CUDA environment, checkpoint and verified embodiment bindings.
- [openwam-alpha-pretrain-foundation-model](examples/openwam-alpha-pretrain-foundation-model): world-action adapter template; the base pretrained checkpoint needs adaptation. A video-only world model additionally needs an action-selection planner, which is not provided here.

Bindings marked `requires-qualification` intentionally refuse inference. Do not change that flag until action units, order, normalization, timing and embodiment compatibility have been tested with the actual checkpoint.

## Using an agent

Give the agent this repository, the model SDK/checkpoint reference, the agreed robot profile, and access to a suitable runtime. Ask it to **implement the adapter and prove compatibility**, not recreate the evaluation system. [AGENTS.md](AGENTS.md) supplies the adaptation and evidence requirements. Missing checkpoint metadata or incompatible action semantics must be reported rather than guessed.

For NVIDIA, share a clean snapshot containing this README, contracts, adapter example, container recipe and conformance tests. Keep internal research, SOW discussion, private results and credentials separate. This private development repository's Git history contains earlier internal notes, so do not treat its full history as the customer handoff package.
