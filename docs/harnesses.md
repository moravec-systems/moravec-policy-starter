# Model access and harnesses

A model performs inference. A harness supplies context, calls the model, validates its output, executes allowed tools through a station controller, records feedback, and repeats. The harness is part of the evaluated system: pin its version, prompts, tools, context policy, decoding settings, checkpoint bindings, and controller configuration alongside the model.

## LLM HTTP boundary

`examples/gpt-6-astra/policy.json` specifies a full HTTP endpoint, exact model identifier, environment-variable credential reference, timeout, maximum calls, output token budget, and reasoning effort. The example URL is intentionally non-routable. Configure a provider that actually exposes the requested model; the Codex model selector is not proof of public API access.

This implementation supports a chat-completions-compatible transport. Other transports can implement the same internal policy interface without changing the station contract. API credentials stay in the container, never in station observations or source control.

Illustrative model request (image abbreviated):

```json
{
  "model": "gpt-6-astra",
  "messages": [
    {"role": "system", "content": "Use the declared station tools and measured feedback."},
    {"role": "user", "content": [
      {"type": "text", "text": "Instruction, named joint positions, TCP poses, robot profile and prior feedback"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}
  ],
  "tools": [{"type": "function", "function": {
    "name": "move_tcp", "description": "Request a bounded Cartesian goal",
    "parameters": {"type": "object", "properties": {"kind": {"const": "cartesian-goal"}}}
  }}],
  "tool_choice": "required", "parallel_tool_calls": false,
  "max_tokens": 4096, "reasoning_effort": "medium", "stream": false
}
```

The abbreviated tool schema above is explanatory only. The running harness generates complete schemas from `CartesianGoal` and `Stop`; inspect `src/moravec_policy/llm.py` or the recorded request in tests. A model response has `choices[0].message.tool_calls`, with a call ID, function name, and JSON-string arguments. The harness accepts exactly one known tool call and rejects a mismatched returned model ID.

`move_tcp` arguments specify `kind`, `arm`, a target pose (`frame`, meters, WXYZ quaternion), duration in seconds, and an optional gripper opening fraction. `stop` requests termination; it does not award a score. On the next call, the harness sends a `tool` message referencing the same call ID and the station's measured execution status. It retains text/tool history and only the latest camera images. Reset clears this state between episodes.

## Station HTTP boundary

Authenticated endpoints:

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /v1/episode/start` | Episode ID and versioned robot profile | Episode started |
| `POST /v1/policy/step` | `Observation` | `Decision` |
| `POST /v1/episode/end` | Episode ID | Episode ended |
| `GET /v1/capabilities` | Bearer token | Policy identity/capabilities |

`Observation`: episode ID, monotonic sequence, station clock timestamps, instruction, profile ID/revision, named joint positions, named RGB cameras with capture times, TCP poses, gripper fractions, and previous-decision feedback. It contains no evaluator reward or hidden target state. RGB transport is raw uint8 HWC encoded as base64; the LLM harness converts it to PNG. This simple transport should be replaced with negotiated binary media for high-rate deployment, without changing image semantics.

`Decision`: matching episode/sequence, unique decision ID, station-clock expiry, evidence kind, and one action: named-joint trajectory, Cartesian goal, or stop. A trajectory contains increasing time offsets and physical joint positions. Robot profiles define radian/meter units, joint limits, frames, camera names, control period, freshness bounds, and maximum action duration.

The station checks the response using its current clock after inference, then performs planning, collision checks, velocity/acceleration checks, and controller execution. This starter implements schema, identity, freshness, frame, duration, and joint-position checks. Production collision avoidance, dynamic limits, watchdogs, and emergency stopping remain station responsibilities and are not implemented here. No model receives arbitrary shell or hardware access.

The service supports one episode at a time; deploy isolated replicas for parallel episodes. Duplicate or out-of-order observations are rejected. Do not blindly retry a possibly executed action after an HTTP failure; reconcile station execution state first. Production resumption/idempotency is a separate qualification requirement.

## Learned-model boundary

| Model | Native input | Native output | Qualification dependency |
| --- | --- | --- | --- |
| `gr00t-n1.7-3b` | Video uint8 `[B,T,H,W,C]`, float state `[B,T,D]`, nested language strings | Named action arrays `[B,T,D]` | Checkpoint modality config, normalization, embodiment and action representation |
| `openwam-alpha-pretrain-foundation-model` | Named PIL camera images, prompt, ordered state vector | Action vector from deployment predictor | Fine-tuned checkpoint, normalization and action/state bindings |

The supplied drivers accept only qualified absolute-joint-position outputs. They deliberately refuse unqualified bindings, missing checkpoint hashes, incompatible profiles, and unsupported temporal inputs. They do not pretend that a base checkpoint can operate an arbitrary robot. End-effector deltas, temporal observation windows, base motion, force control, and predicted-video artifacts need explicit adapters and tests before support is claimed.

A VLM used for reasoning follows the LLM path if it supports images and structured tools. A value/reward model should be a separate evaluator or planner component. A world model used to imagine candidate futures must be scored together with the planner selecting actions. Model family is descriptive; declared input/output capabilities determine compatibility.

## Research basis

- [Anthropic: building effective agents](https://www.anthropic.com/engineering/building-effective-agents) distinguishes model/tool loops and environment feedback; this motivates keeping orchestration outside the model.
- [Anthropic tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) documents client execution and returning tool results.
- [Chat-completions HTTP schema](https://openrouter.ai/docs/api_reference/overview) documents messages, multimodal content, function tools, and responses. Our provider URL is configurable.
- [NVIDIA model card](https://huggingface.co/nvidia/GR00T-N1.7-3B) and [official SDK](https://github.com/NVIDIA/Isaac-GR00T) define the model-specific interface.
- [OpenWAM deployment documentation](https://github.com/OpenWAM-Official/OpenWAM/blob/main/assets/openwam_usage_docs/train-and-deploy.md) and [checkpoint card](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model) distinguish pretrained artifacts from deployment adaptation.

These are research references, not dependencies on another evaluation harness. Moravec owns the service, contract, validation, controller boundary, and test scene in this repository.
