# Moravec policy starter

A small container interface for evaluating a model in simulation or at a robot station. The model adapter handles inference; Moravec owns task execution and scoring. This is a v0.3 prototype, not a qualified hardware deployment.

```text
Station ↔ WebSocket / MessagePack ↔ Policy container
                                     ├─ Astra harness → OpenAI HTTPS API
                                     └─ GR00T / OpenWAM → local model SDK
```

## Run the example

```sh
uv sync --extra sim
uv run pytest -q
uv run python sim/validate.py
```

The MuJoCo test exercises joint trajectories, relative pose chunks, and an LLM tool loop with deterministic fixtures. Reports, events and video go to `runs/conformance/`. Numeric targets are provided explicitly; this is an integration test, not a model benchmark.

To test real Astra, supply `OPENAI_API_KEY` securely, then run:

```sh
uv run python sim/validate.py --live-astra --output runs/astra-live
```

This makes at most two paid calls for one episode using `gpt-6-astra` at `https://api.openai.com/v1/responses`. HTTP is only the model-provider connection. Astra's policy container uses the same station protocol as the other models. No model substitution. See [validation status](VALIDATION.md) for what has actually run.

## Implement an adapter

Use `configure(profile)` once per episode, `reset()` to clear state, `update_obs(observation)` to ingest observations, and `get_action()` to return an action chunk. Observation updates do not trigger inference, so a temporal policy can accumulate history first. [The fixture](src/moravec_policy/policies.py) is the smallest example; [native.py](src/moravec_policy/native.py) demonstrates checkpoint conversion. Batch methods are declared but explicitly unsupported; use isolated replicas for parallel episodes.

Serve an implementation through [create_app](src/moravec_policy/service.py). The service owns episode IDs, request IDs, decision IDs, feedback matching and validation. The adapter returns actions, not scores or trial metadata.

## Interface

The binary socket is `/policy`. Authenticate the handshake with `Authorization: Bearer <MORAVEC_POLICY_TOKEN>`. Use WSS outside loopback. Each request is a MessagePack map:

```text
{protocol: "moravec.policy/v0.3", request_id: "unique-id", method: "...", payload: {...}}
```

Methods: `hello`, `reset` (episode ID and robot profile), `update_obs` ([observation envelope](schemas/ObservationEnvelope.json)), `get_action` (empty payload), and `end` (episode ID). Replies echo protocol/request ID and contain `result` or `error`. One connection owns one container session. Errors require reset; disconnects abort the session. No automatic action retry or cross-connection resume is implemented.

Observation envelopes separate `metadata` from `observation: {instruction, vision, state}`. The [Python types](src/moravec_policy/contracts.py) are authoritative:

- **Vision:** named RGB uint8 HWC byte buffers; optional registered depth as little-endian float32 meters (zero = invalid), 3×3 intrinsics, and 4×4 camera-to-reference-frame extrinsics. Capture timestamps are explicit. Required channels are declared in the policy manifest; missing channels fail validation.
- **State:** named joint positions/velocities, TCP and end-effector poses, gripper joints/opening fractions, and optional mobile-base pose/twist. Position units are meters, revolute joints radians, rotations WXYZ quaternions. Base twist is `[vx, vy, vz, wx, wy, wz]` in the base pose's reference frame.
- **Actions:** timestamped joint chunks, absolute/relative pose chunks, gripper commands, coordinated tracks, or stop. All coordinated tracks start at one station time. Each pose chunk specifies timestep, execution horizon, frame, and relative semantics. Frame-relative deltas use `p'=p+dp, q'=dq*q`; tool-relative deltas use `p'=p+R(q)dp, q'=q*dq`. Deltas reference either the observation pose or the previous predicted point, explicitly. A gripper fraction is 0 closed, 1 open.

MessagePack carries binary image/depth buffers directly; JSON schemas describe those byte fields using their JSON/base64 representation. This protocol is Moravec-versioned; matching lifecycle and robotics conventions does not imply wire compatibility with another server.

The station validates expiry using its current clock, plans and executes only the declared horizon, and sends measured feedback. The simulator pauses its clock during inference; physical latency/watchdog behavior is not validated by this test. Collision avoidance, velocity/force limits and emergency stopping remain station responsibilities.

## Model examples and submission

- [gpt-6-astra](examples/gpt-6-astra): Responses API tools become absolute pose chunks; reasoning/tool history resets between episodes.
- [gr00t-n1.7-3b](examples/gr00t-n1.7-3b): native SDK template with temporal observations and explicit pose/joint bindings. Requires the correct CUDA runtime and checkpoint.
- [openwam-alpha-pretrain-foundation-model](examples/openwam-alpha-pretrain-foundation-model): native world-action template; requires checkpoint adaptation. Video-only world models need a separate action-selection planner.

Bind camera names, state ordering, normalization, observation cadence, rotation representation, action timestep and execution horizon from actual checkpoint metadata. Rotation adapters cover WXYZ, rotation vectors, XYZ Euler radians, and explicitly declared row/column 6D rotations. The SDK must supply denormalized outputs; gripper fractions must already match the declared convention. Placeholder bindings remain `requires-qualification` and refuse inference.

`docker compose up --build` serves the fixture after `MORAVEC_POLICY_TOKEN` is set. The CPU image has no GPU SDKs or weights. Build a separate pinned model image for GPU inference. Isolated deployments have no internet; API-connected models use a separate approved network profile.

Submit the image digest, policy manifest, checkpoint/binding hashes, compute requirements, and executed validation report. Keep credentials, weights and private evaluation results out of Git. A customer or coding agent should implement the adapter against this fixed interface, preserving the evaluator, task assets, seeds and scoring rules. Real checkpoint simulation qualification comes before physical station qualification.
