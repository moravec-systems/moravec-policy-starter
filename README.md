# Moravec policy starter

A small container interface for evaluating a model in simulation or at a robot station. The model adapter handles inference; Moravec owns task execution and scoring. This is a prototype, not a qualified hardware deployment.

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

## Add or customize a policy

```text
policies/<model-name>/    policy.py, policy.json, Dockerfile, optional bindings
robots/                  reusable embodiment definitions, including YAM and dual YAM
src/moravec_policy/
  client_server/         shared client, default server and binary encoding
  lifecycle.py           policy interface
  contracts.py           shared observations/actions, independent of model SDKs
sim/                     MuJoCo contract test and real-checkpoint LIBERO runner
tests/                   policy, transport and robot-definition compatibility
```

Policy IDs preserve upstream names (`GR00T_N17`, `OpenWAM`); checkpoint-specific examples live in their `libero/` subfolders. Provider model IDs, artifact revisions, and checkpoint hashes distinguish the actual weights.

Each folder owns its SDK, preprocessing, observation history, inference settings, action conversion and container recipe. `policy.json` declares `entrypoint: "policy.py:Model"` (or an installed `package.module:factory`) and a policy-owned `config` object. The callable receives `(manifest, manifest_path)` and returns a `Policy`. Adding a model never requires a core driver or model-family registration. Entrypoints are trusted container code, not network-request fields.

Use `configure(profile)` once per episode, `reset()` to clear state, `update_obs(observation)` to ingest observations, and `get_action()` to return an action chunk. Observation updates do not trigger inference. Start with the [fixture](policies/contract-fixture/policy.py), [Astra](policies/gpt-6-astra/policy.py), [GR00T](policies/GR00T_N17/policy.py), or [OpenWAM](policies/OpenWAM/policy.py) implementation.

```sh
uv run moravec-policy serve policies/gpt-6-astra/policy.json --host 127.0.0.1 --port 8000
uv run moravec-policy robot robots/dual-yam.json
```

The default [server](src/moravec_policy/client_server/service.py) handles authentication, episode/request IDs and boundary validation. You may replace it with your own server/container implementing the same wire interface; transport conformance is required in either case. The evaluator retains execution and scoring. The default server supports one episode at a time and advertises `batching: false`.

Robot definitions provide arm/gripper dimensions and known joint ordering. Read one with `load_robot(path)` from `moravec_policy.robots`; use it in a policy's embodiment-specific conversion. YAM has six arm joints plus one commanded gripper dimension; `dual-yam` has two of each. These files do not supply calibration, controllers or trained checkpoint support. Runtime camera/frame information still arrives in the existing robot profile; there is no separate stations directory. Reference dimensions and attribution are documented in [third-party notices](THIRD_PARTY_NOTICES.md).

Submission manifests use v0.5 (`entrypoint` + policy-owned `config`, replacing `backend`). The observation/action wire protocol remains v0.4.

## Interface

The binary socket is `/policy`. Authenticate the handshake with `Authorization: Bearer <MORAVEC_POLICY_TOKEN>`. Use WSS outside loopback. Each request is a MessagePack map:

```text
{protocol: "moravec.policy/v0.4", request_id: "unique-id", method: "...", payload: {...}}
```

Methods: `hello`, `reset` (episode ID and robot profile), `update_obs` ([observation envelope](schemas/ObservationEnvelope.json)), `get_action` (empty payload), and `end` (episode ID). Replies echo protocol/request ID and contain `result` or `error`. One connection owns one container session. Errors require reset; disconnects abort the session. No automatic action retry or cross-connection resume is implemented.

Observation envelopes separate `metadata` from `observation: {instruction, vision, state}`. The [Python types](src/moravec_policy/contracts.py) are authoritative:

- **Vision:** named RGB uint8 HWC byte buffers; optional registered depth as little-endian float32 meters (zero = invalid), 3×3 intrinsics, and 4×4 camera-to-reference-frame extrinsics. Capture timestamps are explicit. Required channels are declared in the policy manifest; missing channels fail validation.
- **State:** named joint positions/velocities, TCP and end-effector poses, gripper joints/opening fractions, and optional mobile-base pose/twist. Position units are meters, revolute joints radians, rotations WXYZ quaternions. Base twist is `[vx, vy, vz, wx, wy, wz]` in the base pose's reference frame.
- **Actions:** `action-chunk` or `stop`. A chunk contains `timestep_s`, `execution_horizon`, and `steps`. Each step holds named `joint_positions`, `tcp_targets`, and `grippers`; all targets in step i are due at `(i + 1) * timestep_s` from the same station start time. Target names stay constant throughout the chunk. Single-arm and bimanual policies use the same structure. Joint and Cartesian targets cannot be mixed in one chunk without a qualified resource mapping.
- **Omission:** `omitted_targets: "hold-start"` means uncommanded joints, arms, and grippers hold their state captured at chunk start; omitted fields never mean zero or release. Stations must implement this hold behavior or reject the chunk. Only the first `execution_horizon` steps execute; later steps are predictions. Validate the complete chunk before executing any step.
- **Pose targets:** each named target specifies its pose, absolute/relative representation, frame/tool reference and observation/previous-point origin. Frame-relative deltas use `p'=p+dp, q'=dq*q`; tool-relative deltas use `p'=p+R(q)dp, q'=q*dq`. Frame and semantics remain constant per arm within a chunk. Gripper fractions are 0 closed, 1 open.

See [a two-arm action chunk](examples/bimanual-action.json). v0.4 replaces the v0.3 trajectory/command variants and `coordinated` wrapper; old envelopes are rejected. Adapters and stations must upgrade together. Shared timestamps describe execution intent, not collision-free planning or hardware synchronization guarantees.

MessagePack carries binary image/depth buffers directly; JSON schemas describe those byte fields using their JSON/base64 representation. This protocol is Moravec-versioned; matching lifecycle and robotics conventions does not imply wire compatibility with another server.

The station validates expiry using its current clock, plans and executes only the declared horizon, and sends measured feedback. The simulator pauses its clock during inference; physical latency/watchdog behavior is not validated by this test. Collision avoidance, velocity/force limits and emergency stopping remain station responsibilities.

## Model examples and submission

- [gpt-6-astra](policies/gpt-6-astra): Responses API tools become absolute pose chunks; reasoning/tool history resets between episodes.
- [GR00T_N17 (LIBERO)](policies/GR00T_N17/libero): task-trained NVIDIA example with pinned weights, an offline GPU container, and a matching LIBERO adapter.
- [OpenWAM (LIBERO)](policies/OpenWAM/libero): task-trained world-action model using the same simulation client; GPU container qualification remains separate.
- [GR00T_N17](policies/GR00T_N17): native SDK template with temporal observations and explicit pose/joint bindings. Requires the correct CUDA runtime and checkpoint.
- [OpenWAM](policies/OpenWAM): native world-action template; requires checkpoint adaptation. Video-only world models need a separate action-selection planner.

Bind camera names, state ordering, normalization, observation cadence, rotation representation, action timestep and execution horizon from actual checkpoint metadata. Rotation adapters cover WXYZ, rotation vectors, XYZ Euler radians, and explicitly declared row/column 6D rotations. The SDK must supply denormalized outputs; gripper fractions must already match the declared convention. Placeholder bindings remain `requires-qualification` and refuse inference.

`docker compose up --build` serves the fixture after `MORAVEC_POLICY_TOKEN` is set. Build the shared CPU base with `docker build -t moravec-policy-base:local .`, then select a policy recipe with `docker build -f policies/gpt-6-astra/Dockerfile -t moravec-astra .`. For a GPU policy, build the shared base using `--build-arg BASE_IMAGE=<pinned-image-with-that-policy-SDK>`, then use that base in its policy Dockerfile. The checked-in CPU base has no GPU SDKs or weights. The GR00T LIBERO folder provides its own pinned GPU recipe and licensed-weight preparation command; other GPU recipes remain templates unless the validation report explicitly qualifies them. Isolated deployments have no internet; API-connected models use a separate approved network profile.

Submit the image digest, policy manifest, checkpoint/binding hashes, compute requirements, and executed validation report. Keep credentials, weights and private evaluation results out of Git. A customer or coding agent should implement the adapter against this fixed interface, preserving the evaluator, task assets, seeds and scoring rules. Real checkpoint simulation qualification comes before physical station qualification.
