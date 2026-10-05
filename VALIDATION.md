# Validation — v0.3

Verified locally on 2026-10-04:

- **54 tests passed.** These cover schema and lifecycle validation, WebSocket authentication and binary messages, multiple observation updates before inference, duplicate requests, missing feedback, reset behavior, observation freshness, RGB/depth/calibration round trips, pose composition, frame/tool-relative deltas, rotation conversion, execution horizons, coordinated controls, synthetic native SDK tensor shapes/history, provider tool errors and unqualified-checkpoint rejection.
- **15 MuJoCo 3.14.0 episodes passed**, five each for joint trajectories, relative poses and the LLM harness producing absolute pose chunks. All station traffic used real local WebSocket connections and MessagePack. The LLM provider was an explicit deterministic HTTP fixture. Thirty decisions total; no paid model calls.
- Each episode met the 1 cm position threshold without physics warnings. These are numeric-target integration checks, not perception tests or model leaderboard results.
- Ruff and diff checks passed; generated schemas reflect v0.3. The old HTTP policy routes and OpenAPI description were removed.

[Committed report](validation/conformance-report.json) and [artifact checksums](validation/artifacts.json). Local video/events: `runs/websocket-conformance/`.

Not verified:

- Broader Astra performance beyond the single numeric-target smoke test below remains unverified.
- Real GR00T/OpenWAM weights, SDK execution on GPU, or checkpoint/embodiment compatibility. Temporal input and rotation/action conversion tests use a synthetic SDK fixture; bindings remain unqualified.
- Container execution: the Docker daemon was unavailable in the previous check; this turn ran the Python service directly.
- Isaac Sim, physical robots, wall-clock hardware latency, collision/force limits, or safety-controller behavior. MuJoCo pauses while awaiting inference. Gripper/coordinated/multimodal paths have contract tests, not physical task qualification in this planar scene.
- Batched inference and reconnect/resume. Batch methods explicitly reject calls; disconnects abort episodes; there are no automatic action retries.

One upstream Starlette TestClient deprecation warning remains; all tests passed. v0.3 deliberately changes the station wire protocol, image encoding and adapter lifecycle. It follows familiar policy lifecycle and robotics conventions but is not wire-compatible with another framework.

## Real Astra smoke test — 2026-10-05

Passed one numeric-target MuJoCo episode using the actual OpenAI Responses API, with two completed responses identifying `gpt-6-astra`. The first produced a pose trajectory; the station executed it and supplied measured feedback. The second produced stop. Independent simulation scoring confirmed the target was within the 1 cm threshold. No fixture or model fallback was used.

Total usage: 2,498 input tokens and 140 output tokens (2,638 total). The key was supplied only to the test process; it was not saved to the repository or reports. [Provider metadata and artifact hashes](validation/astra-live-20261005.json); local video/events are in `runs/astra-live-20261005/`.

This verifies account/model access and a two-turn model-to-simulation loop. It does not establish visual task competence, Isaac Sim compatibility, physical robot readiness, or benchmark success rate. The simulator was paused during API inference and the numeric target was explicitly supplied.
