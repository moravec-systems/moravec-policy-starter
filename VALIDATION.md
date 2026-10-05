# Validation boundaries

Verified on 2026-10-05 (before the naming-only alignment to upstream policy IDs; rebuild images for the renamed paths):

| Path | Evidence | Boundary |
| --- | --- | --- |
| Shared contracts and adapters | 83 tests pass | Includes synthetic SDK outputs; unit tests alone do not establish model capability |
| Shared CPU container | Build and authenticated socket lifecycle pass across two resets | Fixture policy, not learned inference |
| `gpt-6-astra` | 1/1 live OpenAI API MuJoCo reach episode passes | Bounded two-call planar integration check |
| `GR00T_N17 (LIBERO)` | 4/4 LIBERO episodes pass: tasks 0 and 1, two initial states each | Actual released checkpoint, repository loader, shared client/server, built GPU container on L40S |
| `OpenWAM (LIBERO)` | 4/4 matching LIBERO episodes pass | Actual released checkpoint and repository adapter through the shared server; process runtime, not a qualified container |

These are preliminary compatibility checks, not leaderboard scores. Setup failures and development reruns are excluded from the final qualification set. The base GR00T and foundation OpenWAM folders remain unqualified templates; success by task-trained checkpoints does not qualify them or every robot definition.

GR00T's final container ran with a read-only filesystem, UID 10001, immutable prepared-weight checksums, no Hugging Face credential, offline SDK settings, and verified blocked outbound networking. A validation-only loopback ingress proxy connected the simulator to its isolated Docker bridge. The example's simple local Docker command publishes loopback but does not itself enforce network isolation. The GPU recipe pins SDK source and package versions; preserve the resulting image digest for deployment.

GR00T predicts 16 actions and buffers the first 8 internally. Both LIBERO examples emit one physical delta per fresh observation. This preserves LIBERO's measured-state OSC semantics; a multi-step `relative_to: previous` trajectory would instead refer to previous targets. The runner verifies 20 Hz control and 0.05 m / 0.5 rad controller scales. GR00T replans after eight returned steps; OpenWAM manages its prediction buffer inside the SDK. Request counts are not GPU forward-pass counts.

Run `uv run pytest -q` for interface tests and `uv run python sim/validate.py` for deterministic MuJoCo conformance. `--live-astra` explicitly enables at most two paid OpenAI calls. See [simulation instructions](sim/README.md) and the task-trained policy folders for checkpoint tests. The tested LIBERO checkout includes the documented PyTorch initial-state loading compatibility patch.

Run-specific reports, provider IDs, usage records, videos, images and traces remain in ignored `runs/` or private evaluation storage. They must not be committed. Synthetic test inputs remain in the test suite.

Remaining qualifications: OpenWAM's distributable GPU image; Isaac Sim execution through these exact new adapters; customer-specific checkpoints/embodiments; and physical station control, latency, watchdogs, calibration and safety behavior. Earlier reference-harness runs do not establish those properties. Simulation pauses during inference. Batch methods reject calls; disconnects abort episodes without automatic retries.

This supports sharing a simulation integration starter and a working GR00T serving example for engineering review. It does not claim production station readiness, YAM model support, or validated customer-task performance.
