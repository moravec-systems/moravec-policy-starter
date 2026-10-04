# Validation — 2026-10-04

## Verified locally

- 31 tests passed: contract parsing, observation freshness/camera synchronization, finite values, frame and joint checks, action expiry, HTTP authentication, episode lifecycle, feedback matching, LLM tool parsing/provider errors, reset behavior, and refusal of unqualified native checkpoints.
- Ruff checks and formatting passed.
- Ten physical MuJoCo 3.14.0 episodes passed through actual local HTTP servers: five joint-trajectory episodes and five LLM-harness episodes using an explicit deterministic HTTP model. Twenty decisions total.
- Final TCP position errors ranged from 0.014 to 0.028 mm against a 10 mm conformance threshold. These are controller/interface checks with supplied numeric targets, not model performance scores.
- Physics warnings invalidate episodes. The initial Euler-integrator run was unstable; switching this stiff position-controller fixture to implicitfast resolved it. The successful rerun completed without physics warnings.
- Generated schema and OpenAPI files describe the implemented contract.

The committed [conformance report](validation/conformance-report.json) records the scene hash and explicit model-execution=false status. Local video and event traces are in `runs/conformance/`; checksums are in [artifact hashes](validation/artifacts.json).

## Not yet verified

No real Astra, GR00T or OpenWAM inference ran. No model weights, GPU, paid provider call, Isaac Sim run, or physical robot was used. Native SDK drivers remain unqualified templates requiring adapted checkpoints and station bindings. Video-only world-model planning is not implemented.

Docker execution could not be tested because the local Docker daemon is not running. The Dockerfile and isolated Compose configuration are supplied, but image build, network isolation and GPU runtime behavior are unverified. Security, retention, storage delivery and customer-owned hardware requirements are documented boundaries, not completed infrastructure.

Pytest emitted one upstream Starlette warning about its httpx TestClient compatibility path. All tests passed; resolve that dependency transition before pinning a production image.
