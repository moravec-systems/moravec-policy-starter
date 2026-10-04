# Moravec policy starter

A local sample repository for onboarding robot policy systems: **Astra (LLM + reference harness), NVIDIA GR00T N1.7 (VLA), and OpenWAM (world-action model)**. Built around pinned XPolicyLab adapters and RoboProbe's actual LLM harness, not a new robotics controller.

**Status:** runnable CPU contract demo, authenticated reference service/client, strict schemas, and source-pinned real-model integration examples. Real Astra/GR00T/OpenWAM inference has **not** been run from this repo. No GPU provisioned, weights downloaded, or API credits spent to create it. This is a proposed Moravec interface, not an official XPolicyLab wire standard.

| Example | Integration | Output | Still needed for a real run |
| --- | --- | --- | --- |
| [Astra](examples/astra/README.md) | Native RoboProbe EEF harness | Grasp-point tool calls → planned native joint actions | Astra endpoint/key, native compatible RoboDojo runtime |
| [GR00T N1.7](examples/groot/README.md) | Existing XPolicyLab GR00T_N17 adapter | Joint action chunks | Compatible fine-tuned checkpoint, modality config, CUDA environment |
| [OpenWAM](examples/openwam/README.md) | Existing XPolicyLab OpenWAM adapter | Absolute EE action chunks | Released RoboDojo checkpoint, normalization, CUDA environment |
| [Remote](examples/remote/README.md) | Authenticated HTTP policy client | Same action contract | A running compatible service |
| Mock | Hold measured state | Explicitly labeled mock chunk | Nothing; contract testing only |

## Try it locally (no GPU, model key, or simulator)

```bash
uv sync --locked
uv run pytest
uv run moravec-policy validate examples/astra/policy.json
uv run moravec-policy validate examples/groot/policy.json
uv run moravec-policy validate examples/openwam/policy.json
uv run moravec-policy schemas schemas

# Terminal 1. Pick a local test token and use the same token in terminal 2.
export MORAVEC_POLICY_TOKEN='local-contract-test-token-only'
uv run moravec-policy serve examples/mock/policy.json

# Terminal 2
export MORAVEC_POLICY_TOKEN='local-contract-test-token-only'
uv run python examples/remote/client.py
```

The result must say `evidence_kind: mock`. It is not task execution or a benchmark score. The synthetic RGB fixture is a four-color 2×2 image; zero joints/poses are schema examples, not safe robot commands.

## How the pieces fit

```text
Customer model + preprocessing + memory + optional harness
                         ↓
              versioned policy manifest
                         ↓
    local container or remote policy deployment
                         ↓
     episode observations → inference → action chunks
                         ↓
    benchmark-owned execution, scoring, video and logs
```

The SDK wraps the existing GR00T/OpenWAM adapters. Those upstream adapters own frame conversions and normalization. Astra retains the environment-side RoboProbe harness; its `model.py` is a compatibility stub, not an inference server. `astra-plan` constructs the official fixed-layout command and only executes with `--execute`. We do not pretend all three models use the same upstream process topology.

- [Contract and schema decisions](docs/contract.md)
- [Validation and limits](docs/validation.md)
- [Deployment/container guide](docs/deployment.md)
- [Sources and provenance](docs/sources.md)
- [Adding VLMs, pure world models, and other policies](docs/model-types.md)

Start with one episode per worker. Scale independent workers; native model batching is intentionally not exposed by this starter. Keep the policy manifest, deployment digest, evaluation spec, runtime version and native result together. Container/API readiness and physical readiness are separate qualifications.

## CPU example container

```bash
docker build -t moravec-policy-starter:local .
docker run --rm -p 127.0.0.1:8000:8000 \
  -e MORAVEC_POLICY_TOKEN moravec-policy-starter:local
```

This image contains only the mock adapter and SDK. For GPU examples install the SDK into the corresponding upstream environment; don't install incompatible model stacks into one universal image. The image build itself has not been verified unless recorded in `VALIDATION.md`.

No credentials, customer documents, checkpoint binaries, or benchmark results are included. Source licenses and model asset terms remain separate; this sample does not redistribute third-party weights.
