# OpenWAM

Moravec policy adapter. `model.py` converts observations and actions; `deploy.yml` selects checkpoint and serving settings. The model implementation is downloaded during setup and is not tracked in this repository.

## Setup

Fetch with Python 3.12+, then install in a dedicated GPU runtime with a suitable PyTorch/CUDA combination:

```sh
python scripts/fetch_policy.py OpenWAM
python -m pip install -e . -e policy/OpenWAM/OpenWAM
```

Use a compatible checkpoint with its configuration and normalization files. `model.py` currently contains ARX X5-specific coordinate transforms; it is not a generic YAM policy.

## Serving

Activate the model runtime, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads. Training and benchmark orchestration are outside this starter's scope.

The [source manifest](../../scripts/fetch_policy.py) pins the exact implementation snapshot used by this adapter.

## Research fixture

Download the [released simulation checkpoint](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-RoboDojo) into the path selected by `deploy.yml`:

```sh
hf download OpenWAM/OpenWAM-Alpha-Sim-RoboDojo \
  --revision 2c1302294e3ba8319bbdb2c803b7a27de9292d03 \
  --local-dir policy/OpenWAM/checkpoints/openwam-sim
```

This fixture requires the evaluator's ARX X5 environment configuration. It does not qualify YAM or FR3 hardware. The example uses single-environment evaluation.
