# GR00T_N17

Moravec policy adapter. `model.py` converts observations and actions; `deploy.yml` selects checkpoint and serving settings. The model implementation is downloaded during setup and is not tracked in this repository.

## Setup

On a compatible Linux NVIDIA GPU host, fetch with Python 3.12+, then create the model's Python 3.10 environment:

```sh
python scripts/fetch_policy.py GR00T_N17
uv venv --python 3.10 policy/GR00T_N17/gr00t_n17/.venv
cd policy/GR00T_N17/gr00t_n17
uv pip install --python .venv/bin/python -e . -e ../../..
```

This preserves the previous platform-specific editable installation approach. Checkpoint access, CUDA compatibility, and a matching modality configuration under `configs/` are still required. Source fetching is verified separately from GPU installation.

## Serving

Activate the model runtime, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads. Training and benchmark orchestration are outside this starter's scope.

The [source manifest](../../scripts/fetch_policy.py) pins the exact implementation snapshot used by this adapter. The current shared demo validates transport only; real-checkpoint simulation qualification remains outstanding.
