# Pi_05

Moravec policy adapter. `model.py` converts observations and actions; `deploy.yml` selects checkpoint and serving settings. The model implementation is downloaded during setup and is not tracked in this repository.

## Setup

Fetch with Python 3.12+, then prepare OpenPI's own runtime:

```sh
python scripts/fetch_policy.py Pi_05
cd policy/Pi_05/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
uv pip install --python .venv/bin/python -e . -e ../../..
```

Configure the checkpoint, training configuration, normalization assets, and environment in `deploy.yml`. The defaults are not a YAM-trained checkpoint. Source fetching is verified separately from GPU installation.

## Serving

Activate the model runtime, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads. Training and benchmark orchestration are outside this starter's scope.

The [source manifest](../../scripts/fetch_policy.py) pins the exact implementation snapshot used by this adapter. The current shared demo validates transport only; real-checkpoint simulation qualification remains outstanding.
