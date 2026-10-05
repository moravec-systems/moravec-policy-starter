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

Configure the checkpoint, training configuration, normalization assets, and environment in `deploy.yml`. The defaults are not a YAM-trained checkpoint.

## Serving

Activate the model runtime, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads. Training and benchmark orchestration are outside this starter's scope.

The [source manifest](../../scripts/fetch_policy.py) pins the exact implementation snapshot used by this adapter.

## Concrete research fixture

The deployment example uses [this community π0.5 checkpoint](https://huggingface.co/cjgogo/RoboDojo-pi05-checkpoints), revision `c9d496dd7901f4d7bf9055fb88b41cb66aea3e36`, step `14077`. Its model card restricts the benchmark-derived checkpoint to noncommercial research/evaluation; this is a test fixture, not a recommendation for customer deployment.

From the repository root, download inference parameters and normalization assets only:

```sh
hf download cjgogo/RoboDojo-pi05-checkpoints \
  --revision c9d496dd7901f4d7bf9055fb88b41cb66aea3e36 \
  --include 'checkpoints/sim-10task/14077/params/**' \
            'checkpoints/sim-10task/14077/assets/**' \
            'checkpoints/sim-10task/14077/_CHECKPOINT_METADATA' \
  --local-dir policy/Pi_05
```

Activate the model environment and set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing a GPU with simulation. `env_cfg_type: arx_x5` requires the corresponding evaluator configuration. The training config name and normalization asset ID are checkpoint-specific identifiers; do not rename them as branding.
