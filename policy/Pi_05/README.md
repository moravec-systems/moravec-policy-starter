# Pi_05

Moravec policy adapter for Physical Intelligence's π0.5, built on [openpi](https://github.com/Physical-Intelligence/openpi), PI's reference implementation. `model.py` converts observations and actions; `deploy.yml` selects checkpoint and serving settings.

The adapter follows openpi's ALOHA recipe (`LeRobotAlohaDataConfig`): cameras `cam_high`, `cam_left_wrist` and `cam_right_wrist`, a 14-D state of six joints plus one gripper per arm, and delta joint actions. Instead of registering a named training config, `model.py` builds the inference config inline from `deploy.yml`:

- `asset_id`: normalization stats under `<model_path>/assets/<asset_id>/`.
- `adapt_to_pi`: `true` only for checkpoints trained in π's internal joint/gripper space (openpi's default for standard ALOHA data); `false` for checkpoints trained on raw joint values, like the fixture below.

## Setup

Clone openpi at the pinned commit and use its own environment:

```sh
git clone https://github.com/Physical-Intelligence/openpi policy/Pi_05/openpi
git -C policy/Pi_05/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
cd policy/Pi_05/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install --python .venv/bin/python -e ../../..
```

## Serving

Activate `policy/Pi_05/openpi/.venv`, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads.

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

Activate the model environment and set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing a GPU with simulation. `env_cfg_type: arx_x5` requires the corresponding evaluator configuration. The normalization asset ID is a checkpoint-specific identifier.

## Container

Build on a Linux x86_64 NVIDIA GPU host with Docker and NVIDIA Container Toolkit. The image clones openpi at the pinned commit and installs it from openpi's own lockfile; CUDA/JAX and PyTorch make the image substantially larger than the CPU demo. Weights are not included.

```sh
docker build --platform linux/amd64 -f policy/Pi_05/Dockerfile -t moravec-pi05:demo .
docker run -d --name pi05 --gpus all --shm-size=2g \
  -p 127.0.0.1:6000:6000 \
  --mount type=bind,src="$PWD/policy/Pi_05/checkpoints",dst=/app/policy/Pi_05/checkpoints,readonly \
  --mount type=volume,src=moravec-pi05-cache,dst=/cache \
  moravec-pi05:demo
docker logs -f pi05
```

After the server is listening, run the inference check in another terminal:

```sh
docker exec pi05 python policy/Pi_05/check_client.py
docker stop pi05
```

The check sends three synthetic RGB images, a 14-element joint/gripper state, and an instruction. It expects 50 steps containing both six-joint arms and their grippers. This checks the real checkpoint's inference and wire format; task performance requires simulation with real observations. The first inference includes JAX compilation and can take several minutes.

The included robot dimensions and check client target the ARX X5 research fixture above. They do not qualify YAM or arbitrary checkpoints. For another checkpoint, mount a deployment YAML and its weights, then override the default command with `python setup_policy_server.py --config_path /config/deploy.yml --host 0.0.0.0`. Its `asset_id`, `adapt_to_pi`, normalization assets, cameras, and action convention must agree.

On first startup OpenPI downloads its tokenizer into `/cache/openpi`; this example requires network access unless that cache is already populated. Download restricted weights separately using your authenticated host environment. Do not bake tokens or checkpoints into the image.
