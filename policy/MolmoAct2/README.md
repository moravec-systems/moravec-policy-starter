# MolmoAct2

Minimal Moravec adapter for [Ai2's MolmoAct2 BimanualYAM checkpoint](https://huggingface.co/allenai/MolmoAct2-BimanualYAM). It calls the checkpoint's public Transformers `predict_action` API through the shared policy server. The model implementation and weights stay outside Git.

## Setup

From the repository root, create a separate model environment:

```sh
uv venv --python 3.12 policy/MolmoAct2/.venv
uv pip install --python policy/MolmoAct2/.venv/bin/python \
  'torch==2.9.0' 'torchvision==0.24.0' 'transformers==4.57.1' \
  'huggingface-hub>=0.36,<1' 'einops==0.8.1' 'pillow>=10' -e .
source policy/MolmoAct2/.venv/bin/activate
hf download allenai/MolmoAct2-BimanualYAM \
  --revision 8dcbed66f2380e4393189c303ea72488eb9e63c2 \
  --local-dir policy/MolmoAct2/checkpoints/BimanualYAM
```

This snapshot includes executable model/processor source loaded with `trust_remote_code=True`, plus normalization metadata and weights. Review that pinned source before use. Downloads retain provider notices and Hugging Face download metadata. Serving loads only the local snapshot; a relative `model_path` resolves beside this policy's `model.py`.

## Observation and action contract

This example requires `env_cfg_type: yam` and `action_type: joint`. Supply the evaluator's neighboring `env_cfg/` as described in the [integration guide](../../INTEGRATION.md). Its YAM registry must specify `arm_dim: [6, 6]` and `ee_dim: [1, 1]`. Matching dimensions alone do not establish hardware compatibility.

Each observation contains a nonempty `instruction`, three decoded HWC uint8 RGB images under `vision`, and the shared bimanual `state` dictionary. Camera order is fixed:

| Checkpoint camera | Accepted observation names |
| --- | --- |
| Top | `top`, `cam_high`, `cam_head`, `top_camera` |
| Left wrist | `left`, `cam_left_wrist`, `left_camera` |
| Right wrist | `right`, `cam_right_wrist`, `right_camera` |

Camera entries may hold the array directly or under `color`/`rgb`. State packs left six arm joints, left gripper, right six arm joints, right gripper. Arm angles are radians; gripper coordinates follow the checkpoint's normalized YAM convention. Returned actions are de-normalized absolute joint targets in that same order, unpacked into shared action dictionaries. No delta conversion, calibration, or gripper rescaling is performed. The evaluator must match the provider's joint order, gripper convention, camera placement and calibration before robot execution.

The adapter checks the checkpoint's `yam_dual_molmoact2` normalization tag, 14-D state/action metadata, 30-step horizon, and absolute-joint control mode. ARX X5 and Franka checkpoints are outside this example's contract.

## Serving

With the model environment activated, from the repository root:

```sh
python setup_policy_server.py --config_path policy/MolmoAct2/deploy.yml
```

Defaults use CUDA with bfloat16 and 10 flow-solver steps. CPU/MPS require `device=cpu` or `device=mps` with `dtype=float32`; float32 substantially increases memory use. CUDA graphs are disabled. Set `actions_per_chunk` between 1 and 30 to limit returned steps. Batch calls run one public inference call per environment; reset clears stored observations. The model samples actions, so repeated calls need not be identical.

Use the existing [WebSocket client](../../INTEGRATION.md#remote-serving). The evaluator owns action execution and scoring. This adapter does not provide a YAM simulator or establish physical task success.

## Container

Build from the repository root on a Linux NVIDIA GPU host. The image pins Python
and all Python dependencies; weights and credentials are mounted at runtime.

```sh
docker build --platform linux/amd64 -f policy/MolmoAct2/Dockerfile \
  -t moravec-molmoact2:demo .
docker volume create moravec-molmoact2-weights
docker volume create moravec-molmoact2-cache
docker run --rm -e HF_TOKEN \
  -v moravec-molmoact2-weights:/weights \
  -v moravec-molmoact2-cache:/cache/huggingface \
  moravec-molmoact2:demo \
  hf download allenai/MolmoAct2-BimanualYAM \
    --revision 8dcbed66f2380e4393189c303ea72488eb9e63c2 \
    --local-dir /weights/BimanualYAM
```

`HF_TOKEN` is optional for this public snapshot; if needed, set it in your host
environment. Never put credentials in build arguments or the image. The download
includes the pinned executable checkpoint source, which is loaded with
`trust_remote_code=True` as described above.

Set `MORAVEC_ENV_CFG` to the absolute path of your evaluator's `env_cfg/`, containing
`yam.yml` (`config.robot: yam`) and `robot/_robot_info.json` with the YAM dimensions
described above. The runtime resolves this directory at `/env_cfg` beside `/app`.

```sh
docker run --rm --name moravec-molmoact2 --gpus all --shm-size=2g \
  -p 127.0.0.1:6000:6000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v moravec-molmoact2-weights:/app/policy/MolmoAct2/checkpoints:ro \
  -v moravec-molmoact2-cache:/cache/huggingface \
  --mount "type=bind,src=${MORAVEC_ENV_CFG:?Set MORAVEC_ENV_CFG},dst=/env_cfg,readonly" \
  moravec-molmoact2:demo
```

Inspect initialization with `docker logs moravec-molmoact2`. A healthy container
means its WebSocket is listening after model loading. From another terminal,
request an action using the image's client:

```sh
docker exec moravec-molmoact2 python policy/MolmoAct2/check_client.py
```

The check sends three synthetic RGB images and a 14-D joint state through the
shared protocol to the loaded checkpoint, then validates its 30-step action
chunk. It does not execute actions or establish task performance. Use
`--expected-steps N` if serving with a different `actions_per_chunk`. Real
simulation requires a separately calibrated YAM evaluator; this image supplies
the policy server only. Stop it with `docker stop moravec-molmoact2`.


## Sources

Inference uses [Ai2's YAM example](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/examples/yam/host_server_yam.py) and the pinned checkpoint's model card/source. Adapter attribution is in [third-party notices](../../THIRD_PARTY.md). Training is outside this example's scope.
