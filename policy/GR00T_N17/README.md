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

This preserves the previous platform-specific editable installation approach. Checkpoint access, CUDA compatibility, and a matching modality configuration under `configs/` are still required.

## Serving

Activate the model runtime, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads. Training and benchmark orchestration are outside this starter's scope.

The [source manifest](../../scripts/fetch_policy.py) pins the exact implementation snapshot used by this adapter.

## Checkpoint configuration

`deploy.yml` selects NVIDIA's LIBERO spatial checkpoint in native-I/O mode. This is a simulation fixture, not a YAM policy. Download the inference files into this policy's `checkpoints/` directory:

```sh
hf download nvidia/GR00T-N1.7-LIBERO \
  --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
  --include 'libero_spatial/*.json' 'libero_spatial/*.safetensors' \
  --local-dir policy/GR00T_N17/checkpoints
```

Cosmos-Reason2-2B access and runtime dependencies are required. Its identifier in `deploy.yml` must also match the dependency in the checkpoint's `config.json`; private offline copies can contain machine-specific paths and need preparation before loading.

Native observations follow the checkpoint's `video`, `state`, and `language` schema. The returned chunk preserves native action names and units; the evaluator performs controller conversion. `io_format: arx_x5` retains the existing bimanual adapter and requires a matching ARX X5 checkpoint and environment configuration. It must not be used with the LIBERO checkpoint.

## Container quickstart

This example targets a Linux x86_64 NVIDIA GPU host with Docker and the NVIDIA
Container Toolkit. It uses the LIBERO checkpoint above; it does not establish
YAM compatibility. Internet access is allowed for model downloads in this demo.
Other examples use their documented Python environments; containers will follow.

Build from the repository root:

```sh
docker build --platform linux/amd64 -f policy/GR00T_N17/Dockerfile \
  -t moravec-gr00t:demo .
```

The image fetches pinned model source and installs the pinned inference runtime.
Weights and credentials are excluded from its build context. Obtain access to
Cosmos-Reason2-2B on Hugging Face, then set `HF_TOKEN` in your local environment
without putting its value in the Dockerfile or Git.

Download the checkpoint into a Docker volume:

```sh
docker volume create moravec-gr00t-weights
docker volume create moravec-gr00t-cache
docker run --rm -e HF_TOKEN \
  -v moravec-gr00t-weights:/app/policy/GR00T_N17/checkpoints \
  -v moravec-gr00t-cache:/cache/huggingface \
  moravec-gr00t:demo \
  hf download nvidia/GR00T-N1.7-LIBERO \
    --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
    --include 'libero_spatial/*.json' 'libero_spatial/*.safetensors' \
    --local-dir /app/policy/GR00T_N17/checkpoints
```

Start the policy server:

```sh
docker run --rm --name moravec-gr00t --gpus all --shm-size=2g \
  -p 127.0.0.1:6000:6000 -e HF_TOKEN \
  -v moravec-gr00t-weights:/app/policy/GR00T_N17/checkpoints \
  -v moravec-gr00t-cache:/cache/huggingface \
  moravec-gr00t:demo
```

The weights volume is writable because the current adapter temporarily adjusts
processor configuration while loading. Use a dedicated volume for this example.
Cosmos files download on first use and are cached. A listening socket/healthy
container means model initialization completed; it does not prove task success.
Use `docker logs moravec-gr00t` to inspect startup and
`docker inspect --format '{{.State.Health.Status}}' moravec-gr00t` for readiness. Host port exposure is loopback-only; remote
connections use the SSH tunnel in the integration guide.

From a second terminal with the starter installed, request an action:

```sh
python policy/GR00T_N17/check_client.py
```

This sends synthetic camera/state input to the actual loaded model and checks
its returned action chunk. It never executes actions. For simulation evaluation,
the separate evaluator must supply real scene observations and controllers.
Stop the server with `docker stop moravec-gr00t`; the named volumes retain weights.

The [web demo](https://moravec-evaluation-demo.vercel.app/demo) previews policy
submission and configuration validation. It does not launch containers or real
simulations. Its uploaded sample configuration is a UI fixture, not a production
submission contract. Registry access, scheduling, offline deployment, and result
storage are separate upcoming platform work.
