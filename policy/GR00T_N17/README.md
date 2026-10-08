# GR00T_N17

Moravec policy adapter for [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.7. `model.py` converts observations and actions; `deploy.yml` selects checkpoint and serving settings. GR00T itself is installed from NVIDIA's repository at a pinned commit.

## Setup

On a Linux NVIDIA GPU host, create a Python 3.10 environment and install the pinned runtime, GR00T, and this starter:

```sh
uv venv --python 3.10 .venv-gr00t && source .venv-gr00t/bin/activate
uv pip install -r policy/GR00T_N17/requirements.lock
uv pip install --no-deps "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T@23ace64f17aa5015259b8609d371eb61a357c776"
uv pip install --no-deps -e .
```

Checkpoint access and CUDA compatibility are still required.

## Serving

Activate the environment, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads.

## Checkpoint configuration

`deploy.yml` selects NVIDIA's LIBERO spatial checkpoint in native-I/O mode. This is a simulation fixture, not a YAM policy. Download the inference files into this policy's `checkpoints/` directory:

```sh
hf download nvidia/GR00T-N1.7-LIBERO \
  --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
  --include 'libero_spatial/*.json' 'libero_spatial/*.safetensors' \
  --local-dir policy/GR00T_N17/checkpoints
```

Cosmos-Reason2-2B access and runtime dependencies are required. Its identifier in `deploy.yml` must also match the dependency in the checkpoint's `config.json`; private offline copies can contain machine-specific paths and need preparation before loading. GR00T only accepts a backbone whose name contains `nvidia/Cosmos-Reason2`, so keep that in the path of a local copy (for example `/weights/nvidia/Cosmos-Reason2-2B`) and point both `cosmos_model_path` and the checkpoint's `config.json` `model_name` at it.

Native observations follow the checkpoint's `video`, `state`, and `language` schema. The returned chunk preserves native action names and units; the evaluator performs controller conversion.

## Container quickstart

This example targets a Linux x86_64 NVIDIA GPU host with Docker and the NVIDIA
Container Toolkit. It uses the LIBERO checkpoint above; it does not establish
YAM compatibility. Internet access is allowed for model downloads in this demo.

Build from the repository root:

```sh
docker build --platform linux/amd64 -f policy/GR00T_N17/Dockerfile \
  -t moravec-gr00t:demo .
```

The image installs NVIDIA's GR00T at a pinned commit and the pinned inference runtime.
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

From a second terminal with the starter installed (`pip install -e .`), validate it:

```sh
moravec-validate policy/GR00T_N17 --url ws://127.0.0.1:6000
```

This checks the manifest, then runs the policy's native check (`check_client.py`,
named in `moravec-policy.yaml`), which sends synthetic camera/state input in the
checkpoint's own format to the loaded model and checks its returned action chunk. It never executes actions. For simulation evaluation,
the separate evaluator must supply real scene observations and controllers.
Stop the server with `docker stop moravec-gr00t`; the named volumes retain weights.
