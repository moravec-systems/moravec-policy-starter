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

The [source manifest](../../scripts/fetch_policy.py) pins the exact implementation snapshot used by this adapter. The shared demo validates transport only; the separate checkpoint validation below exercises learned inference and simulation.

## Verified checkpoint configuration

`deploy.yml` selects NVIDIA's LIBERO spatial checkpoint in native-I/O mode. This is a simulation fixture, not a YAM policy. Download the inference files into this policy's `checkpoints/` directory:

```sh
hf download nvidia/GR00T-N1.7-LIBERO \
  --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
  --include 'libero_spatial/*.json' 'libero_spatial/*.safetensors' \
  --local-dir policy/GR00T_N17/checkpoints
```

Cosmos-Reason2-2B access and runtime dependencies are required. Its identifier in `deploy.yml` must also match the dependency in the checkpoint's `config.json`; private offline copies can contain machine-specific paths and need preparation before loading.

Native observations follow the checkpoint's `video`, `state`, and `language` schema. The returned chunk preserves native action names and units; the evaluator performs controller conversion. `io_format: arx_x5` retains the existing bimanual adapter and requires a matching ARX X5 checkpoint and environment configuration. It must not be used with the LIBERO checkpoint.

Validation on 2026-10-05: exact native/adapter action agreement on one captured MuJoCo observation; 4/4 successful LIBERO episodes across two tasks through the WebSocket server; a Mac client also retrieved a real action chunk from the L40S server through an SSH tunnel. These are integration checks, not a benchmark-wide score or hardware qualification.

To repeat the two-task simulation check, install LIBERO separately (tested revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`), configure its asset paths (`LIBERO_CONFIG_PATH` if outside the default location), start the policy server, and run from the evaluator environment:

```sh
python scripts/validate_gr00t_libero.py --host 127.0.0.1 --port 6000 \
  --tasks 0 1 --trials 2 --output runs/gr00t-libero
```

The script uses LIBERO's provided initial states and success checks, runs at most 600 control steps per episode, and executes eight actions from each 16-step prediction before observing again. Videos and JSON results stay outside Git. LIBERO's controller consumes normalized OSC deltas; these native commands must not be sent directly to YAM hardware.
