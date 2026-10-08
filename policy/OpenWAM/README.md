# OpenWAM

Moravec policy adapter for [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM), a world-action model. `model.py` converts observations and actions; `deploy.yml` selects checkpoint and serving settings. OpenWAM itself is installed from its repository at a pinned commit.

## Setup

In a dedicated GPU environment with a suitable PyTorch/CUDA combination, clone OpenWAM at the pinned commit and install it with this starter:

```sh
git clone https://github.com/OpenWAM-Official/OpenWAM policy/OpenWAM/OpenWAM
git -C policy/OpenWAM/OpenWAM checkout 48bd67b89d489b14d03b8d92bc66e65d306df32e
python -m pip install -e policy/OpenWAM/OpenWAM -e .
```

The editable install matters: the adapter reads OpenWAM's default `configs/deploy.yaml` from the checkout. `model.py` contains ARX X5-specific coordinate transforms; it is not a generic YAM policy. Batched requests run one OpenWAM generation per environment.

## Serving

Activate the environment, then use the shared Python server from the repository root as described in the [integration guide](../../INTEGRATION.md#remote-serving). Weights are separate downloads.

## Research fixture

Download the [released simulation checkpoint](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-RoboDojo) into the path selected by `deploy.yml`:

```sh
hf download OpenWAM/OpenWAM-Alpha-Sim-RoboDojo \
  --revision 2c1302294e3ba8319bbdb2c803b7a27de9292d03 \
  --local-dir policy/OpenWAM/checkpoints/openwam-sim
```

This fixture requires the evaluator's ARX X5 environment configuration. It does not qualify YAM or FR3 hardware. The example uses single-environment evaluation.

## Container

Build from the repository root on a Linux x86_64 NVIDIA GPU host with Docker and the NVIDIA Container Toolkit:

```sh
docker build -f policy/OpenWAM/Dockerfile -t moravec-openwam:demo .
docker run --rm --gpus all --shm-size=2g \
  --name moravec-openwam -p 127.0.0.1:6000:6000 \
  -v "$PWD/policy/OpenWAM/checkpoints/openwam-sim:/weights/openwam-sim:ro" \
  -v moravec-hf-cache:/cache/huggingface \
  moravec-openwam:demo
```

The image installs OpenWAM at the pinned commit above; weights stay outside the image. Download the checkpoint above before starting it, including its configuration and normalization files. If a configured backbone requires authenticated downloads, inject `HF_TOKEN` with `--env HF_TOKEN` from your host environment. Never include credentials in the image or submission files.

In another terminal, with the starter installed (`pip install -e .`), send synthetic three-camera observations and verify the returned action chunks:

```sh
moravec-validate policy/OpenWAM --url ws://127.0.0.1:6000
```

The validator executes no actions. Container health means the server has initialized and accepts WebSocket connections; it does not establish task success. Use simulation observations and controllers for task evaluation. The container preserves this checkpoint's ARX X5 coordinate transforms and gripper conventions; it is not a YAM or FR3 deployment. The foundation pretraining checkpoint has a different action representation and cannot replace this simulation checkpoint directly.
