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
