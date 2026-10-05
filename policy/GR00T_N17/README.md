# NVIDIA GR00T N1.7

Moravec adapter for `nvidia/GR00T-N1.7-3B`. The model implementation is in `gr00t_n17/`; the adapter and deployment configuration are at this folder's root.

Read [INSTALLATION.md](INSTALLATION.md) for the GPU runtime and checkpoint prerequisites. Run `bash install.sh` from this directory to install the policy runtime. The runtime uses `gr00t_n17/.venv`.

Configure `deploy.yml` with a checkpoint and matching environment configuration before serving. The checkpoint's embodiment, camera mapping, action dimensions, and normalization must match the evaluator. Adding YAM dimensions alone does not make a pretrained checkpoint compatible with YAM.

Use the [integration guide](../../INTEGRATION.md#remote-serving) to launch the Python server locally or remotely. Training and data-conversion scripts are retained as optional upstream utilities; they are not prerequisites when using a compatible prepared checkpoint.

The current shared demo validates transport only. This fork still requires real-checkpoint simulation qualification.
