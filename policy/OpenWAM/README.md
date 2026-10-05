# OpenWAM

Moravec world-action policy adapter. The model implementation is in `OpenWAM/`; `model.py` handles observation and action conversion and `deploy.yml` configures inference.

Run `bash install.sh` from this directory to install the adapter and model runtime. Follow the [model documentation](OpenWAM/README.md) for checkpoint downloads and runtime requirements. Use a compatible checkpoint with its configuration and normalization files, then set the checkpoint and environment in `deploy.yml`.

The current adapter uses ARX X5-specific coordinate transforms and camera mappings. It is not a generic YAM policy. Preserve checkpoint format and loader identifiers when configuring inference.

See the [integration guide](../../INTEGRATION.md#remote-serving) for the Python server and remote client. Training scripts are retained as optional model utilities. The current shared demo validates transport only; this fork still requires real-checkpoint simulation qualification.
