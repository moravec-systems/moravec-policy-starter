# Pi_05

Moravec adapter for Physical Intelligence π0.5. The model implementation is in `openpi/`; inference configuration is in `deploy.yml` and preprocessing is in `model.py`.

Run `bash install.sh` from this directory to prepare the model-specific runtime. Follow the [model documentation](openpi/README.md) for runtime and checkpoint requirements. Configure the checkpoint path, training configuration, normalization assets, and environment in `deploy.yml` to match the chosen checkpoint; the bundled defaults are an example, not a YAM-trained model.

Serve with the shared Python entry point described in the [integration guide](../../INTEGRATION.md#remote-serving). Data conversion and training scripts are optional retained utilities. The adapter maps evaluator observations into the checkpoint's inputs and converts predictions back into robot actions.

The current shared demo validates transport only. This fork still requires real-checkpoint simulation qualification.
