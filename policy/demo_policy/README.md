# Demo policy

Minimal Moravec client/server example. It returns two synthetic zero-action steps with the configured joint or end-effector shapes. It tracks observation environment IDs for batch selection, but ignores image/state contents and loads no checkpoint. Never send its actions to physical hardware.

From the repository root:

```sh
python -m pip install -e .
python scripts/demo_roundtrip.py
```

The example supplies temporary configuration and exercises observations, action retrieval, batching, and reset. No simulator or GPU is required.

To build an adapter, start with `model.py`, keep deployment settings in `deploy.yml`, and follow the [integration guide](../../INTEGRATION.md).
