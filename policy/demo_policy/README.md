# Demo policy

Minimal Moravec client/server example. It returns two synthetic zero-action steps with the configured joint or end-effector shapes. It tracks observation environment IDs for batch selection, but ignores image/state contents and loads no checkpoint. Never send its actions to physical hardware.

From the repository root:

```sh
python -m pip install -e .
python scripts/demo_roundtrip.py
```

The example supplies temporary configuration and exercises observations, action retrieval, batching, and reset. No simulator or GPU is required.

To build an adapter, start with `model.py`, keep deployment settings in `deploy.yml`, and follow the [integration guide](../../INTEGRATION.md).

## Container quickstart

This CPU-only container supplies synthetic YAM dimensions internally. It needs
Docker, but no GPU, checkpoint, simulator, or external evaluator configuration.
From the repository root:

```sh
docker build -f policy/demo_policy/Dockerfile -t moravec-demo-policy:demo .
docker run --rm -d --name moravec-demo-policy \
  -p 127.0.0.1:6000:6000 moravec-demo-policy:demo
moravec-validate policy/demo_policy --url ws://127.0.0.1:6000
docker stop moravec-demo-policy
```

`moravec-validate` (installed with `pip install -e .`) checks the manifest, then sends
synthetic observations and checks the two returned action steps, batching, and reset.
No actions are executed. Port 6000 must be available. For a validator
on another machine, use the SSH tunnel in the integration guide. This is a
transport example, not a learned policy or a physical robot controller.
