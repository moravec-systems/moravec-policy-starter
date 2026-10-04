# NVIDIA GR00T N1.7: VLA policy

This uses XPolicyLab's `GR00T_N17` adapter, not a reimplementation of NVIDIA preprocessing. NVIDIA's base model is `nvidia/GR00T-N1.7-3B`. The **specific RoboDojo adapter** expects an ARX X5 joint-space fine-tuned checkpoint and matching modality config; the base model's general relative-EEF interface is not interchangeable with this adapter.

## GPU setup

1. Check out XPolicyLab at the revision in `sources.lock.json`. Read `policy/GR00T_N17/INSTALLATION.md`, then run that directory's `install.sh` in a dedicated environment. Source its `gr00t_n17/.venv/bin/activate`.
2. Obtain/train a compatible RoboDojo checkpoint and generate `configs/arx_x5_config.py` through the upstream data-processing procedure. We do not claim an available unmodified base checkpoint can control ARX X5 or YAM.
3. Choose an explicit checkpoint step in `upstream-config.json`. **60000 is an example selection, not a supplied checkpoint.** Set `cosmos_model_path` to the required backbone. The manifest checkpoint path is the parent containing `checkpoint-60000/`, matching the pinned upstream resolver.
4. Install this SDK into that environment: `python -m pip install -e /path/to/moravec-policy-starter`. Install upstream XPolicyLab editable per its instructions.
5. Set the checkpoint `path`, its `sha256` (from `moravec-policy hash-checkpoint /path/to/checkpoints`), and `status: configured` in `policy.json`.
6. Inject `MORAVEC_POLICY_TOKEN`, then start:

```bash
moravec-policy serve /path/to/moravec-policy-starter/examples/groot/policy.json --timestep 0.05
```

Use the environment's Python/CLI here, not `uv run` from the starter's CPU-only environment. Timesteps must match your benchmark control frequency; 0.05 is illustrative and does not configure the simulator. The adapter loads weights at startup and refuses an unconfigured checkpoint. It maps named RGB/proprioception to upstream observations, calls `update_obs`/`get_action`, and validates the returned native action chunk.

The server returns actions only. A benchmark-side client must consume them using the declared cadence; this sample does not command physical hardware. Native RoboDojo eval is also available through the upstream adapter's `eval.sh`; it uses upstream transport rather than this sample HTTP protocol.

Checkpoint hashing includes configs and normalization files. Upstream temporarily rewrites/restores a processor backbone path while loading; the checkpoint mount may therefore need controlled write access. Validate checkpoint provenance and the loaded modality config before publishing results.
