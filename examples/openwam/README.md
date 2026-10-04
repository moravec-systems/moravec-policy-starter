# OpenWAM: open world-action model

Chosen because it predicts actions and has an existing RoboDojo adapter. It is not a video-only generator labeled as a robot policy. Upstream source: https://github.com/OpenWAM-Official/OpenWAM (Apache-2.0 code; check each downloaded asset's own terms).

## GPU setup

1. Use the pinned XPolicyLab checkout from `sources.lock.json` and follow `policy/OpenWAM/README.md` / `install.sh` in its own CUDA environment. The adapter vendors a modified OpenWAM tree with batched inference; do not replace it with latest official source without compatibility testing.
2. Obtain a released RoboDojo SFT checkpoint via upstream's `scripts/download_assets/download_openwam_checkpoints.py`. The checkpoint directory must contain `config.yaml`, checkpoint safetensors and `normalization_stats.npy` (plus the referenced tokenizer/assets). No weights are downloaded by this sample.
3. Set checkpoint path/hash/status in `policy.json`; use `moravec-policy hash-checkpoint /path/to/checkpoint`.
4. Install this SDK into that model environment, inject `MORAVEC_POLICY_TOKEN`, and run:

```bash
moravec-policy serve /path/to/moravec-policy-starter/examples/openwam/policy.json --timestep 0.05
```

`upstream-config.json` selects eight executed actions per prediction; this is a sample configuration, **not** a claim of matching published results. Match the checkpoint/control frequency before live use. The upstream adapter owns the EEF20 representation, robot-base/world transforms and normalization, then returns absolute world EE poses with WXYZ quaternions plus normalized gripper commands. This starter performs no extra coordinate transform.

Dummy inference is prohibited here. CPU contract tests use the separate `mock` example, never an OpenWAM fallback. The adapter's generated-video path is disabled in the pinned integration; task evidence must come from the actual simulator cameras, not predicted video. The wrapper currently runs one episode at a time even though upstream supports batching.
