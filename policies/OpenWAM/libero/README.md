# OpenWAM — LIBERO checkpoint

Task-trained world-action model example for LIBERO, using the shared Moravec server. It consumes the same two-camera observation as GR00T, then applies its own training-specific image sizes, state representation, and action conversion.

The policy resizes overview RGB to 320×256 and wrist RGB to 160×128 with Lanczos. State is achieved XYZ, the first two rotation-matrix columns, and measured gripper opening. Native EEF10 output becomes a physical relative target at 20 Hz. Translation uses the controller's 0.05 m scale, decoded axis-angle uses 0.5 rad, and gripper changes from -1 closed / +1 open to 0 closed / 1 open. The native SDK manages its prediction buffer; the shared server returns one executable step per observation. Compile is disabled for this baseline.

Use the pinned checkpoint revision in `policy.json`. Its checksum covers exactly `checkpoint_step_10690.safetensors`, `config.yaml`, `normalization_stats.npy`, and the complete `tokenizer/` directory. Download metadata, README files, and cache directories are excluded by preparing this separate weights directory. Mount it at the manifest's checkpoint path.

The matching OpenWAM SDK must be installed in the policy environment. Its runtime version is essential: the older public API lacks `generate_batch`. See the root validation report for the tested source. The ordinary CPU base container is insufficient, and this example does not claim a qualified GPU container yet.

```sh
moravec-policy serve policies/OpenWAM/libero/policy.json --host 0.0.0.0
```

Set a private `MORAVEC_POLICY_TOKEN`. Use `sim/validate_libero.py` in a separate LIBERO environment. Qualification is limited to the named simulation profile; it does not establish real-robot capability or support by the foundation checkpoint.
