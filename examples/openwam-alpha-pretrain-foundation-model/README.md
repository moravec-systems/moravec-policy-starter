# openwam-alpha-pretrain-foundation-model

Build a separate GPU image with the official model SDK and its supported runtime. Mount a locally adapted checkpoint read-only, calculate its tree hash with `moravec-policy hash-checkpoint`, and populate policy.json. Replace every binding in bindings.json with checkpoint metadata and the calibrated station profile. Leave status as requires-qualification until simulation confirms action units, ordering, normalization and embodiment compatibility. Changing the status alone is not qualification.

The official SDK has passed a synthetic GPU inference smoke test with the base checkpoint. This starter adapter has not yet executed a qualified simulation episode; the base checkpoint requires embodiment-specific adaptation and action mapping. See the root [README](../../README.md) and [validation status](../../VALIDATION.md). The checked-in bindings are illustrative placeholders, not checkpoint metadata.

The official SDK requires `head_camera` in its image dictionary; `left_wrist_camera` and `right_wrist_camera` are optional SDK keys. Bind station camera names to those keys. Do not interpret the foundation checkpoint's 80-dimensional output as physical commands for an arbitrary robot.
