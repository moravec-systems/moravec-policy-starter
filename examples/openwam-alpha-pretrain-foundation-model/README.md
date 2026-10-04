# openwam-alpha-pretrain-foundation-model

Build a separate GPU image with the official model SDK and its supported runtime. Mount a locally adapted checkpoint read-only, calculate its tree hash with `moravec-policy hash-checkpoint`, and populate policy.json. Replace every binding in bindings.json with checkpoint metadata and the calibrated station profile. Leave status as requires-qualification until simulation confirms action units, ordering, normalization and embodiment compatibility. Changing the status alone is not qualification.

This example has not executed real model inference. See the root [README](../../README.md) and [validation status](../../VALIDATION.md). The checked-in bindings are illustrative placeholders, not checkpoint metadata.
