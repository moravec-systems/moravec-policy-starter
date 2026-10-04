# Customer model integration

Goal: adapt a specified model to the existing Moravec observation/action contract and produce a reproducible container submission.

Required inputs: exact model/checkpoint revision, model SDK version, authorized checkpoint access, robot profile and calibration version, camera/state/action metadata, and a suitable compute environment. Report missing inputs before claiming integration is complete.

- Keep the versioned contract, station controller, task scene, evaluator, seeds and acceptance criteria unchanged. Propose incompatible requirements separately.
- Implement only model loading, reset, observation conversion and action conversion. Use the existing service lifecycle and validation.
- Verify units, coordinate frames, joint ordering, normalization, action horizon and control frequency against actual checkpoint metadata. Never guess from tensor shape.
- Do not replace real inference with a fixture, silently select another model, or mark placeholder bindings qualified.
- Run offline contract tests first. API/GPU runs require the user's authorized scope; physical robot motion requires separate supervised authorization.
- Report fixture tests, real inference, simulation task outcomes and hardware validation separately. A valid JSON response is not task success.
- Keep credentials and weights out of Git. Pin the image, SDK/checkpoint and bindings; document compute requirements and network needs.
- Deliver the adapter, manifest, container recipe, executed test commands, evidence paths, version hashes and remaining limitations.

Suggested task prompt:

“Adapt the specified checkpoint to this repository's policy interface for the provided robot profile. Preserve the evaluator and task configuration. Run conformance tests, then one authorized simulation episode using real inference. Produce an immutable container submission and a report separating passed checks from unverified assumptions. If checkpoint metadata or runtime access is missing, identify it precisely; do not fabricate a successful integration.”
