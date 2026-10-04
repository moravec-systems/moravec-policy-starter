# Validation gates

1. **Schema:** all supplied manifests parse; observations/actions reject bad dimensions, nonfinite values, wrong camera profile, hidden fields, invalid quaternions and mixed action spaces.
2. **CPU contract replay:** mock service→remote client works with synthetic RGB; RGB order, response identity, auth, episode reset and stale request rejection are tested. This proves wiring only.
3. **Real inference:** compatible checkpoint/config/source hashes, model-specific installation, representative recorded observations, valid returned actions, model reset checks and latency/memory measurements. Not completed for these three examples.
4. **Live sim:** frozen task/layout/runtime, native control and scorer, synchronized RGB/video/action record, no hidden-state leakage, finalized score with explicit termination. Not completed from this repo.
5. **Real robot:** calibration and control limits verified, fresh cameras/telemetry, supervised initial trial, independent stop mechanism. Never inferred from simulation success.

Record `task_failure`, `infrastructure_fault`, and `invalid_execution` separately. The starter's 502 means an inference fault, not a failed task score. Preserve all attempts; retry rules cannot turn an interrupted episode into a silent success. Benchmark results must identify exact model/provider, prompts/harness/config, checkpoint tree hash, adapter/source revisions, simulator version, layouts, call/step limits and artifacts.

The source pin checks tracked changes but does not audit the entire Python environment or untracked generated configs. Freeze modality config and dependency lockfiles in the evaluation artifact before publishing. Container deployment schemas require image digests; the CPU Dockerfile itself is an illustrative build and should be locked by resulting digest for release.
