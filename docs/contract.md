# Contract decisions (proposal v0.1)

## Three separate records

- **PolicyManifest:** identity/family, embodiment and observation/action profile, source pin, backend-specific settings, checkpoint identity. Family is metadata; runtime compatibility is enforced through the selected backend/profile.
- **DeploymentManifest:** policy-manifest hash plus container image digest or remote endpoint and secret reference. This describes deployment; it does not provision infrastructure.
- **EvaluationSpec:** benchmark/runtime/task/layouts and retry policy. This describes intent; the native runner and scorer remain authoritative. `official_protocol` is a requested scope label, not a validator's certification of official compliance.

Pydantic models in `src/moravec_policy/contracts.py` generate `schemas/*.json`. Extra fields are rejected. Observation payloads cannot contain hidden task state or reward fields. The interface supports only the two-arm ARX X5 profile currently demonstrated by the examples. Do not mark YAM or FR3 Duo supported until its joint/actuator/calibration profile and corresponding checkpoint have passed integration validation.

## Observation wire format

`Observation` includes episode ID, monotonically increasing sequence, capture time, instruction, three RGB cameras, and measured left/right joint/gripper/EE states. Each camera carries its own capture timestamp. Timestamps are nanoseconds on the environment's monotonic timeline, not cross-host wall-clock synchronization. This prototype validates ordering but does not impose a maximum camera skew or age; an environment-specific executor must enforce those bounds.

For portability the sample HTTP transport uses raw HWC uint8 RGB bytes encoded in base64. This is deliberately distinct from XPolicyLab's legacy encoded-image representation; the bridge decodes only this declared format and passes arrays to `update_obs`. No RGB↔BGR conversion. Image dimensions and byte count are validated. Production deployments should negotiate a binary/compressed transport and body-size limits before handling high-rate video.

## Action semantics

Joint vectors use the six-element native ARX X5 order, independently for each arm; angles are radians. EE poses are native RoboDojo environment-relative world `[x,y,z,qw,qx,qy,qz]`, meters and unit quaternion. The frame profile refers to the existing benchmark's EE definition, not the gripper grasp face. Grippers use the benchmark's normalized convention, 0 closed / 1 open. Each action is an absolute native target after upstream postprocessing. Joint delta conventions inside a trained model must be resolved by its own verified adapter.

`ActionChunk` echoes episode/observation identity, declares one action type and a timestep in seconds. The runtime must enforce velocity/position limits, action freshness, chunk truncation, interruption and measured execution feedback. The SDK does not do IK, adjust reference-frame transforms, score episodes or actuate hardware. The reference HTTP server demonstrates inference only; no independent claim of hardware safety.

Astra's **model-facing** grasp-point position/Euler tools are intentionally not this robot-facing EE schema. The pinned RoboProbe harness owns grasp-point→flange conversion and CuRobo trajectories. Replacing that conversion would define a different harness.

## Session semantics

One active episode per server process. Calls are serialized, sequence numbers strictly increase, and a second start is rejected until end. A failed inference poisons the episode until end/reset; no silent retry may advance model memory twice. Repeated requests receive 409, not a cached action. If a response is lost, reconcile/abort the episode; do not execute an uncertain action again. This is conservative inference semantics, not exactly-once physical execution.

HTTP routes: `/health`, `/v1/capabilities`, `/v1/episode/start`, `/v1/predict`, `/v1/episode/end`. Except health, all require bearer authentication. No implicit batching. An episode reset must clear model state; tests verify the wrapper invokes upstream reset. Actual third-party model reset behavior remains a live integration gate.
