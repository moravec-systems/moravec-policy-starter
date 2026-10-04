# Qualifying the contract in simulation

Use four gates. Passing one does not imply passing later gates.

1. **Schema and transport:** reject wrong episodes/sequences, stale cameras, unknown frames, invalid units/shapes, nonfinite values, malformed tools, provider failures and out-of-range actions. Confirm reset clears model state and execution feedback matches the action ID.
2. **Physical conformance:** exercise actual HTTP, RGB rendering, named robot state, controller execution, and measured feedback in a small deterministic scene. Compare Cartesian-goal and joint-trajectory paths. Physics warnings invalidate an episode. This repository supplies a two-joint MuJoCo scene and five seeded goals per path.
3. **Isaac Sim integration:** run the same container API against a pinned Isaac Sim build and a supported calibrated embodiment. Start with reach, then a rigid-object pick/place task using existing assets. Pin scene/assets, camera intrinsics/extrinsics, units, transforms, controller and simulator timestep. Test latency, stale commands, disconnects, unreachable poses, collisions, resets, and concurrent isolated replicas. Require actual multiview video and state traces. Never infer success from an API 200 response.
4. **Real model and station qualification:** replace the test double with one exact model at a time. First a short Astra tool loop, then a checkpoint with verified action bindings for GR00T and OpenWAM. Keep the same observation/action schema in sim and the real station; qualify embodiment-specific adapters and safety controls separately. Real hardware requires independent readiness checks and attended authorization.

## Acceptance criteria

For the included scene, each of ten episodes must reach within 1 cm of its target, with no physics warning and correct feedback/sequence/reset behavior. The target is explicitly included in the fixture instruction; this tests mechanics, not visual reasoning. Both paths use two decisions per episode. The planar executor is deliberately limited to one-point, one-second trajectories and a reachable planar Cartesian pose. It is not a general robot planner.

For Isaac integration, require the same contract examples to execute without changing message semantics, accurate timestamps and transforms, repeatable resets, evidence-linked independent scoring, and zero actuation for rejected decisions. Then test camera-only task reasoning: remove numeric target instructions and keep privileged scoring state inaccessible to the policy. Select thresholds and trial counts before evaluating real models.

## What this reveals about the proposed schema

- Station clock, observation sequence and decision expiry are needed to reject delayed inference.
- Named joints, units and calibrated frames are necessary; anonymous float vectors are insufficient.
- Checkpoint action semantics must be explicit. A valid tensor shape does not establish a valid robot command.
- An action needs measured feedback; returning a tool call does not mean a motion succeeded.
- Harness state must reset between trials, and each parallel trial needs an isolated session.
- Prediction artifacts from a world model are not interchangeable with executable actions.

The existing tests establish only the first two gates. Container execution, Isaac Sim integration, model weights, and physical deployment are not yet validated.
