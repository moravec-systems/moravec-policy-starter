"""Moravec policy entrypoints. Fixture identity never claims a real model result."""

import json
from pathlib import Path
from uuid import uuid4

from .contracts import CartesianGoal, Decision, FixtureBackend, JointTrajectory, ModelEndpoint, Pose
from .llm import LLMHarness


class FixturePolicy:
    def __init__(self, output):
        self.output = output

    def reset(self):
        pass

    def predict(self, obs, profile):
        # Targets are explicit task instructions, not hidden object/scorer state.
        target = json.loads(obs.instruction)
        if self.output == "joint-trajectory":
            action = JointTrajectory(
                kind="joint-trajectory",
                points=[{"time_from_start_s": 1.0, "positions": target["joint_target"]}],
            )
        else:
            action = CartesianGoal(
                kind="cartesian-goal",
                arm="arm",
                duration_s=1.0,
                target=Pose(
                    frame="world",
                    position_m=target["tcp_target_m"],
                    orientation_wxyz=target["orientation_wxyz"],
                ),
            )
        return Decision(
            episode_id=obs.episode_id,
            observation_sequence=obs.sequence,
            decision_id=str(uuid4()),
            valid_until_station_time_s=obs.captured_at_s + profile.max_observation_age_s,
            evidence_kind="contract-fixture",
            action=action,
        )


def load_policy(manifest, path: Path):
    if isinstance(manifest.backend, FixtureBackend):
        return FixturePolicy(manifest.backend.output)
    if isinstance(manifest.backend, ModelEndpoint):
        return LLMHarness(manifest)
    from .native import NativeModelPolicy

    return NativeModelPolicy(manifest, path)
