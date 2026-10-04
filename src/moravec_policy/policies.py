"""Moravec policy entrypoints. Fixture identity never claims a real model result."""

import json
from pathlib import Path

import numpy as np

from .contracts import FixtureBackend, JointTrajectory, ModelEndpoint, Pose, PoseTrajectory
from .lifecycle import Policy
from .llm import LLMHarness


class FixturePolicy(Policy):
    def __init__(self, output):
        self.output = output

    def _reset(self):
        pass

    def _infer(self, obs, profile):
        # Targets are explicit task instructions, not hidden object/scorer state.
        target = json.loads(obs.instruction)
        if self.output == "joint-trajectory":
            action = JointTrajectory(
                kind="joint-trajectory",
                points=[{"time_from_start_s": 1.0, "positions": target["joint_target"]}],
            )
        else:
            from .geometry import multiply

            base = obs.tcp_poses["arm"]
            q = np.asarray(base.orientation_wxyz)
            delta = multiply(target["orientation_wxyz"], q * np.array([1, -1, -1, -1]))
            action = PoseTrajectory(
                kind="pose-trajectory",
                arm="arm",
                representation="relative",
                reference="frame",
                relative_to="observation",
                timestep_s=1.0,
                execution_horizon=1,
                points=[
                    {
                        "pose": Pose(
                            frame=base.frame,
                            position_m=np.asarray(target["tcp_target_m"]) - base.position_m,
                            orientation_wxyz=delta,
                        )
                    }
                ],
            )
        return action


def load_policy(manifest, path: Path):
    if isinstance(manifest.backend, FixtureBackend):
        return FixturePolicy(manifest.backend.output)
    if isinstance(manifest.backend, ModelEndpoint):
        return LLMHarness(manifest)
    from .native import NativeModelPolicy

    return NativeModelPolicy(manifest, path)
