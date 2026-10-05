"""Moravec policy entrypoints. Fixture identity never claims a real model result."""

import json

import numpy as np

from moravec_policy.contracts import ActionChunk, Pose
from moravec_policy.lifecycle import Policy


class FixturePolicy(Policy):
    def __init__(self, output):
        self.output = output

    def _reset(self):
        pass

    def _infer(self, obs, profile):
        # Targets are explicit task instructions, not hidden object/scorer state.
        target = json.loads(obs.instruction)
        if self.output == "joint-trajectory":
            action = ActionChunk(
                timestep_s=1.0,
                execution_horizon=1,
                steps=[{"joint_positions": target["joint_target"]}],
            )
        else:
            from moravec_policy.geometry import multiply

            base = obs.tcp_poses["arm"]
            q = np.asarray(base.orientation_wxyz)
            delta = multiply(target["orientation_wxyz"], q * np.array([1, -1, -1, -1]))
            action = ActionChunk(
                timestep_s=1.0,
                execution_horizon=1,
                steps=[
                    {
                        "tcp_targets": {
                            "arm": {
                                "representation": "relative",
                                "pose": Pose(
                                    frame=base.frame,
                                    position_m=np.asarray(target["tcp_target_m"]) - base.position_m,
                                    orientation_wxyz=delta,
                                ),
                            }
                        }
                    }
                ],
            )
        return action


def build_policy(manifest, path):
    if manifest.family != "fixture":
        raise ValueError("fixture must declare its evidence identity")
    output = manifest.config["output"]
    if output not in {"joint-trajectory", "relative-pose"}:
        raise ValueError("unsupported fixture output")
    return FixturePolicy(output)
