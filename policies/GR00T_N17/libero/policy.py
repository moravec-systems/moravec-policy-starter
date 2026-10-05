"""Qualification adapter: released GR00T LIBERO checkpoint to Moravec physical units."""

from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from moravec_policy.checkpoint import qualified_checkpoint, validate_binding_profile
from moravec_policy.contracts import ActionChunk
from moravec_policy.lifecycle import Policy
from moravec_policy.provenance import tree_hash


class Model(Policy):
    def __init__(self, manifest, path):
        from gr00t.policy import Gr00tPolicy

        self.cfg, checkpoint = qualified_checkpoint(manifest, path)
        if tree_hash(Path("/weights/Cosmos-Reason2-2B")) != manifest.config["cosmos_sha256"]:
            raise ValueError("Cosmos dependency hash mismatch")
        self.model = Gr00tPolicy(
            model_path=str(checkpoint),
            embodiment_tag="libero_sim",
            device=manifest.config.get("device", "cuda"),
        )
        self.reset()

    def _reset(self):
        self.model.reset()
        self.pending = deque()

    def _infer(self, obs, profile):
        validate_binding_profile(self.cfg, obs)
        if profile.id != "libero-panda" or profile.control_period_s != 0.05:
            raise ValueError("requires validated LIBERO 20Hz profile")
        if self.pending:
            return ActionChunk(timestep_s=0.05, execution_horizon=1, steps=[self.pending.popleft()])
        pose = obs.tcp_poses["arm"]
        xyz = pose.position_m
        rot = Rotation.from_quat(pose.orientation_wxyz, scalar_first=True).as_rotvec()
        state = {
            k: np.asarray([[[v]]], dtype=np.float32)
            for k, v in zip(["x", "y", "z", "roll", "pitch", "yaw"], [*xyz, *rot])
        }
        state["gripper"] = np.asarray([[obs.gripper_joint_positions["gripper"]]], dtype=np.float32)
        video = {
            k: np.frombuffer(obs.cameras[c].data, dtype=np.uint8).reshape(obs.cameras[c].shape)[
                None, None
            ]
            for k, c in [("image", "overview"), ("wrist_image", "wrist")]
        }
        output, _ = self.model.get_action(
            {
                "video": video,
                "state": state,
                "language": {"annotation.human.action.task_description": [[obs.instruction]]},
            }
        )
        actions = np.concatenate(
            [np.asarray(output[k])[0] for k in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]],
            axis=-1,
        )
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError("unexpected native action")
        steps = []
        for a in actions:
            # Native actions are normalized OSC commands; controller scales xyz by .05 and rotvec by .5.
            q = Rotation.from_rotvec(np.clip(a[3:6], -1, 1) * 0.5).as_quat(scalar_first=True)
            steps.append(
                {
                    "tcp_targets": {
                        "arm": {
                            "pose": {
                                "frame": "world",
                                "position_m": np.clip(a[:3], -1, 1) * 0.05,
                                "orientation_wxyz": q,
                            },
                            "representation": "relative",
                            "reference": "frame",
                            "relative_to": "observation",
                        }
                    },
                    "grippers": {"gripper": float((np.sign(2 * a[6] - 1) + 1) / 2)},
                }
            )
        self.pending.extend(steps[:8])
        return ActionChunk(timestep_s=0.05, execution_horizon=1, steps=[self.pending.popleft()])
