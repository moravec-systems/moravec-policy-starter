"""XPolicyLab bridge; never reimplement GR00T/OpenWAM model preprocessing."""

from __future__ import annotations

import base64
import importlib
import json
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np

from .contracts import ActionChunk, JointStep, Observation, PolicyManifest, XPolicyBackend


class Policy(Protocol):
    def reset(self) -> None: ...
    def predict(self, observation: Observation) -> ActionChunk: ...


def verify_checkout(root: Path, revision: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"upstream revision mismatch: expected {revision}, found {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "diff", "HEAD", "--name-only"], text=True
    ).strip()
    if dirty:
        raise ValueError("upstream has tracked changes; use a clean pinned checkout")


def to_xpolicy(obs: Observation) -> dict:
    """Raw RGB transport is decoded here; no legacy XPolicy image-bit decoding."""
    state = {}
    for side in ("left", "right"):
        arm = getattr(obs, side)
        state[f"{side}_arm_joint_state"] = np.asarray(arm.joints, dtype=np.float32)
        state[f"{side}_ee_joint_state"] = np.asarray([arm.gripper], dtype=np.float32)
        state[f"{side}_ee_pose"] = np.asarray(arm.ee_pose, dtype=np.float32)
    return {
        "instruction": obs.instruction,
        "env_idx": 0,
        "vision": {
            name: {
                "color": np.frombuffer(base64.b64decode(cam.data), dtype=np.uint8)
                .reshape(cam.shape)
                .copy()
            }
            for name, cam in obs.cameras.items()
        },
        "state": state,
    }


def from_xpolicy(raw: list, obs: Observation, action_type: str, timestep_s: float) -> ActionChunk:
    steps = []
    arm_key = "arm_joint_state" if action_type == "joint" else "ee_pose"
    for step in raw:
        expected = {f"{s}_{k}" for s in ("left", "right") for k in (arm_key, "ee_joint_state")}
        if set(step) != expected:
            raise ValueError("upstream action keys do not match the negotiated profile")
        result = {"kind": action_type}
        for side in ("left", "right"):
            result[side] = np.asarray(step[f"{side}_{arm_key}"]).reshape(-1).tolist()
            grip = np.asarray(step[f"{side}_ee_joint_state"]).reshape(-1)
            if grip.size != 1:
                raise ValueError("expected one gripper value per arm")
            result[f"{side}_gripper"] = float(grip[0])
        steps.append(result)
    return ActionChunk(
        episode_id=obs.episode_id,
        observation_sequence=obs.sequence,
        action_type=action_type,
        timestep_s=timestep_s,
        steps=steps,
        evidence_kind="model_inference",
    )


class XPolicyAdapter:
    def __init__(self, model, action_type: str, timestep_s: float):
        self.model = model
        self.action_type = action_type
        self.timestep_s = timestep_s

    def reset(self):
        self.model.reset()

    def predict(self, observation):
        self.model.update_obs(to_xpolicy(observation))
        return from_xpolicy(self.model.get_action(), observation, self.action_type, self.timestep_s)


class MockPolicy:
    """Hold measured state; contract fixture only, never a robot-ready policy."""

    def reset(self):
        pass

    def predict(self, observation):
        return ActionChunk(
            episode_id=observation.episode_id,
            observation_sequence=observation.sequence,
            action_type="joint",
            timestep_s=0.05,
            evidence_kind="mock",
            steps=[
                JointStep(
                    kind="joint",
                    left=observation.left.joints,
                    right=observation.right.joints,
                    left_gripper=observation.left.gripper,
                    right_gripper=observation.right.gripper,
                )
            ],
        )


def load_policy(manifest: PolicyManifest, manifest_path: Path, timestep_s: float) -> Policy:
    if manifest.backend.kind == "mock":
        return MockPolicy()
    if not isinstance(manifest.backend, XPolicyBackend):
        raise TypeError("Astra uses the native RoboProbe harness; use the astra-plan command")
    backend = manifest.backend
    config = json.loads((manifest_path.parent / backend.config_path).read_text())
    if backend.checkpoint.status != "configured":
        raise ValueError("configure a compatible checkpoint before loading real weights")
    if not backend.checkpoint.sha256:
        raise ValueError("pin the checkpoint directory with its tree hash")
    from .provenance import tree_hash

    ckpt = Path(backend.checkpoint.path).expanduser().resolve()
    if tree_hash(ckpt) != backend.checkpoint.sha256:
        raise ValueError("checkpoint hash mismatch")
    if config.get("allow_dummy_policy", False):
        raise ValueError("dummy policy prohibited for real model examples")
    if (
        config.get("action_type") != manifest.action_type
        or config.get("env_cfg_type") != manifest.embodiment
    ):
        raise ValueError("upstream config does not match manifest")
    package = importlib.import_module("XPolicyLab")
    root = Path(package.__file__).resolve().parent
    verify_checkout(root, backend.source.revision)
    # The checkpoint paths have different meanings in the two upstream adapters.
    if backend.policy_name == "GR00T_N17":
        config.pop("model_dir", None)
    config["ckpt_dir" if backend.policy_name == "OpenWAM" else "ckpt_name"] = str(ckpt)
    model_module = importlib.import_module(f"XPolicyLab.policy.{backend.policy_name}.model")
    return XPolicyAdapter(model_module.Model(config), manifest.action_type, timestep_s)
