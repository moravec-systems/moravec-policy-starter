"""Direct model SDK entrypoints. Checkpoint-owned bindings are required and fail closed."""

import base64
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .contracts import Decision, JointTrajectory
from .provenance import tree_hash


class NativeModelPolicy:
    def __init__(self, manifest, path):
        self.manifest = manifest
        self.cfg = json.loads((Path(path).parent / manifest.backend.binding_config).read_text())
        if self.cfg.get("status") != "qualified":
            raise ValueError(
                "checkpoint/embodiment bindings require qualification; no placeholder inference"
            )
        checkpoint = Path(manifest.backend.checkpoint_path)
        if (
            not manifest.model.checkpoint_sha256
            or tree_hash(checkpoint) != manifest.model.checkpoint_sha256
        ):
            raise ValueError("checkpoint tree hash missing or mismatched")
        if self.cfg["action_semantics"] != "absolute-joint-position":
            raise ValueError("this driver requires qualified absolute-joint output bindings")
        if manifest.backend.driver == "gr00t":
            from gr00t.policy import Gr00tPolicy

            self.model = Gr00tPolicy(
                model_path=str(checkpoint),
                embodiment_tag=self.cfg["embodiment_tag"],
                device="cuda",
                strict=True,
            )
            for name in ["video", "state"]:
                if len(self.model.get_modality_config()[name].delta_indices) != 1:
                    raise ValueError("temporal checkpoint requires a qualified history adapter")
        else:
            from openwam.deploy.server import build_server_from_config

            self.model = build_server_from_config(None, str(checkpoint), device="cuda")

    def reset(self):
        self.model.reset()

    def predict(self, obs, profile):
        if (obs.profile_id, obs.profile_revision) != (
            self.cfg["profile_id"],
            self.cfg["profile_revision"],
        ):
            raise ValueError("checkpoint bindings do not match station profile")
        frames = {
            name: np.frombuffer(base64.b64decode(cam.data), dtype=np.uint8)
            .reshape(cam.shape)
            .copy()
            for name, cam in obs.cameras.items()
        }
        if self.manifest.backend.driver == "gr00t":
            native = {
                "video": {k: frames[v][None, None] for k, v in self.cfg["camera_bindings"].items()},
                "state": {
                    k: np.array([[[obs.joint_positions[n] for n in names]]], dtype=np.float32)
                    for k, names in self.cfg["state_bindings"].items()
                },
                "language": {self.cfg["language_key"]: [[obs.instruction]]},
            }
            output, _ = self.model.get_action(native)
            streams = {k: np.asarray(v)[0] for k, v in output.items()}
        else:
            native = {
                "images": {
                    k: Image.fromarray(frames[v]) for k, v in self.cfg["camera_bindings"].items()
                },
                "prompt": obs.instruction,
                "state": [obs.joint_positions[n] for n in self.cfg["state_order"]],
            }
            result = self.model.predict(native)
            streams = {"action": np.asarray(result["action"]).reshape(1, -1)}
        positions = []
        lengths = {len(streams[k]) for k in self.cfg["action_bindings"]}
        if len(lengths) != 1:
            raise ValueError("action streams have different horizons")
        for i in range(next(iter(lengths))):
            point = {}
            for key, names in self.cfg["action_bindings"].items():
                values = streams[key][i]
                if len(values) != len(names) or set(point).intersection(names):
                    raise ValueError("invalid checkpoint action bindings")
                point.update(zip(names, values.tolist()))
            positions.append(
                {"time_from_start_s": (i + 1) * profile.control_period_s, "positions": point}
            )
        return Decision(
            episode_id=obs.episode_id,
            observation_sequence=obs.sequence,
            decision_id=str(uuid4()),
            valid_until_station_time_s=obs.captured_at_s + profile.max_observation_age_s,
            evidence_kind="model-inference",
            action=JointTrajectory(kind="joint-trajectory", points=positions),
        )
