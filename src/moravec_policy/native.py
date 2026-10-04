"""Direct model SDK entrypoints. Checkpoint-owned bindings are required and fail closed."""

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from .lifecycle import Policy
from .provenance import tree_hash
from .state_adapter import state_vector


class NativeModelPolicy(Policy):
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
        if manifest.backend.driver == "gr00t":
            from gr00t.policy import Gr00tPolicy

            self.model = Gr00tPolicy(
                model_path=str(checkpoint),
                embodiment_tag=self.cfg["embodiment_tag"],
                device="cuda",
                strict=True,
            )
            self.modalities = self.model.get_modality_config()
            offsets = [
                i for name in ["video", "state"] for i in self.modalities[name].delta_indices
            ]
            if any(i > 0 for i in offsets):
                raise ValueError("future observations are unavailable")
            self.history_length = 1 - min(offsets)

        else:
            from openwam.deploy.server import build_server_from_config

            self.model = build_server_from_config(None, str(checkpoint), device="cuda")

    def _reset(self):
        self.model.reset()
        self.history = deque(maxlen=getattr(self, "history_length", 1))

    def update_obs(self, obs):
        super().update_obs(obs)
        if self.history and self.history[-1].sequence >= obs.sequence:
            raise ValueError("observation history must increase")
        if self.history:
            period = self.cfg["observation_period_s"]
            if not np.isclose(
                obs.captured_at_s - self.history[-1].captured_at_s, period, rtol=0.1, atol=1e-6
            ):
                raise ValueError("observation cadence differs from checkpoint bindings")
        self.history.append(obs)

    def _infer(self, obs, profile):
        if (obs.profile_id, obs.profile_revision) != (
            self.cfg["profile_id"],
            self.cfg["profile_revision"],
        ):
            raise ValueError("checkpoint bindings do not match station profile")
        frames = {
            name: np.frombuffer(cam.data, dtype=np.uint8).reshape(cam.shape).copy()
            for name, cam in obs.cameras.items()
        }
        if self.manifest.backend.driver == "gr00t":
            if len(self.history) < self.history_length:
                raise ValueError("checkpoint requires more observation history before inference")

            def samples(modality):
                return [self.history[int(i) - 1] for i in self.modalities[modality].delta_indices]

            native = {
                "video": {
                    k: np.stack(
                        [
                            np.frombuffer(o.cameras[v].data, dtype=np.uint8).reshape(
                                o.cameras[v].shape
                            )
                            for o in samples("video")
                        ]
                    )[None]
                    for k, v in self.cfg["camera_bindings"].items()
                },
                "state": {
                    k: np.array(
                        [[state_vector(o, binding) for o in samples("state")]],
                        dtype=np.float32,
                    )
                    for k, binding in self.cfg["state_bindings"].items()
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
            array = np.asarray(result["action"])
            streams = {"action": array[None] if array.ndim == 1 else array}
        from .action_adapter import decode_actions

        return decode_actions(streams, self.cfg, profile)
