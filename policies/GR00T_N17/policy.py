"""GR00T owns its checkpoint bindings, temporal observations and SDK configuration."""

from collections import deque

import numpy as np

from moravec_policy.action_adapter import decode_actions
from moravec_policy.checkpoint import qualified_checkpoint, validate_binding_profile
from moravec_policy.lifecycle import Policy
from moravec_policy.state_adapter import state_vector


class Model(Policy):
    def __init__(self, manifest, path):
        self.cfg, checkpoint = qualified_checkpoint(manifest, path)
        from gr00t.policy import Gr00tPolicy

        self.model = Gr00tPolicy(
            model_path=str(checkpoint),
            embodiment_tag=self.cfg["embodiment_tag"],
            device=manifest.config.get("device", "cuda"),
            strict=True,
        )
        self.modalities = self.model.get_modality_config()
        offsets = [i for name in ["video", "state"] for i in self.modalities[name].delta_indices]
        if not offsets or any(i > 0 for i in offsets):
            raise ValueError("checkpoint must use present or past observations")
        self.history_length = 1 - min(offsets)
        self.reset()

    def _reset(self):
        self.model.reset()
        self.history = deque(maxlen=self.history_length)

    def update_obs(self, obs):
        if self.history:
            if self.history[-1].sequence >= obs.sequence:
                raise ValueError("observation history must increase")
            if not np.isclose(
                obs.captured_at_s - self.history[-1].captured_at_s,
                self.cfg["observation_period_s"],
                rtol=0.1,
                atol=1e-6,
            ):
                raise ValueError("observation cadence differs from checkpoint bindings")
        super().update_obs(obs)
        self.history.append(obs)

    def _infer(self, obs, profile):
        validate_binding_profile(self.cfg, obs)
        if len(self.history) < self.history_length:
            raise ValueError("checkpoint requires more observation history before inference")

        def samples(modality):
            return [self.history[int(i) - 1] for i in self.modalities[modality].delta_indices]

        native = {
            "video": {
                k: np.stack(
                    [
                        np.frombuffer(o.cameras[v].data, dtype=np.uint8).reshape(o.cameras[v].shape)
                        for o in samples("video")
                    ]
                )[None]
                for k, v in self.cfg["camera_bindings"].items()
            },
            "state": {
                k: np.array(
                    [[state_vector(o, binding) for o in samples("state")]], dtype=np.float32
                )
                for k, binding in self.cfg["state_bindings"].items()
            },
            "language": {self.cfg["language_key"]: [[obs.instruction]]},
        }
        output, _ = self.model.get_action(native)
        return decode_actions({k: np.asarray(v)[0] for k, v in output.items()}, self.cfg, profile)
