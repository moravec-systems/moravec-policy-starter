"""OpenWAM-specific deployment and input/output conversion; no family dispatch in core."""

from pathlib import Path

import numpy as np
from PIL import Image

from moravec_policy.action_adapter import decode_actions
from moravec_policy.checkpoint import qualified_checkpoint, validate_binding_profile
from moravec_policy.lifecycle import Policy
from moravec_policy.state_adapter import state_vector


class Model(Policy):
    def __init__(self, manifest, path):
        self.cfg, checkpoint = qualified_checkpoint(manifest, path)
        from omegaconf import OmegaConf
        from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

        config_path = manifest.config.get("deploy_config")
        if config_path:
            config_path = str((Path(path).parent / config_path).resolve())
        deploy = _load_deploy_yaml(config_path)
        deploy = OmegaConf.merge(deploy, manifest.config.get("inference_overrides", {}))
        self.model = build_server_from_config(
            deploy, str(checkpoint), device=manifest.config.get("device", "cuda")
        )
        self.reset()

    def _reset(self):
        self.model.reset()

    def _infer(self, obs, profile):
        validate_binding_profile(self.cfg, obs)
        images = {
            k: Image.fromarray(
                np.frombuffer(obs.cameras[v].data, dtype=np.uint8).reshape(obs.cameras[v].shape)
            )
            for k, v in self.cfg["camera_bindings"].items()
        }
        # Explicit ordered components support joint, gripper and pose checkpoints.
        # Frame conversion belongs here when a checkpoint needs a calibrated transform.
        state = [
            value
            for binding in self.cfg["state_components"]
            for value in state_vector(obs, binding)
        ]
        result = self.model.predict({"images": images, "prompt": obs.instruction, "state": state})
        actions = np.asarray(result["action"])
        if actions.ndim == 1:
            actions = actions[None]
        return decode_actions({"action": actions}, self.cfg, profile)
