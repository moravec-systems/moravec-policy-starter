"""Thin adapter for Ai2's MolmoAct2 BimanualYAM Transformers checkpoint.

Source revisions and attribution are recorded in README.md and THIRD_PARTY.md.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from moravec_policy.model_template import ModelTemplate
from moravec_policy.utils.process_data import (
    get_robot_action_dim_info, pack_robot_state, unpack_robot_state,
)

NORM_TAG = "yam_dual_molmoact2"
CAMERAS = (
    ("top", "cam_high", "cam_head", "top_camera"),
    ("left", "cam_left_wrist", "left_camera"),
    ("right", "cam_right_wrist", "right_camera"),
)


def encode_images(observation):
    """Preserve the checkpoint's top, left-wrist, right-wrist RGB order."""
    images = []
    for names in CAMERAS:
        vision = observation["vision"]
        name = next((name for name in names if name in vision), None)
        if name is None:
            raise KeyError(f"Missing camera; expected one of {names}")
        value = vision[name]
        if isinstance(value, dict):
            value = value.get("color", value.get("rgb"))
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Camera {name} must contain HWC uint8 RGB pixels")
        images.append(Image.fromarray(array))
    return images


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if model_cfg.get("action_type", "joint") != "joint":
            raise ValueError("MolmoAct2 BimanualYAM requires absolute joint actions")
        if model_cfg.get("env_cfg_type") != "yam":
            raise ValueError("This example supports only the BimanualYAM checkpoint and yam environment")
        self.robot_action_dim_info = get_robot_action_dim_info("yam")
        if any(self.robot_action_dim_info.get(key) != value for key, value in
               {"arm_dim": [6, 6], "ee_dim": [1, 1]}.items()):
            raise ValueError("YAM requires six arm joints and one gripper per side")
        self.num_steps = int(model_cfg.get("num_steps", 10))
        self.actions_per_chunk = int(model_cfg.get("actions_per_chunk", 30))
        if self.num_steps < 1 or not 1 <= self.actions_per_chunk <= 30:
            raise ValueError("num_steps must be positive; actions_per_chunk must be in [1, 30]")
        self.device = torch.device(model_cfg.get("device", "cuda"))
        dtype = model_cfg.get("dtype", "bfloat16")
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        self.dtype = getattr(torch, dtype)
        if self.device.type != "cuda" and self.dtype != torch.float32:
            raise ValueError("Use float32 on CPU or MPS; bfloat16 requires CUDA autocast")
        checkpoint = Path(model_cfg["model_path"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parent / checkpoint
        if not (checkpoint / "norm_stats.json").is_file():
            raise FileNotFoundError(f"Download the complete pinned checkpoint first: {checkpoint}")
        load_options = {"trust_remote_code": True, "local_files_only": True}
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, extra_special_tokens={}, **load_options,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, torch_dtype=self.dtype, **load_options,
        ).to(self.device).eval()
        stats = self.model._get_robot_stats()
        metadata = stats.get_metadata(NORM_TAG)
        if (stats.get_action_dim(NORM_TAG) != 14 or stats.get_state_dim(NORM_TAG) != 14
                or stats.get_action_horizon(NORM_TAG) != 30
                or metadata.get("control_mode") != "absolute joint pose"):
            raise ValueError("Checkpoint must implement the BimanualYAM 14-D absolute-joint contract")
        self.reset()

    def update_obs(self, obs):
        self.latest_observations = {0: obs}

    def update_obs_batch(self, obs_list):
        indices = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate environment indices")
        self.latest_observations = dict(zip(indices, obs_list))

    def _predict(self, observation):
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A nonempty instruction is required")
        images = encode_images(observation)
        state = pack_robot_state(observation, "joint", self.robot_action_dim_info,
                                 source_type="obs").astype(np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("State must contain 14 finite values")
        # CUDA autocast handles float32 processor images with bf16 model weights.
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype,
            enabled=self.device.type == "cuda" and self.dtype == torch.bfloat16,
        ):
            output = self.model.predict_action(
                processor=self.processor, images=images, task=instruction, state=state,
                norm_tag=NORM_TAG, inference_action_mode="continuous",
                enable_depth_reasoning=False, num_steps=self.num_steps,
                n_action_steps=self.actions_per_chunk, normalize_language=True,
                enable_cuda_graph=False,
            )
        actions = output.actions
        if torch.is_tensor(actions):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (self.actions_per_chunk, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid action chunk: {actions.shape}")
        return unpack_robot_state(actions, "joint", self.robot_action_dim_info, source_type="obs")

    def get_action(self, **kwargs):
        if 0 not in self.latest_observations:
            raise RuntimeError("Call update_obs before get_action")
        return self._predict(self.latest_observations[0])

    def get_action_batch(self, env_idx_list=None, **kwargs):
        indices = list(self.latest_observations) if env_idx_list is None else env_idx_list
        missing = set(indices) - self.latest_observations.keys()
        if missing:
            raise KeyError(f"Missing observations for environments {sorted(missing)}")
        return [self._predict(self.latest_observations[index]) for index in indices]

    def reset(self):
        self.latest_observations = {}
