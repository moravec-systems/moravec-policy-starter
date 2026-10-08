"""Moravec adapter for Physical Intelligence's π0.5, using upstream openpi."""
from pathlib import Path
from typing import Any

import numpy as np

from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.training import config as openpi_config

from moravec_policy.model_template import ModelTemplate
from moravec_policy.utils.paths import resolve_policy_path
from moravec_policy.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.reset()
        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        # openpi's ALOHA recipe: three cameras (cam_high, cam_left_wrist,
        # cam_right_wrist), 14-D state of 6 joints + 1 gripper per arm, and
        # delta joint actions. Only inference fields are needed.
        model_root = resolve_policy_path(_POLICY_DIR, model_cfg["model_path"])
        if not (model_root / "params").is_dir():
            raise FileNotFoundError(f"No openpi checkpoint (params/) in {model_root}")
        train_config = openpi_config.TrainConfig(
            name="pi05_aloha_moravec",
            model=pi0_config.Pi0Config(pi05=True),
            data=openpi_config.LeRobotAlohaDataConfig(
                # True only for checkpoints trained in π's internal joint/gripper space.
                adapt_to_pi=model_cfg.get("adapt_to_pi", False),
                # Normalization stats ship with the checkpoint: <model_root>/assets/<asset_id>.
                assets=openpi_config.AssetsConfig(
                    assets_dir=str(model_root / "assets"), asset_id=model_cfg["asset_id"]
                ),
            ),
        )
        return policy_config.create_trained_policy(train_config, model_root)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._order = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._obs = {index: encode_obs(obs, self.action_type, self.robot_action_dim_info)
                     for index, obs in zip(self._order, obs_list)}

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=[self._order[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = self._order if env_idx_list is None else env_idx_list
        missing = [index for index in env_idx_list if index not in self._obs]
        if missing:
            raise ValueError(f"No observation for env_idx {missing}; call update_obs first")
        action_list = []
        # openpi infers one observation at a time.
        for env_idx in env_idx_list:
            actions = self.policy.infer(self._obs[env_idx], **kwargs)["actions"]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self._obs: dict[int, dict[str, Any]] = {}
        self._order: list[int] = [0]


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
