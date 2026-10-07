"""Moravec adapter for NVIDIA GR00T N1.7 in the checkpoint's native I/O format.

Observations carry the checkpoint's own `video`, `state`, and `language`
dictionaries; actions come back as one dictionary per step with the
checkpoint's native action names and units. The evaluator maps its cameras,
state, and controllers to that schema.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from gr00t.policy import Gr00tPolicy

from moravec_policy.model_template import ModelTemplate
from moravec_policy.utils.paths import resolve_policy_path

_POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_COSMOS_MODEL = "nvidia/Cosmos-Reason2-2B"


def _resolve_cosmos_model(model_cfg: dict[str, Any]) -> str:
    """A local Cosmos-Reason2 directory if one exists at the configured path, else the Hugging Face id."""
    value = str(model_cfg.get("cosmos_model_path") or DEFAULT_COSMOS_MODEL)
    local = resolve_policy_path(_POLICY_DIR, value)
    return str(local) if (local / "config.json").is_file() else value


@contextmanager
def _override_processor_cosmos_model(checkpoint_dir: Path, cosmos_model: str) -> Iterator[None]:
    """Point processor_config.json at the configured Cosmos model while the checkpoint loads."""
    config_path = checkpoint_dir / "processor_config.json"
    if not config_path.is_file():
        yield
        return

    data = json.loads(config_path.read_text(encoding="utf-8"))
    processor_kwargs = data.setdefault("processor_kwargs", {})
    previous = processor_kwargs.get("model_name")
    if previous == cosmos_model:
        yield
        return

    processor_kwargs["model_name"] = cosmos_model
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        yield
    finally:
        if previous is not None:
            processor_kwargs["model_name"] = previous
        else:
            processor_kwargs.pop("model_name", None)
        config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        checkpoint_dir = resolve_policy_path(_POLICY_DIR, model_cfg["model_dir"])
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(f"No GR00T checkpoint (config.json) in {checkpoint_dir}")
        embodiment_tag = model_cfg.get("embodiment_tag", "NEW_EMBODIMENT")
        cosmos_model = _resolve_cosmos_model(model_cfg)

        with _override_processor_cosmos_model(checkpoint_dir, cosmos_model):
            self.policy = Gr00tPolicy(
                model_path=str(checkpoint_dir),
                embodiment_tag=embodiment_tag,
                device=model_cfg.get("device", "cuda"),
                strict=True,
            )
        self.model = self.policy
        self.action_horizon = len(self.policy.modality_configs["action"].delta_indices)
        print(f"[GR00T_N17] loaded {checkpoint_dir} (embodiment {embodiment_tag}, cosmos {cosmos_model}, "
              f"horizon {self.action_horizon})")
        self.reset()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._order = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._obs = dict(zip(self._order, obs_list))

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=[self._order[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        indices = self._order if env_idx_list is None else env_idx_list
        missing = [index for index in indices if index not in self._obs]
        if missing:
            raise ValueError(f"No observation for env_idx {missing}; call update_obs first")
        chunks = []
        for index in indices:
            action, _ = self.policy.get_action(self._obs[index], **kwargs)
            arrays = {key: np.asarray(value)[0] for key, value in action.items()}
            if any(value.shape[0] != self.action_horizon or not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Invalid native action chunk")
            chunks.append([{key: value[step] for key, value in arrays.items()} for step in range(self.action_horizon)])
        return chunks

    def reset(self):
        self._obs: dict[int, Any] = {}
        self._order: list[int] = [0]
        if hasattr(self, "policy"):
            self.policy.reset()
