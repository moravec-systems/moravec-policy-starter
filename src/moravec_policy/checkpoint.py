"""Optional helpers for examples that qualify a local checkpoint and binding file."""

import json
from pathlib import Path

from .provenance import tree_hash


def qualified_checkpoint(manifest, path):
    root = Path(path).resolve().parent
    cfg = json.loads((root / manifest.config["binding_config"]).read_text())
    if cfg.get("status") != "qualified":
        raise ValueError(
            "checkpoint/embodiment bindings require qualification; no placeholder inference"
        )
    checkpoint = (root / manifest.config["checkpoint_path"]).resolve()
    if (
        not manifest.model.checkpoint_sha256
        or tree_hash(checkpoint) != manifest.model.checkpoint_sha256
    ):
        raise ValueError("checkpoint tree hash missing or mismatched")
    return cfg, checkpoint


def validate_binding_profile(cfg, obs):
    if (obs.profile_id, obs.profile_revision) != (cfg["profile_id"], cfg["profile_revision"]):
        raise ValueError("checkpoint bindings do not match robot profile")
