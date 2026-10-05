"""Checkpoint-specific output conversion. All fields come from verified bindings."""

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import ActionChunk


def decode_actions(streams, cfg, profile):
    semantics = cfg["action_semantics"]
    steps = []
    if semantics == "absolute-joint-position":
        bindings = cfg["action_bindings"]
        gripper_bindings = cfg.get("gripper_bindings", {})
        lengths = {len(streams[k]) for k in [*bindings, *gripper_bindings]}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("inconsistent action horizons")
        for i in range(next(iter(lengths))):
            positions = {}
            for key, names in bindings.items():
                values = streams[key][i]
                if len(values) != len(names) or set(positions).intersection(names):
                    raise ValueError("invalid joint bindings")
                positions.update(zip(names, values.tolist()))
            grippers = {}
            for key, name in gripper_bindings.items():
                if name in grippers:
                    raise ValueError("duplicate gripper binding")
                grippers[name] = float(np.asarray(streams[key][i]).reshape(1)[0])
            steps.append({"joint_positions": positions, "grippers": grippers})
    elif semantics in {"absolute-ee-pose", "relative-ee-pose"}:
        bindings = cfg.get("pose_bindings") or [cfg["pose_binding"]]
        arms = set()
        gripper_names = set()
        for binding in bindings:
            arm = binding["arm"]
            if arm in arms:
                raise ValueError("duplicate arm binding")
            arms.add(arm)
            translations = np.asarray(streams[binding["translation_stream"]])
            rotations = np.asarray(streams[binding["rotation_stream"]])
            if (
                len(translations) == 0
                or translations.ndim != 2
                or translations.shape[1] != 3
                or len(rotations) != len(translations)
            ):
                raise ValueError("invalid pose stream shapes")
            rotations = [rotation_wxyz(value, binding["rotation_format"]) for value in rotations]
            grippers = streams[binding["gripper_stream"]] if binding.get("gripper_stream") else None
            if grippers is not None and len(grippers) != len(translations):
                raise ValueError("gripper horizon mismatch")
            gripper = binding.get("gripper", arm)
            if grippers is not None:
                if gripper in gripper_names:
                    raise ValueError("duplicate gripper binding")
                gripper_names.add(gripper)
            if not steps:
                steps = [{"tcp_targets": {}, "grippers": {}} for _ in translations]
            if len(steps) != len(translations):
                raise ValueError("inconsistent action horizons")
            for i, (xyz, quat) in enumerate(zip(translations, rotations)):
                steps[i]["tcp_targets"][arm] = {
                    "pose": {
                        "frame": binding["frame"],
                        "position_m": xyz,
                        "orientation_wxyz": quat,
                    },
                    "representation": "relative"
                    if semantics.startswith("relative")
                    else "absolute",
                    "reference": binding["reference"],
                    "relative_to": binding["relative_to"],
                }
                if grippers is not None:
                    steps[i]["grippers"][gripper] = float(np.asarray(grippers[i]).reshape(1)[0])
    else:
        raise ValueError("unsupported checkpoint action semantics")
    return ActionChunk(
        timestep_s=cfg["timestep_s"], execution_horizon=cfg["execution_horizon"], steps=steps
    )


def rotation_wxyz(value, representation):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite rotation")
    if representation == "wxyz":
        if value.shape != (4,) or not np.isclose(np.linalg.norm(value), 1.0, atol=1e-3):
            raise ValueError("rotation must be unit WXYZ")
        return value
    if representation in {"rotvec", "euler-xyz"}:
        if value.shape != (3,):
            raise ValueError("rotation needs three radians")
        rotation = (
            Rotation.from_rotvec(value)
            if representation == "rotvec"
            else Rotation.from_euler("xyz", value)
        )
    elif representation in {"rotation6d-rows", "rotation6d-columns"}:
        if value.shape != (6,):
            raise ValueError("6D rotation needs six values")
        a, b = value[:3], value[3:]
        if np.linalg.norm(a) < 1e-8:
            raise ValueError("degenerate rotation")
        a = a / np.linalg.norm(a)
        b = b - np.dot(a, b) * a
        if np.linalg.norm(b) < 1e-8:
            raise ValueError("degenerate rotation")
        b = b / np.linalg.norm(b)
        matrix = np.stack([a, b, np.cross(a, b)])
        rotation = Rotation.from_matrix(matrix if representation.endswith("rows") else matrix.T)
    else:
        raise ValueError("unsupported rotation representation")
    return rotation.as_quat(scalar_first=True)
