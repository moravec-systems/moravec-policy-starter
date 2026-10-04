"""Checkpoint-specific output conversion. All fields come from verified bindings."""

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import CoordinatedAction, JointTrajectory, PoseTrajectory


def decode_actions(streams, cfg, profile):
    semantics = cfg["action_semantics"]
    if semantics == "absolute-joint-position":
        bindings = cfg["action_bindings"]
        lengths = {len(streams[k]) for k in bindings}
        if len(lengths) != 1:
            raise ValueError("inconsistent action horizons")
        points = []
        for i in range(next(iter(lengths))):
            positions = {}
            for key, names in bindings.items():
                values = streams[key][i]
                if len(values) != len(names) or set(positions).intersection(names):
                    raise ValueError("invalid joint bindings")
                positions.update(zip(names, values.tolist()))
            points.append(
                {"time_from_start_s": (i + 1) * cfg["timestep_s"], "positions": positions}
            )
        return JointTrajectory(
            kind="joint-trajectory", points=points, execution_horizon=cfg["execution_horizon"]
        )
    if semantics not in {"absolute-ee-pose", "relative-ee-pose"}:
        raise ValueError("unsupported checkpoint action semantics")
    if "pose_bindings" in cfg:
        tracks = []
        for binding in cfg["pose_bindings"]:
            single = {k: v for k, v in cfg.items() if k != "pose_bindings"}
            single["pose_binding"] = binding
            tracks.append(decode_actions(streams, single, profile))
        return CoordinatedAction(kind="coordinated", tracks=tracks)
    binding = cfg["pose_binding"]
    translations = np.asarray(streams[binding["translation_stream"]])
    rotations = np.asarray(streams[binding["rotation_stream"]])
    if translations.ndim != 2 or translations.shape[1] != 3 or len(rotations) != len(translations):
        raise ValueError("invalid pose stream shapes")
    rotations = np.stack([rotation_wxyz(value, binding["rotation_format"]) for value in rotations])
    grippers = streams[binding["gripper_stream"]] if binding.get("gripper_stream") else None
    if grippers is not None and len(grippers) != len(translations):
        raise ValueError("gripper horizon mismatch")
    points = []
    for i, (xyz, quat) in enumerate(zip(translations, rotations)):
        points.append(
            {
                "pose": {"frame": binding["frame"], "position_m": xyz, "orientation_wxyz": quat},
                "gripper_open_fraction": None
                if grippers is None
                else float(np.asarray(grippers[i]).reshape(1)[0]),
            }
        )
    return PoseTrajectory(
        kind="pose-trajectory",
        arm=binding["arm"],
        representation="relative" if semantics.startswith("relative") else "absolute",
        reference=binding["reference"],
        relative_to=binding["relative_to"],
        timestep_s=cfg["timestep_s"],
        execution_horizon=cfg["execution_horizon"],
        points=points,
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
