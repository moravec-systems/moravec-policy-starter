"""Explicit checkpoint state bindings; no inference of semantics from vector length."""

import numpy as np
from scipy.spatial.transform import Rotation


def state_vector(obs, binding):
    source = binding["source"]
    if source == "joint_positions":
        return [obs.joint_positions[name] for name in binding["names"]]
    if source == "gripper_open_fraction":
        return [obs.gripper_open_fraction[binding["name"]]]
    if source == "gripper_joint_positions":
        return obs.gripper_joint_positions[binding["name"]]
    if source not in {"tcp_poses", "ee_poses"}:
        raise ValueError("unsupported checkpoint state source")
    pose = getattr(obs, source)[binding["name"]]
    if pose.frame != binding["frame"]:
        raise ValueError("state pose needs calibrated frame conversion")
    rotation = Rotation.from_quat(pose.orientation_wxyz, scalar_first=True)
    fmt = binding["rotation_format"]
    if fmt == "wxyz":
        values = pose.orientation_wxyz
    elif fmt == "rotvec":
        values = rotation.as_rotvec()
    elif fmt == "euler-xyz":
        values = rotation.as_euler("xyz")
    elif fmt == "rotation6d-rows":
        values = rotation.as_matrix()[:2, :].reshape(-1)
    elif fmt == "rotation6d-columns":
        values = rotation.as_matrix()[:, :2].T.reshape(-1)
    else:
        raise ValueError("unsupported state rotation format")
    return np.concatenate([pose.position_m, values]).tolist()
