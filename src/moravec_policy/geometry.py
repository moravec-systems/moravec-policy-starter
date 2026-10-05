"""WXYZ pose composition. Relative translation is meters, rotation is a unit quaternion."""

import numpy as np

from .contracts import Pose


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array(
        [
            w * v - x * i - y * j - z * k,
            w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i,
            w * k + x * j - y * i + z * v,
        ]
    )


def compose_pose(base, delta, reference):
    if base.frame != delta.frame:
        raise ValueError("pose frames differ; calibrated transform required")
    b, d = np.array(base.orientation_wxyz), np.array(delta.orientation_wxyz)
    translation = np.array(delta.position_m)
    if reference == "tool":
        translation = multiply(multiply(b, [0, *translation]), b * [1, -1, -1, -1])[1:]
        q = multiply(b, d)
    elif reference == "frame":
        q = multiply(d, b)
    else:
        raise ValueError("unknown relative reference")
    return Pose(
        frame=base.frame,
        position_m=np.array(base.position_m) + translation,
        orientation_wxyz=q / np.linalg.norm(q),
    )


def resolve_pose_chunk(chunk, initial):
    """Resolve all named arms on the shared timeline, holding omitted arms at start."""
    poses = []
    previous = dict(initial)
    for step in chunk.steps[: chunk.execution_horizon]:
        resolved = dict(initial)
        for arm, target in step.tcp_targets.items():
            if target.representation == "absolute":
                resolved[arm] = target.pose
            else:
                base = initial[arm] if target.relative_to == "observation" else previous[arm]
                resolved[arm] = compose_pose(base, target.pose, target.reference)
        poses.append(resolved)
        previous = resolved
    return poses
