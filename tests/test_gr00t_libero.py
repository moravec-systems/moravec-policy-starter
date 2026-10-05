from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moravec_policy.loader import load_entrypoint

Model = load_entrypoint(
    "policy.py:Model",
    Path(__file__).resolve().parents[1] / "policies/GR00T_N17/libero/policy.json",
)


def setup(native):
    observed = []
    p = Model.__new__(Model)
    p.pending = deque()
    p.cfg = {"profile_id": "libero-panda", "profile_revision": "test"}
    p.model = SimpleNamespace(
        reset=lambda: None, get_action=lambda x: (observed.append(x) or native, {})
    )
    cam = SimpleNamespace(data=bytes(256 * 256 * 3), shape=(256, 256, 3))
    obs = SimpleNamespace(
        profile_id="libero-panda",
        profile_revision="test",
        instruction="pick up bowl",
        tcp_poses={
            "arm": SimpleNamespace(
                position_m=[0.1, 0.2, 0.3],
                orientation_wxyz=Rotation.from_rotvec([0.2, -0.1, 0.3]).as_quat(scalar_first=True),
            )
        },
        gripper_joint_positions={"gripper": [0.04, -0.04]},
        cameras={"overview": cam, "wrist": cam},
    )
    return p, obs, SimpleNamespace(id="libero-panda", control_period_s=0.05), observed


def infer(native):
    p, obs, profile, observed = setup(native)
    chunk = p._infer(obs, profile)
    return chunk, observed[0]


def output():
    return {
        k: np.full((1, 16, 1), v)
        for k, v in zip(
            ["x", "y", "z", "roll", "pitch", "yaw", "gripper"], [2, -0.4, 0.2, 0.3, -0.2, 0.1, 0]
        )
    }


def test_normalized_commands_become_physical_units():
    chunk, native = infer(output())
    target = chunk.steps[0].tcp_targets["arm"]
    np.testing.assert_allclose(target.pose.position_m, [0.05, -0.02, 0.01])
    np.testing.assert_allclose(
        Rotation.from_quat(target.pose.orientation_wxyz, scalar_first=True).as_rotvec(),
        [0.15, -0.1, 0.05],
    )
    assert target.relative_to == "observation"
    assert chunk.steps[0].grippers == {"gripper": 0}
    assert len(chunk.steps) == 1 and chunk.execution_horizon == 1
    np.testing.assert_allclose(
        [native["state"][k].item() for k in ["roll", "pitch", "yaw"]], [0.2, -0.1, 0.3]
    )
    np.testing.assert_allclose(native["state"]["gripper"], [[[0.04, -0.04]]])
    assert native["video"]["image"].shape == (1, 1, 256, 256, 3)


@pytest.mark.parametrize("bad", ["nan", "shape"])
def test_reject_invalid_native_output(bad):
    native = output()
    native["x"] = np.full((1, 16, 1), np.nan) if bad == "nan" else np.zeros((1, 16, 2))
    with pytest.raises(ValueError, match="unexpected native action"):
        infer(native)


def test_buffered_predictions_replan_after_eight_steps_and_reset():
    p, obs, profile, calls = setup(output())
    for _ in range(8):
        assert p._infer(obs, profile).execution_horizon == 1
    assert len(calls) == 1
    p._infer(obs, profile)
    assert len(calls) == 2
    p.reset()
    assert not p.pending
    p._infer(obs, profile)
    assert len(calls) == 3
