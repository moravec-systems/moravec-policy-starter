"""Multi-arm alignment and rejection at the public action boundary."""

import numpy as np
import pytest
from pydantic import ValidationError

from moravec_policy.action_adapter import decode_actions
from moravec_policy.contracts import ActionChunk, Decision, Pose
from moravec_policy.geometry import resolve_pose_chunk
from moravec_policy.wire import pack, unpack


def pose(x):
    return Pose(frame="world", position_m=(x, 0, 0), orientation_wxyz=(1, 0, 0, 0))


def test_bimanual_native_chunk_preserves_alignment_and_grippers():
    cfg = {
        "action_semantics": "relative-ee-pose",
        "timestep_s": 0.02,
        "execution_horizon": 2,
        "pose_bindings": [],
    }
    streams = {}
    for arm, distance in [("left", 0.1), ("right", -0.2)]:
        cfg["pose_bindings"].append(
            {
                "arm": arm,
                "frame": "world",
                "translation_stream": arm,
                "rotation_stream": arm + "_rot",
                "rotation_format": "wxyz",
                "reference": "frame",
                "relative_to": "previous",
                "gripper_stream": arm + "_grip",
            }
        )
        streams[arm] = np.array([[distance, 0, 0]] * 2)
        streams[arm + "_rot"] = np.array([[1, 0, 0, 0]] * 2)
        streams[arm + "_grip"] = np.array([[0.2], [0.8]])
    action = decode_actions(streams, cfg, None)
    assert ActionChunk.model_validate(unpack(pack(action.model_dump()))) == action
    resolved = resolve_pose_chunk(action, {"left": pose(1), "right": pose(2)})
    assert resolved[-1]["left"].position_m[0] == pytest.approx(1.2)
    assert resolved[-1]["right"].position_m[0] == pytest.approx(1.6)
    assert action.steps[1].grippers == {"left": 0.8, "right": 0.8}
    streams["right"] = streams["right"][:1]
    streams["right_rot"] = streams["right_rot"][:1]
    streams["right_grip"] = streams["right_grip"][:1]
    with pytest.raises(ValueError, match="horizons"):
        decode_actions(streams, cfg, None)


def test_omitted_arm_holds_initial_pose():
    chunk = ActionChunk(
        timestep_s=0.02, execution_horizon=1, steps=[{"tcp_targets": {"left": {"pose": pose(3)}}}]
    )
    assert resolve_pose_chunk(chunk, {"left": pose(1), "right": pose(2)}) == [
        {"left": pose(3), "right": pose(2)}
    ]


@pytest.mark.parametrize(
    "steps",
    [
        [{"joint_positions": {"left": 0}, "tcp_targets": {"right": {"pose": pose(0)}}}],
        [{"joint_positions": {"left": 0}}, {"joint_positions": {"right": 0}}],
        [{"grippers": {"left": 0}}, {"grippers": {}}],
        [{}],
        [{"joint_positions": {"left": float("nan")}}],
        [
            {"tcp_targets": {"left": {"pose": pose(0)}}},
            {"tcp_targets": {"left": {"pose": pose(1), "representation": "relative"}}},
        ],
    ],
)
def test_ambiguous_or_invalid_chunks_rejected(steps):
    with pytest.raises(ValueError):
        ActionChunk(timestep_s=0.02, execution_horizon=1, steps=steps)


def test_old_coordinated_kind_is_not_accepted():
    with pytest.raises(ValidationError):
        Decision(
            episode_id="e",
            observation_sequence=0,
            decision_id="d",
            valid_until_station_time_s=1,
            evidence_kind="contract-fixture",
            action={"kind": "coordinated", "tracks": []},
        )


def test_joint_channels_keep_both_grippers_on_shared_timeline():
    cfg = {
        "action_semantics": "absolute-joint-position",
        "timestep_s": 0.02,
        "execution_horizon": 1,
        "action_bindings": {"joints": ["left", "right"]},
        "gripper_bindings": {"lg": "left", "rg": "right"},
    }
    streams = {"joints": np.array([[0.1, -0.2]]), "lg": np.array([[0.3]]), "rg": np.array([[0.8]])}
    chunk = decode_actions(streams, cfg, None)
    assert chunk.steps[0].joint_positions == {"left": 0.1, "right": -0.2}
    assert chunk.steps[0].grippers == {"left": 0.3, "right": 0.8}
    streams["rg"] = np.array([[0.8], [0.9]])
    with pytest.raises(ValueError, match="horizons"):
        decode_actions(streams, cfg, None)


def test_example_matches_current_contract():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples/bimanual-action.json"
    chunk = ActionChunk.model_validate_json(example.read_text())
    assert set(chunk.steps[0].tcp_targets) == {"left", "right"}
