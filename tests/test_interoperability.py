import json
import math
from pathlib import Path

import numpy as np
import pytest

from moravec_policy.action_adapter import decode_actions
from moravec_policy.contracts import (
    Camera,
    Decision,
    Depth,
    Observation,
    Pose,
    PoseTrajectory,
    RobotProfile,
)
from moravec_policy.geometry import compose_pose, resolve_pose_chunk
from moravec_policy.validation import validate_channels, validate_decision
from moravec_policy.wire import (
    ObservationEnvelope,
    observation_from_wire,
    observation_to_wire,
    pack,
    unpack,
)

ROOT = Path(__file__).resolve().parents[1]


def observation():
    return Observation(
        episode_id="e",
        sequence=0,
        station_time_s=1,
        captured_at_s=1,
        profile_id="planar-two-joint",
        profile_revision="1",
        instruction="reach",
        cameras={"overview": Camera(shape=(1, 1, 3), data=b"\xff\x00\x00", captured_at_s=1)},
        joint_positions={"shoulder": 0, "elbow": 0},
        tcp_poses={"arm": Pose(frame="world", position_m=(1, 0, 0), orientation_wxyz=(1, 0, 0, 0))},
        gripper_open_fraction={"arm": 0.5},
    )


def profile():
    return RobotProfile.model_validate_json((ROOT / "sim/profile.json").read_text())


def test_binary_modalities_round_trip():
    obs = observation()
    camera = obs.cameras["overview"]
    camera.depth = Depth(shape=(1, 1), data=np.array([1.25], dtype="<f4").tobytes())
    camera.intrinsic_matrix = ((100.0, 0.0, 0.5), (0.0, 100.0, 0.5), (0.0, 0.0, 1.0))
    camera.extrinsics_matrix = tuple(tuple(float(x) for x in row) for row in np.eye(4))
    camera.extrinsics_frame = "world"
    wire = observation_to_wire(obs)
    assert set(wire) == {"metadata", "observation"}
    assert wire["observation"]["vision"]["overview"]["data"] == b"\xff\x00\x00"
    ObservationEnvelope.model_validate(wire)
    decoded = observation_from_wire(unpack(pack(wire)))
    assert decoded == obs
    validate_channels(decoded, {"overview": ["rgb", "depth", "intrinsics", "extrinsics"]})
    with pytest.raises(ValueError):
        validate_channels(observation(), {"overview": ["depth"]})


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_depth(value):
    with pytest.raises(ValueError):
        Depth(shape=(1, 1), data=np.array([value], dtype="<f4").tobytes())


def test_invalid_calibration():
    with pytest.raises(ValueError):
        Camera(
            shape=(1, 1, 3),
            data=b"123",
            captured_at_s=0,
            intrinsic_matrix=((0, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
    with pytest.raises(ValueError):
        Camera(shape=(1, 1, 3), data=b"123", captured_at_s=0, extrinsics_frame="world")


def test_rotated_tool_translation_and_composition():
    q = (math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    base = Pose(frame="world", position_m=(1, 0, 0), orientation_wxyz=q)
    delta = Pose(frame="world", position_m=(1, 0, 0), orientation_wxyz=q)
    tool = compose_pose(base, delta, "tool")
    frame = compose_pose(base, delta, "frame")
    assert tool.position_m == pytest.approx((1, 1, 0))
    assert frame.position_m == pytest.approx((2, 0, 0))
    assert tool.orientation_wxyz == pytest.approx((0, 0, 0, 1), abs=1e-8)


@pytest.mark.parametrize("relative_to,expected", [("observation", 2), ("previous", 3)])
def test_relative_chunk_origins(relative_to, expected):
    base = Pose(frame="world", position_m=(1, 0, 0), orientation_wxyz=(1, 0, 0, 0))
    delta = Pose(frame="world", position_m=(1, 0, 0), orientation_wxyz=(1, 0, 0, 0))
    chunk = PoseTrajectory(
        kind="pose-trajectory",
        arm="arm",
        representation="relative",
        relative_to=relative_to,
        timestep_s=0.1,
        execution_horizon=2,
        points=[{"pose": delta}] * 3,
    )
    assert resolve_pose_chunk(chunk, base)[-1].position_m[0] == expected
    assert len(resolve_pose_chunk(chunk, base)) == 2


def test_checkpoint_pose_mapping():
    cfg = {
        "action_semantics": "relative-ee-pose",
        "timestep_s": 0.02,
        "execution_horizon": 1,
        "pose_binding": {
            "arm": "arm",
            "frame": "world",
            "translation_stream": "xyz",
            "rotation_stream": "rot",
            "rotation_format": "wxyz",
            "reference": "tool",
            "relative_to": "observation",
        },
    }
    streams = {
        "xyz": np.array([[0.01, 0, 0], [0.02, 0, 0]]),
        "rot": np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
    }
    action = decode_actions(streams, cfg, profile())
    assert action.representation == "relative" and action.execution_horizon == 1
    assert action.points[1].pose.position_m == (0.02, 0, 0)
    decision = Decision(
        episode_id="e",
        observation_sequence=0,
        decision_id="d",
        valid_until_station_time_s=2,
        evidence_kind="contract-fixture",
        action=action,
    )
    validate_decision(decision, observation(), profile(), 1)
    cfg["pose_binding"]["rotation_format"] = "unknown"
    with pytest.raises(ValueError):
        decode_actions(streams, cfg, profile())


def test_metadata_cannot_override_payload():
    wire = observation_to_wire(observation())
    wire["metadata"]["joint_positions"] = {}
    with pytest.raises(ValueError):
        observation_from_wire(wire)


def test_batching_is_explicitly_unsupported():
    from moravec_policy.policies import FixturePolicy

    policy = FixturePolicy("joint-trajectory")
    policy.configure(profile())
    with pytest.raises(ValueError):
        policy.get_action()
    with pytest.raises(NotImplementedError):
        policy.update_obs_batch([])
    obs = observation()
    obs.instruction = json.dumps({"joint_target": {"shoulder": 0.1, "elbow": 0.2}})
    policy.update_obs(obs)
    assert policy.get_action().points[0].positions["elbow"] == 0.2
    policy.reset()
    with pytest.raises(ValueError):
        policy.get_action()


@pytest.mark.parametrize(
    "representation", ["wxyz", "rotvec", "euler-xyz", "rotation6d-rows", "rotation6d-columns"]
)
def test_rotation_adapter_roundtrip(representation):
    from moravec_policy.action_adapter import rotation_wxyz
    from moravec_policy.state_adapter import state_vector

    obs = observation()
    q = (math.cos(0.3), 0, math.sin(0.3), 0)
    obs.tcp_poses["arm"] = Pose(frame="world", position_m=(1, 2, 3), orientation_wxyz=q)
    values = state_vector(
        obs,
        {"source": "tcp_poses", "name": "arm", "frame": "world", "rotation_format": representation},
    )
    assert values[:3] == [1, 2, 3]
    actual = rotation_wxyz(values[3:], representation)
    assert abs(np.dot(actual, q)) == pytest.approx(1)


def test_gripper_and_coordinated_action_validation():
    from moravec_policy.contracts import CoordinatedAction, GripperCommand, JointTrajectory

    obs = observation()
    grip = GripperCommand(kind="gripper-command", gripper="arm", open_fraction=1, duration_s=0.1)
    joints = JointTrajectory(
        kind="joint-trajectory",
        points=[{"time_from_start_s": 0.1, "positions": {"shoulder": 0.1, "elbow": 0.2}}],
    )
    action = CoordinatedAction(kind="coordinated", tracks=[joints, grip])
    decision = Decision(
        episode_id="e",
        observation_sequence=0,
        decision_id="d",
        valid_until_station_time_s=2,
        evidence_kind="contract-fixture",
        action=action,
    )
    validate_decision(decision, obs, profile(), 1)
    action.tracks = [grip, grip]
    with pytest.raises(ValueError, match="overlapping"):
        validate_decision(decision, obs, profile(), 1)


def test_native_history_shapes_and_reset():
    from types import SimpleNamespace

    from moravec_policy.native import NativeModelPolicy

    seen = []

    class Model:
        def reset(self):
            pass

        def get_action(self, data):
            seen.append(data)
            return {"action": np.array([[[0.1, 0.2]]])}, {}

    policy = NativeModelPolicy.__new__(NativeModelPolicy)
    policy.model = Model()
    policy.history_length = 2
    policy.manifest = SimpleNamespace(backend=SimpleNamespace(driver="gr00t"))
    policy.modalities = {
        "video": SimpleNamespace(delta_indices=[-1, 0]),
        "state": SimpleNamespace(delta_indices=[-1, 0]),
    }
    policy.cfg = {
        "profile_id": "planar-two-joint",
        "profile_revision": "1",
        "observation_period_s": 0.02,
        "camera_bindings": {"front": "overview"},
        "state_bindings": {"arm": {"source": "joint_positions", "names": ["shoulder", "elbow"]}},
        "language_key": "task",
        "action_semantics": "absolute-joint-position",
        "action_bindings": {"action": ["shoulder", "elbow"]},
        "timestep_s": 0.02,
        "execution_horizon": 1,
    }
    policy.configure(profile())
    first = observation()
    policy.update_obs(first)
    with pytest.raises(ValueError, match="history"):
        policy.get_action()
    second = first.model_copy(deep=True)
    second.sequence = 1
    second.captured_at_s = 1.02
    second.joint_positions["elbow"] = 0.4
    policy.update_obs(second)
    policy.get_action()
    assert seen[0]["video"]["front"].shape == (1, 2, 1, 1, 3)
    assert seen[0]["state"]["arm"][0, :, 1].tolist() == pytest.approx([0, 0.4])
    policy.reset()
    assert len(policy.history) == 0
