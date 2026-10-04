"""Moravec v0.3 contracts, shared by simulation and physical station clients."""

from __future__ import annotations

import itertools
import math
from typing import Annotated, Literal
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, ser_json_bytes="base64", val_json_bytes="base64"
    )


class ModelIdentity(Contract):
    name: Slug
    provider_model_id: str
    artifact_uri: str | None = None
    artifact_revision: str | None = None
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ModelEndpoint(Contract):
    kind: Literal["http-llm"]
    url: str
    api: Literal["openai-responses"]
    token_env: str
    timeout_s: float = Field(gt=0, le=120)
    max_calls: int = Field(gt=0, le=200)
    max_output_tokens: int = Field(gt=0, le=32768)
    reasoning_effort: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def safe_url(self):
        p = urlparse(self.url)
        if p.username or p.password or p.query or p.fragment:
            raise ValueError("endpoint URL must not contain credentials/query/fragment")
        if p.scheme not in {"http", "https"} or not p.hostname:
            raise ValueError("model endpoint must be HTTP(S)")
        return self


class LocalModel(Contract):
    kind: Literal["local-model"]
    driver: Literal["gr00t", "openwam"]
    checkpoint_path: str
    binding_config: str


class FixtureBackend(Contract):
    kind: Literal["contract-fixture"]
    output: Literal["joint-trajectory", "relative-pose"]


class PolicyManifest(Contract):
    schema_version: Literal["moravec.policy/v0.3"]
    model: ModelIdentity
    family: Literal["llm", "vla", "world-action", "fixture"]
    harness_version: str
    required_camera_channels: dict[
        str, list[Literal["rgb", "depth", "intrinsics", "extrinsics"]]
    ] = Field(default_factory=dict)
    runtime_profile: Literal["isolated", "api-connected"]
    backend: Annotated[ModelEndpoint | LocalModel | FixtureBackend, Field(discriminator="kind")]

    @model_validator(mode="after")
    def compatible(self):
        if isinstance(self.backend, ModelEndpoint) and self.family != "llm":
            raise ValueError("HTTP LLM backend requires llm family")
        if isinstance(self.backend, FixtureBackend) and self.family != "fixture":
            raise ValueError("fixtures cannot claim real-model identity")
        if isinstance(self.backend, LocalModel):
            expected = "vla" if self.backend.driver == "gr00t" else "world-action"
            if self.family != expected:
                raise ValueError("model driver/family mismatch")
        return self


class Joint(Contract):
    name: str
    unit: Literal["rad", "m"]
    lower: float
    upper: float

    @model_validator(mode="after")
    def bounds(self):
        if self.lower >= self.upper:
            raise ValueError("joint range must increase")
        return self


class RobotProfile(Contract):
    id: Slug
    revision: str
    joints: list[Joint] = Field(min_length=1)
    frames: list[str] = Field(min_length=1)
    cameras: list[str] = Field(min_length=1)
    control_period_s: float = Field(gt=0, le=1)
    max_observation_age_s: float = Field(gt=0, le=10)
    max_camera_skew_s: float = Field(ge=0, le=1)
    max_action_duration_s: float = Field(gt=0, le=30)

    @model_validator(mode="after")
    def unique_names(self):
        for names in ([j.name for j in self.joints], self.frames, self.cameras):
            if len(set(names)) != len(names):
                raise ValueError("duplicate profile names")
        return self


class Pose(Contract):
    frame: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]

    @model_validator(mode="after")
    def quaternion(self):
        if not math.isclose(sum(x * x for x in self.orientation_wxyz), 1.0, abs_tol=1e-3):
            raise ValueError("quaternion must be unit WXYZ")
        return self


class Depth(Contract):
    encoding: Literal["float32-le-meters"] = "float32-le-meters"
    shape: tuple[int, int]
    data: bytes = Field(max_length=16_777_216)

    @model_validator(mode="after")
    def valid_depth(self):
        h, w = self.shape
        if not (1 <= h <= 2048 and 1 <= w <= 2048) or len(self.data) != h * w * 4:
            raise ValueError("invalid depth shape/bytes")
        values = np.frombuffer(self.data, dtype="<f4")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("depth must be finite nonnegative meters; zero means invalid")
        return self


class Camera(Contract):
    encoding: Literal["rgb8"] = "rgb8"
    shape: tuple[int, int, Literal[3]]
    data: bytes = Field(max_length=16_777_216)
    captured_at_s: float = Field(ge=0)

    depth: Depth | None = None
    intrinsic_matrix: (
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | None
    ) = None
    extrinsics_matrix: (
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ]
        | None
    ) = None
    extrinsics_frame: str | None = None

    @model_validator(mode="after")
    def image(self):
        h, w, _ = self.shape
        if not (1 <= h <= 2048 and 1 <= w <= 2048):
            raise ValueError("invalid image dimensions")
        if len(self.data) != h * w * 3:
            raise ValueError("RGB byte count does not match shape")
        if self.depth and self.depth.shape != self.shape[:2]:
            raise ValueError("depth must be registered to RGB dimensions")
        if self.intrinsic_matrix is not None:
            k = np.asarray(self.intrinsic_matrix)
            if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
                raise ValueError("invalid camera intrinsics")
        if (self.extrinsics_matrix is None) != (self.extrinsics_frame is None):
            raise ValueError("camera extrinsics require a reference frame")
        if self.extrinsics_matrix is not None:
            t = np.asarray(self.extrinsics_matrix)
            r = t[:3, :3]
            if (
                not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1)
            ):
                raise ValueError("camera extrinsics must be a rigid camera-to-frame transform")
        return self


class Feedback(Contract):
    decision_id: str
    status: Literal["executed", "rejected", "interrupted"]
    reason: str
    executed_steps: int = Field(ge=0)


class Observation(Contract):
    schema_version: Literal["moravec.observation/v0.3"] = "moravec.observation/v0.3"
    episode_id: str
    sequence: int = Field(ge=0)
    station_time_s: float = Field(ge=0)
    captured_at_s: float = Field(ge=0)
    profile_id: str
    profile_revision: str
    instruction: str
    cameras: dict[str, Camera]
    joint_positions: dict[str, float]
    tcp_poses: dict[str, Pose]
    gripper_open_fraction: dict[str, Annotated[float, Field(ge=0, le=1)]]
    feedback: Feedback | None = None
    ee_poses: dict[str, Pose] = Field(default_factory=dict)
    joint_velocities: dict[str, float] = Field(default_factory=dict)
    gripper_joint_positions: dict[str, list[float]] = Field(default_factory=dict)
    base_pose: Pose | None = None
    base_twist: tuple[float, float, float, float, float, float] | None = None


class JointPoint(Contract):
    time_from_start_s: float = Field(gt=0)
    positions: dict[str, float]
    grippers: dict[str, Annotated[float, Field(ge=0, le=1)]] = Field(default_factory=dict)


class JointTrajectory(Contract):
    kind: Literal["joint-trajectory"]
    points: list[JointPoint] = Field(min_length=1, max_length=256)
    execution_horizon: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def increasing_time(self):
        if self.execution_horizon is not None and self.execution_horizon > len(self.points):
            raise ValueError("execution horizon exceeds prediction horizon")
        times = [p.time_from_start_s for p in self.points]
        if any(b <= a for a, b in itertools.pairwise(times)):
            raise ValueError("trajectory timestamps must strictly increase")
        return self


class CartesianGoal(Contract):
    kind: Literal["cartesian-goal"]
    arm: str
    target: Pose
    duration_s: float = Field(gt=0, le=30)
    gripper_open_fraction: float | None = Field(default=None, ge=0, le=1)


class PosePoint(Contract):
    pose: Pose
    gripper_open_fraction: float | None = Field(default=None, ge=0, le=1)


class PoseTrajectory(Contract):
    kind: Literal["pose-trajectory"]
    arm: str
    representation: Literal["absolute", "relative"]
    reference: Literal["frame", "tool"] = "frame"
    relative_to: Literal["observation", "previous"] = "observation"
    timestep_s: float = Field(gt=0, le=1)
    execution_horizon: int = Field(gt=0)
    points: list[PosePoint] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def horizon(self):
        if self.execution_horizon > len(self.points):
            raise ValueError("execution horizon exceeds prediction horizon")
        if len({p.pose.frame for p in self.points}) != 1:
            raise ValueError("pose chunk must use one reference frame")
        if self.representation == "absolute" and (
            self.reference != "frame" or self.relative_to != "observation"
        ):
            raise ValueError("absolute poses require frame reference")
        return self


class GripperCommand(Contract):
    kind: Literal["gripper-command"]
    gripper: str
    open_fraction: float = Field(ge=0, le=1)
    duration_s: float = Field(gt=0, le=30)


class Stop(Contract):
    kind: Literal["stop"]
    reason: str


class CoordinatedAction(Contract):
    kind: Literal["coordinated"]
    # All tracks start at the same station time; each track declares its own timestamps.
    tracks: list[
        Annotated[JointTrajectory | PoseTrajectory | GripperCommand, Field(discriminator="kind")]
    ] = Field(min_length=1, max_length=8)


class Decision(Contract):
    schema_version: Literal["moravec.decision/v0.3"] = "moravec.decision/v0.3"
    episode_id: str
    observation_sequence: int
    decision_id: str
    valid_until_station_time_s: float
    evidence_kind: Literal["model-inference", "contract-fixture"]
    action: Annotated[
        JointTrajectory | PoseTrajectory | GripperCommand | CoordinatedAction | Stop,
        Field(discriminator="kind"),
    ]


class EpisodeStart(Contract):
    episode_id: str = Field(min_length=1)
    profile: RobotProfile


class EpisodeEnd(Contract):
    episode_id: str


class DeploymentManifest(Contract):
    schema_version: Literal["moravec.deployment/v0.3"]
    image: str = Field(pattern=r"^.+@sha256:[0-9a-f]{64}$")
    registry_pull_secret_ref: str
    network_profile: Literal["isolated", "api-connected"]
    model_endpoint_allowlist: list[str] = []
    results_destination: str
    results_write_secret_ref: str
    retention_days: int = Field(ge=0)
    publication_opt_in: bool = False


class DeliveryFile(Contract):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DeliveryManifest(Contract):
    campaign_id: str
    policy_image_digest: str
    station_config_version: str
    task_version: str
    files: list[DeliveryFile]
