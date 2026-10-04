"""Moravec v0.2 contracts, shared by simulation and physical station clients."""

from __future__ import annotations

import base64
import itertools
import math
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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
    output: Literal["joint-trajectory", "cartesian-goal"]


class PolicyManifest(Contract):
    schema_version: Literal["moravec.policy/v0.2"]
    model: ModelIdentity
    family: Literal["llm", "vla", "world-action", "fixture"]
    harness_version: str
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


class Camera(Contract):
    encoding: Literal["rgb8-base64"] = "rgb8-base64"
    shape: tuple[int, int, Literal[3]]
    data: str = Field(max_length=16_777_216)
    captured_at_s: float = Field(ge=0)

    @model_validator(mode="after")
    def image(self):
        h, w, _ = self.shape
        if not (1 <= h <= 2048 and 1 <= w <= 2048):
            raise ValueError("invalid image dimensions")
        if len(base64.b64decode(self.data, validate=True)) != h * w * 3:
            raise ValueError("RGB byte count does not match shape")
        return self


class Feedback(Contract):
    decision_id: str
    status: Literal["executed", "rejected", "interrupted"]
    reason: str
    executed_steps: int = Field(ge=0)


class Observation(Contract):
    schema_version: Literal["moravec.observation/v0.2"] = "moravec.observation/v0.2"
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


class JointPoint(Contract):
    time_from_start_s: float = Field(gt=0)
    positions: dict[str, float]


class JointTrajectory(Contract):
    kind: Literal["joint-trajectory"]
    points: list[JointPoint] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def increasing_time(self):
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


class Stop(Contract):
    kind: Literal["stop"]
    reason: str


class Decision(Contract):
    schema_version: Literal["moravec.decision/v0.2"] = "moravec.decision/v0.2"
    episode_id: str
    observation_sequence: int
    decision_id: str
    valid_until_station_time_s: float
    evidence_kind: Literal["model-inference", "contract-fixture"]
    action: Annotated[JointTrajectory | CartesianGoal | Stop, Field(discriminator="kind")]


class EpisodeStart(Contract):
    episode_id: str = Field(min_length=1)
    profile: RobotProfile


class EpisodeEnd(Contract):
    episode_id: str


class DeploymentManifest(Contract):
    schema_version: Literal["moravec.deployment/v0.2"]
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
