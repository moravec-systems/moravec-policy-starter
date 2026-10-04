"""Strict JSON contracts. Pydantic models are the JSON Schema source of truth."""

from __future__ import annotations

import base64
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Source(Contract):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class Checkpoint(Contract):
    path: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Literal["required", "configured"] = "required"


class XPolicyBackend(Contract):
    kind: Literal["xpolicy"]
    policy_name: Literal["GR00T_N17", "OpenWAM"]
    source: Source
    config_path: str
    checkpoint: Checkpoint


class AstraBackend(Contract):
    kind: Literal["roboprobe"]
    source: Source
    harness: Literal["RoboDojo_Agent_L3_Inspect_EEF"]
    planner: Literal["astra"]
    model: Literal["gpt-6-astra"]
    base_url_env: str
    api_key_env: str
    reasoning_effort: Literal["low", "medium", "high"]
    max_model_calls: int = Field(gt=0, le=170)
    image_history: int = Field(gt=0, le=16)


class MockBackend(Contract):
    kind: Literal["mock"]


class PolicyManifest(Contract):
    schema_version: Literal["moravec.policy/v0.1"]
    id: str
    family: Literal["llm", "vla", "world_action", "test_fixture"]
    embodiment: Literal["arx_x5"]
    action_type: Literal["joint", "ee"]
    observation_profile: Literal["rgb-proprioception/v1"]
    execution: Literal["policy_service", "native_harness", "fixture"]
    validation_status: Literal["contract_only", "live_validated"] = "contract_only"
    backend: Annotated[XPolicyBackend | AstraBackend | MockBackend, Field(discriminator="kind")]

    @model_validator(mode="after")
    def compatible(self):
        expected = {
            "mock": ("test_fixture", "fixture", "joint"),
            "roboprobe": ("llm", "native_harness", "joint"),
            "GR00T_N17": ("vla", "policy_service", "joint"),
            "OpenWAM": ("world_action", "policy_service", "ee"),
        }
        key = (
            self.backend.policy_name
            if isinstance(self.backend, XPolicyBackend)
            else self.backend.kind
        )
        if (self.family, self.execution, self.action_type) != expected[key]:
            raise ValueError("family, execution and action type must match the selected adapter")
        return self


class Camera(Contract):
    encoding: Literal["rgb8-base64"] = "rgb8-base64"
    shape: tuple[int, int, Literal[3]]
    data: str = Field(max_length=16_777_216)
    captured_at_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_image(self):
        h, w, _ = self.shape
        if not (1 <= h <= 2048 and 1 <= w <= 2048):
            raise ValueError("image dimensions must be 1..2048")
        try:
            raw = base64.b64decode(self.data, validate=True)
        except ValueError as exc:
            raise ValueError("invalid image base64") from exc
        if len(raw) != h * w * 3:
            raise ValueError("RGB byte count does not match shape")
        return self


class ArmState(Contract):
    joints: list[float] = Field(min_length=6, max_length=6)
    gripper: float = Field(ge=0, le=1)
    ee_pose: list[float] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def unit_quaternion(self):
        if not math.isclose(sum(v * v for v in self.ee_pose[3:]), 1, abs_tol=1e-3):
            raise ValueError("pose must have a unit WXYZ quaternion")
        return self


class Observation(Contract):
    schema_version: Literal["moravec.observation/v0.1"] = "moravec.observation/v0.1"
    episode_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    captured_at_ns: int = Field(ge=0)
    instruction: str = Field(min_length=1)
    embodiment: Literal["arx_x5"] = "arx_x5"
    # Native RoboDojo environment-relative world EE frame; not generic grasp/TCP.
    frame_profile: Literal["robodojo-arx-x5-native/v1"] = "robodojo-arx-x5-native/v1"
    cameras: dict[str, Camera]
    left: ArmState
    right: ArmState

    @model_validator(mode="after")
    def camera_profile(self):
        if set(self.cameras) != {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
            raise ValueError("profile requires exactly the three named RGB cameras")
        if any(c.captured_at_ns > self.captured_at_ns for c in self.cameras.values()):
            raise ValueError("camera timestamp cannot follow observation timestamp")
        return self


class JointStep(Contract):
    kind: Literal["joint"]
    left: list[float] = Field(min_length=6, max_length=6)
    right: list[float] = Field(min_length=6, max_length=6)
    left_gripper: float = Field(ge=0, le=1)
    right_gripper: float = Field(ge=0, le=1)


class EEStep(Contract):
    kind: Literal["ee"]
    left: list[float] = Field(min_length=7, max_length=7)
    right: list[float] = Field(min_length=7, max_length=7)
    left_gripper: float = Field(ge=0, le=1)
    right_gripper: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def unit_quaternions(self):
        for pose in (self.left, self.right):
            if not math.isclose(sum(v * v for v in pose[3:]), 1, abs_tol=1e-3):
                raise ValueError("action quaternion must be unit WXYZ")
        return self


class ActionChunk(Contract):
    episode_id: str
    observation_sequence: int = Field(ge=0)
    action_type: Literal["joint", "ee"]
    frame_profile: Literal["robodojo-arx-x5-native/v1"] = "robodojo-arx-x5-native/v1"
    timestep_s: float = Field(gt=0, le=1)
    steps: list[Annotated[JointStep | EEStep, Field(discriminator="kind")]] = Field(
        min_length=1, max_length=256
    )
    evidence_kind: Literal["model_inference", "mock"]

    @model_validator(mode="after")
    def homogeneous(self):
        if any(s.kind != self.action_type for s in self.steps):
            raise ValueError("mixed action spaces are not permitted")
        return self


class EpisodeStart(Contract):
    episode_id: str = Field(min_length=1)


class PredictRequest(Contract):
    observation: Observation


class ContainerDeployment(Contract):
    kind: Literal["container"]
    image: str = Field(pattern=r"^.+@sha256:[0-9a-f]{64}$")
    token_secret_ref: str
    gpu_count: int = Field(ge=0)


class RemoteDeployment(Contract):
    kind: Literal["remote"]
    endpoint: str = Field(pattern=r"^https://[^\s]+$")
    token_secret_ref: str


class DeploymentManifest(Contract):
    schema_version: Literal["moravec.deployment/v0.1"]
    policy_id: str
    policy_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: Annotated[ContainerDeployment | RemoteDeployment, Field(discriminator="kind")]


class EvaluationSpec(Contract):
    schema_version: Literal["moravec.evaluation/v0.1"]
    policy_id: str
    benchmark: Literal["RoboDojo"]
    runtime_profile: str
    task: str
    layout_ids: list[int] = Field(min_length=1)
    status: Literal["smoke", "preliminary", "official_protocol"]
    infrastructure_retry_limit: int = Field(ge=0, le=3)

    @model_validator(mode="after")
    def unique_layouts(self):
        if any(x < 0 for x in self.layout_ids) or len(set(self.layout_ids)) != len(self.layout_ids):
            raise ValueError("layout ids must be nonnegative and unique")
        return self
