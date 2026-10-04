import base64
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from moravec_policy.adapters import MockPolicy, XPolicyAdapter, from_xpolicy, to_xpolicy
from moravec_policy.astra import plan
from moravec_policy.cli import read_manifest
from moravec_policy.contracts import ActionChunk, Observation
from moravec_policy.provenance import tree_hash
from moravec_policy.service import create_app

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "local-contract-test-token-only"


@pytest.fixture
def obs():
    return Observation.model_validate_json((ROOT / "fixtures/observation.json").read_text())


@pytest.fixture
def client():
    app = create_app(MockPolicy(), read_manifest(ROOT / "examples/mock/policy.json"), TOKEN)
    return TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})


@pytest.mark.parametrize("name", ["astra", "groot", "openwam", "mock"])
def test_manifests(name):
    manifest = read_manifest(ROOT / f"examples/{name}/policy.json")
    assert manifest.validation_status == "contract_only"


def test_rgb_roundtrip_no_channel_swap(obs):
    decoded = to_xpolicy(obs)
    np.testing.assert_array_equal(decoded["vision"]["cam_head"]["color"][0, 0], [255, 0, 0])
    assert decoded["vision"]["cam_head"]["color"].dtype == np.uint8
    assert decoded["state"]["left_ee_pose"].tolist() == obs.left.ee_pose
    assert "reward" not in decoded


@pytest.mark.parametrize(
    "change", ["nan", "missing_camera", "bad_rgb", "bad_quat", "hidden_reward"]
)
def test_invalid_observations_rejected(obs, change):
    data = obs.model_dump()
    if change == "nan":
        data["left"]["joints"][0] = float("nan")
    elif change == "missing_camera":
        del data["cameras"]["cam_head"]
    elif change == "bad_rgb":
        data["cameras"]["cam_head"]["data"] = base64.b64encode(b"short").decode()
    elif change == "bad_quat":
        data["left"]["ee_pose"][3:] = [0, 0, 0, 0]
    else:
        data["reward"] = 1
    with pytest.raises(ValidationError):
        Observation.model_validate(data)


def native_step(obs, action="joint"):
    key = "arm_joint_state" if action == "joint" else "ee_pose"
    return {
        f"{side}_{field}": np.array(values)
        for side in ("left", "right")
        for field, values in [
            (key, getattr(obs, side).joints if action == "joint" else getattr(obs, side).ee_pose),
            ("ee_joint_state", [1.0]),
        ]
    }


@pytest.mark.parametrize("action", ["joint", "ee"])
def test_native_action_roundtrip(obs, action):
    result = from_xpolicy([native_step(obs, action)], obs, action, 0.05)
    assert result.action_type == action
    assert result.episode_id == obs.episode_id
    assert result.steps[0].left_gripper == 1


@pytest.mark.parametrize("failure", ["nan", "bad_gripper", "empty", "extra_key"])
def test_bad_actions_fail_closed(obs, failure):
    raw = [native_step(obs)]
    if failure == "nan":
        raw[0]["left_arm_joint_state"][0] = np.nan
    elif failure == "bad_gripper":
        raw[0]["left_ee_joint_state"][0] = 2
    elif failure == "empty":
        raw = []
    else:
        raw[0]["unexpected"] = 1
    with pytest.raises(ValueError):
        from_xpolicy(raw, obs, "joint", 0.05)


def test_service_episode_lifecycle_and_stale_rejection(client, obs):
    body = {"episode_id": obs.episode_id}
    assert client.post("/v1/predict", json={"observation": obs.model_dump()}).status_code == 409
    assert client.post("/v1/episode/start", json=body).status_code == 200
    assert client.post("/v1/episode/start", json={"episode_id": "other"}).status_code == 409
    response = client.post("/v1/predict", json={"observation": obs.model_dump()})
    assert response.status_code == 200
    assert ActionChunk.model_validate(response.json()).evidence_kind == "mock"
    assert client.post("/v1/predict", json={"observation": obs.model_dump()}).status_code == 409
    assert client.post("/v1/episode/end", json=body).status_code == 200
    assert client.post("/v1/episode/start", json=body).status_code == 200
    assert client.post("/v1/predict", json={"observation": obs.model_dump()}).status_code == 200


def test_auth_required(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/capabilities", headers={"Authorization": ""}).status_code == 401
    assert (
        client.post(
            "/v1/episode/start", json={"episode_id": "a"}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_real_adapter_uses_upstream_methods_and_reset(obs):
    class Upstream:
        def __init__(self):
            self.history = []

        def reset(self):
            self.history.clear()

        def update_obs(self, value):
            self.history.append(value)

        def get_action(self):
            return [native_step(obs)]

    model = Upstream()
    adapter = XPolicyAdapter(model, "joint", 0.05)
    assert adapter.predict(obs).evidence_kind == "model_inference"
    assert len(model.history) == 1
    adapter.reset()
    assert model.history == []


def test_fault_poisoning_no_secret_echo(obs):
    class Fault(MockPolicy):
        def predict(self, observation):
            raise RuntimeError("SECRET-CREDENTIAL")

    app = create_app(Fault(), read_manifest(ROOT / "examples/mock/policy.json"), TOKEN)
    client = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    client.post("/v1/episode/start", json={"episode_id": obs.episode_id})
    result = client.post("/v1/predict", json={"observation": obs.model_dump()})
    assert result.status_code == 502
    assert "SECRET" not in result.text
    next_obs = obs.model_copy(update={"sequence": 1})
    assert (
        client.post("/v1/predict", json={"observation": next_obs.model_dump()}).status_code == 409
    )


def test_astra_keeps_reference_entrypoint(tmp_path):
    manifest = read_manifest(ROOT / "examples/astra/policy.json")
    invocation = plan(manifest, tmp_path, "stack_bowls", 0, "uv", 0)
    assert "RoboDojo_Agent_L3_Inspect_EEF/run_fixed_layout.sh" in invocation["argv"][1]
    assert invocation["environment"]["L3_INSPECT_MODEL"] == "gpt-6-astra"
    assert invocation["argv"][2:4] == ["0", "0"]
    assert "ASTRA_API_KEY" in invocation["required_environment"]


def test_checkpoint_hash_includes_normalization(tmp_path):
    (tmp_path / "weights").write_bytes(b"weights")
    original = tree_hash(tmp_path)
    (tmp_path / "normalization.json").write_text("{}")
    assert tree_hash(tmp_path) != original


def test_gr00t_cannot_silently_use_wrong_action_space():
    data = json.loads((ROOT / "examples/groot/policy.json").read_text())
    data["action_type"] = "ee"
    from moravec_policy.contracts import PolicyManifest

    with pytest.raises(ValidationError):
        PolicyManifest.model_validate(data)


def test_remote_rejects_cleartext_external_and_credentials():
    from moravec_policy.client import RemotePolicy

    for url in ["http://example.com", "https://user:pass@example.com", "https://example.com?key=x"]:
        with pytest.raises(ValueError):
            RemotePolicy(url, TOKEN)


@pytest.mark.parametrize("name", ["groot", "openwam"])
def test_real_models_never_fall_back_when_weights_missing(name):
    from moravec_policy.adapters import load_policy

    path = ROOT / f"examples/{name}/policy.json"
    with pytest.raises(ValueError, match="configure a compatible checkpoint"):
        load_policy(read_manifest(path), path, 0.05)


def test_checkpoint_source_drift_rejected(tmp_path):
    import subprocess

    from moravec_policy.adapters import verify_checkout

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "adapter.py").write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "adapter.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    sha = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    verify_checkout(tmp_path, sha)
    (tmp_path / "adapter.py").write_text("modified")
    with pytest.raises(ValueError, match="tracked changes"):
        verify_checkout(tmp_path, sha)
