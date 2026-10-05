import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from moravec_policy.contracts import Decision, Observation, PolicyManifest, RobotProfile
from moravec_policy.llm import LLMHarness
from moravec_policy.policies import FixturePolicy
from moravec_policy.service import create_app
from moravec_policy.validation import validate_decision, validate_observation

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-contract-token-not-a-secret"


@pytest.fixture
def profile():
    return RobotProfile.model_validate_json((ROOT / "sim/profile.json").read_text())


@pytest.fixture
def obs(profile):
    return Observation(
        episode_id="test",
        sequence=0,
        station_time_s=1,
        captured_at_s=1,
        profile_id=profile.id,
        profile_revision=profile.revision,
        instruction=json.dumps({"joint_target": {"shoulder": 0.3, "elbow": 0.7}}),
        cameras={
            "overview": {
                "shape": [1, 1, 3],
                "data": bytes([255, 0, 0]),
                "captured_at_s": 1,
            }
        },
        joint_positions={"shoulder": 0.0, "elbow": 0.0},
        tcp_poses={
            "arm": {
                "frame": "world",
                "position_m": [0.55, 0, 0.08],
                "orientation_wxyz": [1, 0, 0, 0],
            }
        },
        gripper_open_fraction={},
    )


@pytest.fixture
def client(profile):
    manifest = PolicyManifest.model_validate_json(
        (ROOT / "examples/contract-fixture/policy.json").read_text()
    )
    app = create_app(FixturePolicy("joint-trajectory"), manifest, TOKEN)
    return TestClient(app, headers={"Authorization": "Bearer " + TOKEN})


@pytest.mark.parametrize("path", list((ROOT / "examples").glob("*/policy.json")))
def test_examples(path):
    manifest = PolicyManifest.model_validate_json(path.read_text())
    assert manifest.model.name == manifest.model.name.lower()
    assert "_" not in manifest.model.name


@pytest.mark.parametrize(
    "issue", ["stale", "future", "camera-skew", "unknown-joint", "unknown-frame", "profile"]
)
def test_observation_rejection(profile, obs, issue):
    data = obs.model_dump()
    if issue == "stale":
        data["station_time_s"] = 3
    elif issue == "future":
        data["captured_at_s"] = 2
    elif issue == "camera-skew":
        data["cameras"]["overview"]["captured_at_s"] = 0.5
    elif issue == "unknown-joint":
        data["joint_positions"]["wrong"] = 0
    elif issue == "unknown-frame":
        data["tcp_poses"]["arm"]["frame"] = "unknown"
    else:
        data["profile_revision"] = "wrong"
    with pytest.raises(ValueError):
        validate_observation(Observation.model_validate(data), profile)


@pytest.mark.parametrize("issue", ["nan", "image", "hidden-reward", "quat"])
def test_payload_rejection(obs, issue):
    data = obs.model_dump()
    if issue == "nan":
        data["joint_positions"]["elbow"] = float("nan")
    elif issue == "image":
        data["cameras"]["overview"]["data"] = b"badbytes"
    elif issue == "quat":
        data["tcp_poses"]["arm"]["orientation_wxyz"] = [0, 0, 0, 0]
    else:
        data["reward"] = 1
    with pytest.raises(ValidationError):
        Observation.model_validate(data)


@pytest.mark.parametrize("issue", ["expired", "wrong-episode", "limit", "wrong-joint", "duration"])
def test_executor_rejects_without_actuation(profile, obs, issue):
    decision = infer(FixturePolicy("joint-trajectory"), obs, profile).model_dump()
    now = obs.station_time_s
    if issue == "expired":
        now = 3
    elif issue == "wrong-episode":
        decision["episode_id"] = "other"
    elif issue == "limit":
        decision["action"]["steps"][0]["joint_positions"]["elbow"] = 20
    elif issue == "wrong-joint":
        decision["action"]["steps"][0]["joint_positions"] = {"wrong": 0}
    else:
        decision["action"]["timestep_s"] = 20
    with pytest.raises(ValueError):
        validate_decision(Decision.model_validate(decision), obs, profile, now)


def test_lifecycle_feedback_and_reset(client, obs, profile):
    from moravec_policy.wire import observation_to_wire, pack, unpack

    with client.websocket_connect("/policy") as ws:
        counter = 0

        def call(method, payload):
            nonlocal counter
            counter += 1
            ws.send_bytes(
                pack(
                    {
                        "protocol": "moravec.policy/v0.4",
                        "request_id": str(counter),
                        "method": method,
                        "payload": payload,
                    }
                )
            )
            return unpack(ws.receive_bytes())

        start = {"episode_id": obs.episode_id, "profile": profile.model_dump()}
        assert "result" in call("reset", start)
        assert "result" in call("update_obs", observation_to_wire(obs))
        first = call("get_action", {})["result"]
        assert first["evidence_kind"] == "contract-fixture"
        assert "error" in call("get_action", {})
        assert "result" in call("reset", start)
        assert "result" in call("update_obs", observation_to_wire(obs))
        first = call("get_action", {})["result"]
        data = obs.model_dump()
        data.update(
            sequence=1,
            feedback={
                "decision_id": first["decision_id"],
                "status": "executed",
                "reason": "measured",
                "executed_steps": 50,
            },
        )
        assert "result" in call("update_obs", observation_to_wire(Observation.model_validate(data)))
        assert "result" in call("get_action", {})
        assert "result" in call("end", {"episode_id": obs.episode_id})


def test_auth(client):
    from starlette.websockets import WebSocketDisconnect

    assert client.get("/health").status_code == 200
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/policy", headers={"Authorization": "invalid"}),
    ):
        pass


def llm_config():
    return PolicyManifest.model_validate_json(
        (ROOT / "examples/gpt-6-astra/policy.json").read_text()
    )


def test_llm_http_schema_and_feedback(profile, obs):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert body["store"] is False
        assert body["parallel_tool_calls"] is False
        assert all(t["strict"] for t in body["tools"])
        assert body["model"] == "gpt-6-astra"
        assert body["input"][-1]["content"][1]["text"] == "Camera: overview"
        args = {
            "kind": "cartesian-goal",
            "arm": "arm",
            "duration_s": 1,
            "target": {
                "frame": "world",
                "position_m": [0.4, 0.2, 0.08],
                "orientation_wxyz": [1, 0, 0, 0],
            },
        }
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "reasoning-1",
                        "encrypted_content": "opaque",
                        "summary": [],
                    },
                    {
                        "type": "function_call",
                        "call_id": f"call-{len(requests)}",
                        "name": "move_tcp",
                        "arguments": json.dumps(args),
                    },
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        policy = LLMHarness(llm_config(), client=http, token=TOKEN)
        first = infer(policy, obs, profile)
        assert first.action.kind == "action-chunk"
        next_obs = obs.model_copy(update={"sequence": 1})
        with pytest.raises(ValueError, match="feedback"):
            infer(policy, next_obs, profile)
        from moravec_policy.contracts import Feedback

        next_obs.feedback = Feedback(
            decision_id=first.decision_id, status="rejected", reason="unreachable", executed_steps=0
        )
        infer(policy, next_obs, profile)
        assert any(
            m.get("type") == "function_call_output" and "unreachable" in m["output"]
            for m in requests[-1]["input"]
        )
        assert any(
            m.get("type") == "reasoning" and m["encrypted_content"] == "opaque"
            for m in requests[-1]["input"]
        )
        assert (
            sum(
                p["type"] == "input_image"
                for m in requests[-1]["input"]
                if m.get("role") == "user"
                for p in m["content"]
            )
            == 1
        )
        policy.reset()
        assert len(policy.history) == 0 and policy.calls == 0


@pytest.mark.parametrize(
    "failure", ["model", "tool", "multiple", "json", "http", "incomplete", "refusal"]
)
def test_llm_errors_never_become_motion(profile, obs, failure):
    def respond(request):
        call = {
            "type": "function_call",
            "call_id": "c",
            "name": "stop",
            "arguments": '{"kind":"stop","reason":"test"}',
        }
        payload = {"model": "gpt-6-astra", "status": "completed", "output": [call]}
        if failure == "model":
            payload["model"] = "other-model"
        elif failure == "tool":
            call["name"] = "execute_shell"
        elif failure == "multiple":
            payload["output"] = [call, call]
        elif failure == "json":
            call["arguments"] = "not json"
        elif failure == "incomplete":
            payload["status"] = "incomplete"
        elif failure == "refusal":
            payload["output"] = [
                {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}
            ]
        elif failure == "http":
            return httpx.Response(402, json={"error": "funding"})
        return httpx.Response(200, json=payload)

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as http,
        pytest.raises((ValueError, httpx.HTTPStatusError)),
    ):
        infer(LLMHarness(llm_config(), client=http, token=TOKEN), obs, profile)


def test_isolated_blocks_public_llm():
    cfg = llm_config().model_copy(update={"runtime_profile": "isolated"})
    with pytest.raises(ValueError, match="isolated"):
        LLMHarness(cfg, token=TOKEN)


@pytest.mark.parametrize("name", ["gr00t-n1.7-3b", "openwam-alpha-pretrain-foundation-model"])
def test_native_checkpoints_fail_closed(name):
    from moravec_policy.policies import load_policy

    path = ROOT / "examples" / name / "policy.json"
    with pytest.raises(ValueError, match="qualification"):
        load_policy(PolicyManifest.model_validate_json(path.read_text()), path)


def test_binary_uncertainty():
    from moravec_policy.statistics import wilson_interval

    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.7224672)
    assert hi == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_strict_tool_schema():
    from moravec_policy.llm import tool_definitions

    def check(node):
        if isinstance(node, dict):
            assert "default" not in node and "prefixItems" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    for tool in tool_definitions():
        check(tool["parameters"])


def infer(policy, obs, profile):
    if not hasattr(policy, "profile"):
        policy.configure(profile)
    policy.update_obs(obs)
    action = policy.get_action()
    return Decision(
        episode_id=obs.episode_id,
        observation_sequence=obs.sequence,
        decision_id=getattr(policy, "pending", None) or "test-decision",
        valid_until_station_time_s=obs.captured_at_s + profile.max_observation_age_s,
        evidence_kind="contract-fixture",
        action=action,
    )


def test_multiple_updates_and_duplicate_requests(client, obs, profile):
    from moravec_policy.wire import observation_to_wire, pack, unpack

    with client.websocket_connect("/policy") as ws:

        def send(request_id, method, payload):
            ws.send_bytes(
                pack(
                    {
                        "protocol": "moravec.policy/v0.4",
                        "request_id": request_id,
                        "method": method,
                        "payload": payload,
                    }
                )
            )
            return unpack(ws.receive_bytes())

        assert "result" in send(
            "1", "reset", {"episode_id": obs.episode_id, "profile": profile.model_dump()}
        )
        assert "result" in send("2", "update_obs", observation_to_wire(obs))
        newer = obs.model_copy(update={"sequence": 1})
        assert "result" in send("3", "update_obs", observation_to_wire(newer))
        result = send("4", "get_action", {})
        assert result["result"]["observation_sequence"] == 1
        assert "error" in send("4", "get_action", {})
        assert "error" in send("5", "get_action", {})


def test_missing_feedback_faults_session(client, obs, profile):
    from moravec_policy.wire import observation_to_wire, pack, unpack

    with client.websocket_connect("/policy") as ws:

        def call(i, method, payload):
            ws.send_bytes(
                pack(
                    {
                        "protocol": "moravec.policy/v0.4",
                        "request_id": str(i),
                        "method": method,
                        "payload": payload,
                    }
                )
            )
            return unpack(ws.receive_bytes())

        call(1, "reset", {"episode_id": obs.episode_id, "profile": profile.model_dump()})
        call(2, "update_obs", observation_to_wire(obs))
        call(3, "get_action", {})
        next_obs = obs.model_copy(update={"sequence": 1})
        assert "error" in call(4, "update_obs", observation_to_wire(next_obs))
        assert "error" in call(5, "get_action", {})
