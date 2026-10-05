"""Exercise real policy adapters with fake SDKs; no weights, paid APIs or motors."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from policy_examples import ROOT
from test_interoperability import observation, profile

from moravec_policy.client_server.service import create_app
from moravec_policy.client_server.wire import observation_to_wire, pack, unpack
from moravec_policy.contracts import PolicyManifest
from moravec_policy.loader import load_policy
from moravec_policy.provenance import tree_hash
from moravec_policy.robots import load_robot

TOKEN = "policy-plugin-conformance-token"


def exercise(policy, manifest):
    """Both model families must survive the same server lifecycle and validation."""
    obs, robot = observation(), profile()
    with (
        TestClient(create_app(policy, manifest, TOKEN)) as client,
        client.websocket_connect("/policy", headers={"Authorization": "Bearer " + TOKEN}) as ws,
    ):
        for i, (method, payload) in enumerate(
            [
                ("reset", {"episode_id": obs.episode_id, "profile": robot.model_dump()}),
                ("update_obs", observation_to_wire(obs)),
                ("get_action", {}),
                ("end", {"episode_id": obs.episode_id}),
            ]
        ):
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
            reply = unpack(ws.receive_bytes())
            assert "error" not in reply, reply
            if method == "get_action":
                assert reply["result"]["action"]["steps"][0]["joint_positions"] == {
                    "shoulder": pytest.approx(0.1),
                    "elbow": pytest.approx(0.2),
                }


def test_custom_policy_requires_no_core_registration(tmp_path):
    (tmp_path / "custom.py").write_text("""
from moravec_policy.lifecycle import Policy
from moravec_policy.contracts import ActionChunk
class Custom(Policy):
    def __init__(self, manifest, path):
        self.gain = manifest.config["custom_gain"]
    def _infer(self, obs, profile):
        return ActionChunk(timestep_s=0.02, execution_horizon=1,
            steps=[{"joint_positions": {"shoulder": self.gain, "elbow": 2*self.gain}}])
""")
    manifest = PolicyManifest.model_validate(
        {
            "schema_version": "moravec.policy/v0.5",
            "harness_version": "custom-1",
            "model": {"name": "custom-model", "provider_model_id": "custom-model"},
            "family": "custom-planner",
            "runtime_profile": "isolated",
            "entrypoint": "custom.py:Custom",
            "config": {"custom_gain": 0.1},
        }
    )
    exercise(load_policy(manifest, tmp_path / "policy.json"), manifest)


@pytest.mark.parametrize("name", ["GR00T_N17", "OpenWAM"])
def test_gpu_adapter_configuration_and_wire_compatibility(name, tmp_path, monkeypatch):
    source = ROOT / "policies" / name
    spec = json.loads((source / "policy.json").read_text())
    spec["entrypoint"] = str(source / "policy.py") + ":Model"
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "test-only.txt").write_text("Explicit SDK test double; no real weights")
    spec["model"]["checkpoint_sha256"] = tree_hash(weights)
    spec["config"].update(
        checkpoint_path=str(weights), binding_config="bindings.json", device="cpu"
    )
    bindings = json.loads((source / "bindings.json").read_text())
    bindings.update(
        status="qualified", profile_id=profile().id, profile_revision=profile().revision
    )
    bindings["action_semantics"] = "absolute-joint-position"
    bindings["action_bindings"] = {"action": ["shoulder", "elbow"]}
    if name == "OpenWAM":
        # Exercise a non-joint state path, the important gap in the previous adapter.
        bindings["state_components"] = [
            {"source": "tcp_poses", "name": "arm", "frame": "world", "rotation_format": "rotvec"}
        ]
    (tmp_path / "bindings.json").write_text(json.dumps(bindings))
    seen = {}

    class SDK:
        def __init__(self, **kwargs):
            seen["constructor"] = kwargs

        def reset(self):
            seen["resets"] = seen.get("resets", 0) + 1

        def get_modality_config(self):
            return {k: SimpleNamespace(delta_indices=[0]) for k in ["video", "state"]}

        def get_action(self, inputs):
            seen["inputs"] = inputs
            return {"action": np.array([[[0.1, 0.2]]])}, {}

        def predict(self, inputs):
            seen["inputs"] = inputs
            return {"action": np.array([[0.1, 0.2]])}

    if name == "GR00T_N17":
        monkeypatch.setitem(sys.modules, "gr00t.policy", SimpleNamespace(Gr00tPolicy=SDK))
    else:

        def build(config, checkpoint, device):
            seen["deploy"] = config
            return SDK(checkpoint=checkpoint, device=device)

        monkeypatch.setitem(
            sys.modules,
            "omegaconf",
            SimpleNamespace(
                OmegaConf=SimpleNamespace(merge=lambda base, overrides: {**base, **overrides})
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "openwam.deploy.server",
            SimpleNamespace(_load_deploy_yaml=lambda path: {}, build_server_from_config=build),
        )
    manifest = PolicyManifest.model_validate(spec)
    exercise(load_policy(manifest, tmp_path / "policy.json"), manifest)
    assert seen["constructor"]["device"] == "cpu"
    assert seen["resets"] >= 2
    if name == "GR00T_N17":
        assert seen["inputs"]["video"]["front"].shape == (1, 1, 1, 1, 3)
    else:
        assert seen["inputs"]["state"] == pytest.approx([1, 0, 0, 0, 0, 0])
        assert seen["deploy"]["inference"]["denoise_steps"] == 10
        assert seen["deploy"]["optimization"]["compile"]["enabled"] is False


@pytest.mark.parametrize("path", sorted((ROOT / "robots").glob("*.json")))
def test_robot_definitions(path):
    robot = load_robot(path)
    assert robot.id == path.stem
    assert robot.joint_action_dim > 0


def test_yam_single_and_dual_joint_order():
    single = load_robot(ROOT / "robots/yam.json")
    dual = load_robot(ROOT / "robots/dual-yam.json")
    assert single.arm_dim == [6] and single.ee_dim == [1]
    assert single.joint_names == [[f"joint{i}" for i in range(1, 7)]]
    assert dual.arm_dim == [6, 6] and dual.ee_dim == [1, 1]
    assert dual.joint_action_dim == 14
    assert dual.joint_names[1] == [f"right/joint{i}" for i in range(1, 7)]
