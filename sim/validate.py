"""Physical conformance test: WebSocket/MessagePack → Moravec policy → station executor → MuJoCo.

Defaults to explicit test doubles. --live-astra makes at most two OpenAI API calls.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import socket
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import uvicorn
from fastapi import FastAPI, Request

from moravec_policy.client_server.client import RemotePolicy
from moravec_policy.client_server.service import create_app
from moravec_policy.contracts import (
    CartesianGoal,
    Feedback,
    Observation,
    PolicyManifest,
    RobotProfile,
    Stop,
)
from moravec_policy.loader import load_entrypoint
from moravec_policy.validation import validate_decision

ROOT = Path(__file__).resolve().parents[1]
LLMHarness = load_entrypoint("policy.py:LLMHarness", ROOT / "policies/gpt-6-astra/policy.json")
FixturePolicy = load_entrypoint(
    "policy.py:FixturePolicy", ROOT / "policies/contract-fixture/policy.json"
)
TOKEN = "simulation-conformance-token-only"


@contextlib.contextmanager
def serve(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        if not thread.is_alive():
            raise RuntimeError("test server failed to start")
        time.sleep(0.01)
    else:
        raise RuntimeError("test server startup timeout")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("test server did not stop")


def model_fixture():
    app = FastAPI()

    @app.post("/v1/responses")
    async def completion(request: Request):
        payload = await request.json()
        if request.headers.get("authorization") != "Bearer " + TOKEN:
            raise RuntimeError("bad fixture token")
        last = payload["input"][-1]["content"]
        assert any(x["type"] == "input_image" for x in last)
        obs = json.loads(last[0]["text"])
        target = json.loads(obs["instruction"])
        args = {
            "kind": "cartesian-goal",
            "arm": "arm",
            "duration_s": 1.0,
            "target": {
                "frame": "world",
                "position_m": target["tcp_target_m"],
                "orientation_wxyz": target["orientation_wxyz"],
            },
        }
        return {
            "model": "contract-fixture-llm",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"call-{obs['sequence']}",
                    "name": "move_tcp",
                    "arguments": json.dumps(args),
                }
            ],
        }

    return app


def tcp_target(q):
    a, b = q
    return [
        0.3 * math.cos(a) + 0.25 * math.cos(a + b),
        0.3 * math.sin(a) + 0.25 * math.sin(a + b),
        0.08,
    ]


def solve_goal(action):
    x, y, z = action.target.position_m
    if action.target.frame != "world" or abs(z - 0.08) > 1e-5 or action.arm != "arm":
        raise ValueError("goal incompatible with planar fixture")
    c = (x * x + y * y - 0.3**2 - 0.25**2) / (2 * 0.3 * 0.25)
    if not -1 <= c <= 1:
        raise ValueError("unreachable goal")
    b = math.acos(c)
    a = math.atan2(y, x) - math.atan2(0.25 * math.sin(b), 0.3 + 0.25 * math.cos(b))
    expected = np.array([math.cos((a + b) / 2), 0, 0, math.sin((a + b) / 2)])
    if (
        min(
            np.linalg.norm(expected - np.array(action.target.orientation_wxyz)),
            np.linalg.norm(expected + np.array(action.target.orientation_wxyz)),
        )
        > 1e-4
    ):
        raise ValueError("orientation unreachable on planar fixture")
    return np.array([a, b])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/conformance")
    parser.add_argument(
        "--live-astra",
        action="store_true",
        help="Make at most two paid OpenAI calls in one simulation episode",
    )
    args = parser.parse_args()
    if args.live_astra and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required; no API calls made")
    args.output.mkdir(parents=True, exist_ok=True)
    profile = RobotProfile.model_validate_json((ROOT / "sim/profile.json").read_text())
    base = json.loads((ROOT / "policies/contract-fixture/policy.json").read_text())
    results = []
    events = []
    frames = []
    model = mujoco.MjModel.from_xml_path(str(ROOT / "sim/reach.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        with serve(model_fixture()) as model_url:
            for mode in (
                ["gpt-6-astra"]
                if args.live_astra
                else ["joint-trajectory", "relative-pose", "llm-http-tool-loop"]
            ):
                if mode in {"joint-trajectory", "relative-pose"}:
                    cfg = json.loads(json.dumps(base))
                    cfg["config"]["output"] = mode
                    manifest = PolicyManifest.model_validate(cfg)
                    policy = FixturePolicy(mode)
                else:
                    cfg = json.loads((ROOT / "policies/gpt-6-astra/policy.json").read_text())
                    if not args.live_astra:
                        cfg["model"] = {
                            "name": "contract-fixture-llm",
                            "provider_model_id": "contract-fixture-llm",
                        }
                        cfg["config"]["url"] = model_url + "/v1/responses"
                    cfg["config"]["max_calls"] = 2
                    manifest = PolicyManifest.model_validate(cfg)
                    policy = LLMHarness(manifest, token=None if args.live_astra else TOKEN)

                with serve(create_app(policy, manifest, TOKEN)) as url:
                    client = RemotePolicy(url.replace("http://", "ws://"), TOKEN, timeout_s=75)
                    try:
                        for seed in range(1 if args.live_astra else 5):
                            rng = np.random.default_rng(seed)
                            q_target = np.array([rng.uniform(0.1, 0.5), rng.uniform(0.4, 0.9)])
                            target = tcp_target(q_target)
                            quat = [math.cos(sum(q_target) / 2), 0, 0, math.sin(sum(q_target) / 2)]
                            instruction = json.dumps(
                                {
                                    "joint_target": dict(
                                        zip(["shoulder", "elbow"], q_target.tolist())
                                    ),
                                    "tcp_target_m": target,
                                    "orientation_wxyz": quat,
                                }
                            )
                            mujoco.mj_resetData(model, data)
                            mujoco.mj_forward(model, data)
                            episode = f"{mode}-{seed}"
                            client.start(episode, profile)
                            feedback = None
                            for seq in range(2):
                                renderer.update_scene(data, camera="overview")
                                rgb = renderer.render().copy()
                                yaw = float(sum(data.qpos[:2]))
                                now = float(data.time)
                                obs = Observation(
                                    episode_id=episode,
                                    sequence=seq,
                                    station_time_s=now,
                                    captured_at_s=now,
                                    profile_id=profile.id,
                                    profile_revision=profile.revision,
                                    instruction=instruction,
                                    cameras={
                                        "overview": {
                                            "shape": list(rgb.shape),
                                            "data": rgb.tobytes(),
                                            "captured_at_s": now,
                                        }
                                    },
                                    joint_positions=dict(
                                        zip(["shoulder", "elbow"], data.qpos[:2].tolist())
                                    ),
                                    tcp_poses={
                                        "arm": {
                                            "frame": "world",
                                            "position_m": data.site("tcp").xpos.tolist(),
                                            "orientation_wxyz": [
                                                math.cos(yaw / 2),
                                                0,
                                                0,
                                                math.sin(yaw / 2),
                                            ],
                                        }
                                    },
                                    gripper_open_fraction={},
                                    feedback=feedback,
                                )
                                client.update_obs(obs)
                                decision = client.get_action()
                                validate_decision(decision, obs, profile, float(data.time))
                                if isinstance(decision.action, Stop):
                                    events.append(
                                        {
                                            "episode_id": episode,
                                            "sequence": seq,
                                            "decision": decision.model_dump(),
                                            "provider": policy.last_response
                                            if isinstance(policy, LLMHarness)
                                            else None,
                                        }
                                    )
                                    break
                                from moravec_policy.geometry import resolve_pose_chunk

                                action = decision.action
                                if action.steps[0].tcp_targets:
                                    poses = resolve_pose_chunk(action, obs.tcp_poses)
                                    commands = [
                                        solve_goal(
                                            CartesianGoal(
                                                kind="cartesian-goal",
                                                arm="arm",
                                                target=p["arm"],
                                                duration_s=action.timestep_s,
                                            )
                                        )
                                        for p in poses
                                    ]
                                else:
                                    commands = [
                                        np.array(
                                            [
                                                point.joint_positions.get(n, obs.joint_positions[n])
                                                for n in ["shoulder", "elbow"]
                                            ]
                                        )
                                        for point in action.steps[: action.execution_horizon]
                                    ]
                                times = [(i + 1) * action.timestep_s for i in range(len(commands))]
                                if any(
                                    not j.lower <= v <= j.upper
                                    for command in commands
                                    for j, v in zip(profile.joints, command)
                                ):
                                    raise ValueError("planner exceeded joint limits")
                                times = [0.0, *times]
                                commands = np.vstack([data.qpos[:2].copy(), *commands])
                                step_count = math.ceil((times[-1] + 0.5) / model.opt.timestep)
                                for step in range(step_count):
                                    elapsed = step * model.opt.timestep
                                    data.ctrl[:] = [
                                        np.interp(elapsed, times, commands[:, i]) for i in range(2)
                                    ]
                                    mujoco.mj_step(model, data)
                                    if any(w.number for w in data.warning):
                                        raise RuntimeError("physics warning invalidates episode")
                                    if seed == 0 and step % 25 == 0:
                                        renderer.update_scene(data, camera="overview")
                                        frames.append(renderer.render().copy())
                                feedback = Feedback(
                                    decision_id=decision.decision_id,
                                    status="executed",
                                    reason="measured trajectory completed",
                                    executed_steps=step_count,
                                )
                                events.append(
                                    {
                                        "episode_id": episode,
                                        "sequence": seq,
                                        "decision": decision.model_dump(),
                                        "feedback": feedback.model_dump(),
                                        "provider": policy.last_response
                                        if isinstance(policy, LLMHarness)
                                        else None,
                                        "measured_tcp_m": data.site("tcp").xpos.tolist(),
                                    }
                                )
                            error = float(np.linalg.norm(data.site("tcp").xpos - np.array(target)))
                            results.append(
                                {
                                    "mode": mode,
                                    "seed": seed,
                                    "position_error_m": error,
                                    "passed": error < 0.01,
                                }
                            )
                            client.end(episode)
                    finally:
                        client.close()
                if isinstance(policy, LLMHarness):
                    policy.client.close()
    finally:
        renderer.close()
    if frames:
        imageio.mimsave(args.output / "conformance.mp4", frames, fps=20, macro_block_size=1)
    (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    report = {
        "simulator": "mujoco",
        "station_transport": "WebSocket/MessagePack v0.4",
        "simulator_version": mujoco.__version__,
        "model_execution": args.live_astra,
        "test_double": None
        if args.live_astra
        else "explicit deterministic provider and direct-action fixtures",
        "episodes": len(results),
        "passed": sum(r["passed"] for r in results),
        "results": results,
        "limitations": [
            "Not Isaac Sim validation",
            "Not GR00T/OpenWAM inference"
            if args.live_astra
            else "Not Astra/GR00T/OpenWAM inference",
            "Not robot task benchmark results",
        ],
        "scene_sha256": hashlib.sha256((ROOT / "sim/reach.xml").read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["passed"] != report["episodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
