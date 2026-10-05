"""LIBERO qualification against a running policy server; keep outputs outside Git."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from moravec_policy.client_server.client import RemotePolicy
from moravec_policy.contracts import Feedback, Observation, RobotProfile
from moravec_policy.validation import validate_decision


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", required=True)
    a.add_argument("--tasks", type=int, nargs="+", default=[0, 1])
    a.add_argument("--trials", type=int, default=2)
    a.add_argument("--url", default="ws://127.0.0.1:8000")
    a.add_argument("--libero-root", type=Path, required=True)
    args = a.parse_args()
    if args.trials < 1:
        a.error("--trials must be positive")
    revision = subprocess.check_output(
        ["git", "-C", str(args.libero_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "8f1084e3132a39270c3a13ebe37270a43ece2a01":
        raise ValueError("LIBERO checkout differs from qualified revision")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import yaml

    config_root = out / "libero-config"
    config_root.mkdir(exist_ok=True)
    libero_root = args.libero_root.resolve() / "libero" / "libero"
    (config_root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(libero_root),
                "bddl_files": str(libero_root / "bddl_files"),
                "init_states": str(libero_root / "init_files"),
                "assets": str(libero_root / "assets"),
                "datasets": str(args.libero_root.resolve() / "datasets"),
            }
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root.resolve())
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    results = []
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    client = RemotePolicy(args.url, os.environ["MORAVEC_POLICY_TOKEN"], timeout_s=300)
    (out / "metadata.json").write_text(
        json.dumps(
            {"policy": client.call("hello", {}), "libero_revision": revision, "seed": 42}, indent=2
        )
    )
    try:
        for taskid in args.tasks:
            task = suite.get_task(taskid)
            env = OffScreenRenderEnv(
                bddl_file_name=str(
                    Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                ),
                camera_heights=256,
                camera_widths=256,
            )
            try:
                env.seed(42)
                base = env.env
                controller = base.robots[0].controller
                assert base.control_freq == 20, base.control_freq
                np.testing.assert_allclose(controller.output_max, [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
                np.testing.assert_allclose(
                    controller.output_min, [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
                )
                np.testing.assert_allclose(controller.input_max, 1)
                np.testing.assert_allclose(controller.input_min, -1)
                init = suite.get_task_init_states(taskid)
                names = [f"joint-{i}" for i in range(7)]
                profile = RobotProfile(
                    id="libero-panda",
                    revision="libero-8f1084e-osc-pose",
                    joints=[{"name": n, "unit": "rad", "lower": -7, "upper": 7} for n in names],
                    frames=["world"],
                    cameras=["overview", "wrist"],
                    control_period_s=0.05,
                    max_observation_age_s=1,
                    max_camera_skew_s=0.01,
                    max_action_duration_s=2,
                )
                for trial in range(args.trials):
                    raw = env.reset()
                    raw = env.set_init_state(init[trial])
                    for _ in range(30):
                        raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    episode = f"task-{taskid}-trial-{trial}"
                    client.start(episode, profile)
                    feedback = None
                    step = 0
                    calls = 0
                    success = False
                    start = time.monotonic()
                    while step < 600:
                        t = step * 0.05
                        cameras = {}
                        for name, key in [
                            ("overview", "agentview_image"),
                            ("wrist", "robot0_eye_in_hand_image"),
                        ]:
                            image = np.ascontiguousarray(raw[key][::-1, ::-1])
                            cameras[name] = {
                                "shape": image.shape,
                                "data": image.tobytes(),
                                "captured_at_s": t,
                            }
                        q = raw["robot0_eef_quat"]
                        grip = raw["robot0_gripper_qpos"]
                        obs = Observation(
                            episode_id=episode,
                            sequence=calls,
                            station_time_s=t,
                            captured_at_s=t,
                            profile_id=profile.id,
                            profile_revision=profile.revision,
                            instruction=task.language,
                            cameras=cameras,
                            joint_positions=dict(zip(names, raw["robot0_joint_pos"])),
                            tcp_poses={
                                "arm": {
                                    "frame": "world",
                                    "position_m": raw["robot0_eef_pos"],
                                    "orientation_wxyz": [q[3], *q[:3]],
                                }
                            },
                            gripper_open_fraction={
                                "gripper": float(np.clip((grip[0] - grip[1]) / 0.08, 0, 1))
                            },
                            gripper_joint_positions={"gripper": grip.tolist()},
                            feedback=feedback,
                        )
                        client.update_obs(obs)
                        decision = client.get_action()
                        if decision.evidence_kind != "model-inference":
                            raise ValueError("real policy inference required")
                        validate_decision(decision, obs, profile, t)
                        calls += 1
                        executed = 0
                        action = decision.action
                        assert (
                            action.kind == "action-chunk"
                            and action.timestep_s == 0.05
                            and action.execution_horizon == 1
                        )
                        for s in action.steps[: action.execution_horizon]:
                            target = s.tcp_targets["arm"]
                            assert (
                                target.representation == "relative"
                                and target.reference == "frame"
                                and target.relative_to == "observation"
                            )
                            xyz = np.asarray(target.pose.position_m) / 0.05
                            rv = (
                                Rotation.from_quat(
                                    target.pose.orientation_wxyz, scalar_first=True
                                ).as_rotvec()
                                / 0.5
                            )
                            command = np.r_[xyz, rv, 1 - 2 * s.grippers["gripper"]]
                            assert np.max(np.abs(command)) <= 1.00001
                            raw, _, _, _ = env.step(command)
                            step += 1
                            executed += 1
                            success = bool(env.check_success())
                            if success or step >= 600:
                                break
                        feedback = Feedback(
                            decision_id=decision.decision_id,
                            status="executed",
                            reason="LIBERO controller applied action steps",
                            executed_steps=executed,
                        )
                        if success:
                            break
                    client.end(episode)
                    result = {
                        "task_id": taskid,
                        "task": task.name,
                        "trial": trial,
                        "success": success,
                        "steps": step,
                        "policy_requests": calls,
                        "wall_seconds": time.monotonic() - start,
                    }
                    results.append(result)
                    (out / "results.json").write_text(json.dumps(results, indent=2))
                    print(json.dumps(result), flush=True)
                    from PIL import Image

                    Image.fromarray(raw["agentview_image"][::-1, ::-1]).save(out / f"{episode}.png")
            finally:
                env.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
