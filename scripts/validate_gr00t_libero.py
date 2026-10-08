"""Validate the native GR00T LIBERO example against a running policy server.

Requires a separately installed LIBERO/MuJoCo runtime. Results stay local.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from libero.libero import benchmark, get_libero_path

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--port", type=int, default=6000)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--tasks", type=int, nargs="+", default=[0, 1])
p.add_argument("--trials", type=int, default=2)
args = p.parse_args()
if args.trials < 1:
    p.error("--trials must be positive")
from libero.libero.envs import OffScreenRenderEnv

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
if (out / "results.json").exists():
    raise SystemExit("Choose a new output directory to preserve previous results.")
results = []
from client_server.model_client import WsModelClient


def encode(raw, instruction):
    rot = Rotation.from_quat(raw["robot0_eef_quat"]).as_rotvec()
    state = {
        k: np.asarray([[[v]]], dtype=np.float32)
        for k, v in zip(
            ["x", "y", "z", "roll", "pitch", "yaw"], [*raw["robot0_eef_pos"], *rot]
        )
    }
    state["gripper"] = np.asarray([[raw["robot0_gripper_qpos"]]], dtype=np.float32)
    return {
        "video": {
            k: np.ascontiguousarray(raw[v][::-1, ::-1])[None, None]
            for k, v in [
                ("image", "agentview_image"),
                ("wrist_image", "robot0_eye_in_hand_image"),
            ]
        },
        "state": state,
        "language": {"annotation.human.action.task_description": [[instruction]]},
    }


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
        initial = suite.get_task_init_states(taskid)
        for trial in range(args.trials):
            env.reset()
            raw = env.set_init_state(initial[trial])
            for _ in range(30):
                raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            client = WsModelClient(
                url=f"ws://{args.host}:{args.port}",
                evaluation_id="fork-validation",
                trial_id=f"{taskid}-{trial}",
                request_timeout_s=180,
            )
            steps = calls = 0
            success = False
            started = time.monotonic()
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(out / f"task-{taskid}-trial-{trial}.mp4"), fps=20
            )
            try:
                client.call("reset")
                while steps < 600:
                    client.call("update_obs", obs=encode(raw, task.language))
                    chunk = client.call("get_action")
                    calls += 1
                    assert len(chunk) == 16
                    for action in chunk[:8]:
                        a = np.concatenate(
                            [
                                np.asarray(action[k]).reshape(-1)
                                for k in [
                                    "x",
                                    "y",
                                    "z",
                                    "roll",
                                    "pitch",
                                    "yaw",
                                    "gripper",
                                ]
                            ]
                        )
                        assert a.shape == (7,) and np.isfinite(a).all()
                        command = np.r_[np.clip(a[:6], -1, 1), -np.sign(2 * a[6] - 1)]
                        raw, _, _, _ = env.step(command)
                        steps += 1
                        writer.append_data(raw["agentview_image"][::-1, ::-1])
                        success = bool(env.check_success())
                        if success or steps >= 600:
                            break
                    if success:
                        break
            finally:
                writer.close()
                client.close()
            result = dict(
                task=task.name,
                task_id=taskid,
                trial=trial,
                success=success,
                steps=steps,
                calls=calls,
                wall_seconds=time.monotonic() - started,
            )
            results.append(result)
            (out / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        env.close()
