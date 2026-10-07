"""Request one real LLM agent action using a synthetic scene; never execute actions."""
import argparse

import cv2
import numpy as np

from client_server.model_client import WsModelClient


def observation():
    image = np.full((480, 640, 3), 170, dtype=np.uint8)
    cv2.rectangle(image, (300, 200), (340, 240), (220, 30, 30), -1)  # red block
    pose = lambda xyz: np.array([*xyz, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return {
        "instruction": "Move the left gripper above the red block at x=0.10, y=0.25, z=0.02.",
        "vision": {"cam_head": {"color": image}},
        "state": {
            "left_ee_pose": pose([0.10, 0.0, 0.30]),
            "left_ee_joint_state": np.array([1.0], dtype=np.float32),
            "right_ee_pose": pose([0.45, 0.2, 0.30]),
            "right_ee_joint_state": np.array([1.0], dtype=np.float32),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:6000")
    args = parser.parse_args()
    with WsModelClient(url=args.url, evaluation_id="container-check",
                       trial_id="synthetic-input", request_timeout_s=300) as client:
        client.call("reset")
        client.call("update_obs", obs=observation())
        actions = client.call("get_action")
        client.call("reset")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Expected a nonempty action chunk")
    for action in actions:
        for arm in ("left", "right"):
            pose = np.asarray(action[f"{arm}_ee_pose"])
            grip = np.asarray(action[f"{arm}_ee_joint_state"])
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid {arm} pose")
            if grip.shape != (1,) or not (0 <= grip[0] <= 1):
                raise ValueError(f"Invalid {arm} gripper")
    goal = np.round(actions[-1]["left_ee_pose"][:3], 3).tolist()
    print(f"PASS: one paid model call returned {len(actions)} end-effector steps; left goal {goal}.")
    print("No simulation, task score, or robot execution was performed.")


if __name__ == "__main__":
    main()
