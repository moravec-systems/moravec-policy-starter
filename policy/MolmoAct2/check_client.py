"""Request real MolmoAct2 inference using synthetic inputs; never execute actions."""
import argparse

import numpy as np

from client_server.model_client import WsModelClient


def observation():
    return {
        "instruction": "Pick up the object and place it in the bowl.",
        "vision": {
            name: {"color": np.zeros((256, 256, 3), dtype=np.uint8)}
            for name in ("top", "left", "right")
        },
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.zeros(1, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.zeros(1, dtype=np.float32),
        },
    }


def validate_actions(actions, expected_steps):
    if not isinstance(actions, list) or len(actions) != expected_steps:
        raise ValueError(f"Expected a {expected_steps}-step YAM action chunk")
    expected = {
        "left_arm_joint_state": 6, "left_ee_joint_state": 1,
        "right_arm_joint_state": 6, "right_ee_joint_state": 1,
    }
    for action in actions:
        if not isinstance(action, dict) or set(action) != set(expected):
            raise ValueError("Unexpected action fields")
        for key, size in expected.items():
            values = np.asarray(action[key])
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid action field: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:6000")
    parser.add_argument("--expected-steps", type=int, choices=range(1, 31), default=30)
    args = parser.parse_args()
    with WsModelClient(url=args.url, evaluation_id="container-check",
                       trial_id="synthetic-input", request_timeout_s=180) as client:
        client.call("reset")
        client.call("update_obs", obs=observation())
        validate_actions(client.call("get_action"), args.expected_steps)
        client.call("reset")
    print("PASS: real server returned a finite YAM action chunk for synthetic input.")
    print("No simulation, task score, or robot execution was performed.")


if __name__ == "__main__":
    main()
