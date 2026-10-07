"""Run one closed-loop episode of a policy server in the bimanual YAM tabletop scene.

  python run_episode.py --url ws://127.0.0.1:6000 --chunks 20 --out episode.mp4

With --action-type joint (default) the policy returns absolute YAM joint chunks
({left,right}_arm_joint_state, {left,right}_ee_joint_state with gripper 0 closed to 1 open),
e.g. MolmoAct2-BimanualYAM. With --action-type ee it returns fingertip-centre poses
({left,right}_ee_pose, [x, y, z, qw, qx, qy, qz] in the world frame) plus grippers, e.g. the
Astra LLM agent with examples/yam_mujoco/astra_yam.yml; episodes then start from a
gripper-down ready pose.
"""

import argparse
import json
import time

import imageio
import numpy as np

from client_server.model_client import WsModelClient
from yam_scene import BLOCK_START, YamTabletop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:6000")
    parser.add_argument("--instruction", default="Pick up the red block and place it on the green plate.")
    parser.add_argument("--chunks", type=int, default=20, help="policy calls per episode")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--action-type", choices=("joint", "ee"), default="joint")
    parser.add_argument("--out", default="episode.mp4")
    args = parser.parse_args()

    env = YamTabletop(control_hz=args.hz)
    if args.action_type == "ee":
        env.move_to_ready()
    frames, latencies, steps, max_block_z = [], [], 0, 0.0
    with WsModelClient(url=args.url, evaluation_id="yam-mujoco", trial_id="episode-0", request_timeout_s=300) as client:
        client.call("reset")
        for _ in range(args.chunks):
            client.call("update_obs", obs=env.observation(args.instruction, args.action_type))
            start = time.time()
            actions = client.call("get_action")
            latencies.append(time.time() - start)
            for action in actions:
                env.step_ee(action) if args.action_type == "ee" else env.step(action)
                steps += 1
                max_block_z = max(max_block_z, float(env.data.body("block").xpos[2]))
                obs_images = [env.render(name) for name in ("top", "left", "right")]
                small = [img[::2, ::2] for img in obs_images[1:]]
                frames.append(np.concatenate([obs_images[0], np.concatenate(small, axis=1)], axis=0))
            if env.block_on_plate():
                break
    block = env.data.body("block").xpos
    print(json.dumps({
        "steps": steps, "policy_calls": len(latencies), "mean_latency_s": round(float(np.mean(latencies)), 2),
        "block_moved_m": round(float(np.linalg.norm(block - BLOCK_START)), 3),
        "block_max_height_m": round(max_block_z, 3), "success": env.block_on_plate(), "video": args.out,
    }), flush=True)
    imageio.mimsave(args.out, frames, fps=args.hz)


if __name__ == "__main__":
    main()
