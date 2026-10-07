"""Check action output from the OpenWAM simulation checkpoint; execute no actions."""
import argparse

import numpy as np

from client_server.model_client import WsModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:6000')
    args = parser.parse_args()
    obs = {
        'vision': {name: {'color': np.zeros((256, 256, 3), dtype=np.uint8)}
                   for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
        'state': {
            'left_ee_pose': np.array([0.0, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0]),
            'right_ee_pose': np.array([0.0, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0]),
            'left_ee_joint_state': np.array([1.0]),
            'right_ee_joint_state': np.array([1.0]),
        },
        'instruction': 'stack the bowls',
    }
    with WsModelClient(url=args.url, evaluation_id='container-check',
                       trial_id='synthetic-input', request_timeout_s=300) as client:
        client.call('reset')
        client.call('update_obs', obs=obs)
        actions = client.call('get_action')
        if not isinstance(actions, list) or not actions:
            raise ValueError('Expected a nonempty action chunk')
        for action in actions:
            for arm in ('left', 'right'):
                pose = np.asarray(action[f'{arm}_ee_pose'])
                grip = np.asarray(action[f'{arm}_ee_joint_state'])
                if pose.shape != (7,) or not np.isfinite(pose).all():
                    raise ValueError(f'Invalid {arm} pose')
                if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4):
                    raise ValueError(f'Invalid {arm} quaternion')
                if grip.shape != (1,) or not np.isfinite(grip).all() or not (0 <= grip[0] <= 1):
                    raise ValueError(f'Invalid {arm} gripper')
        client.call('reset')
    print(f'PASS: server returned {len(actions)} finite bimanual action steps.')
    print('Use the default allow_dummy_policy=false configuration for real inference.')
    print('No simulation, task score, or robot execution was performed.')


if __name__ == '__main__':
    main()
