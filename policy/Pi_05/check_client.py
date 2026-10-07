"""Check real Pi_05 inference for the bundled ARX X5 fixture; never execute actions."""
import argparse

import numpy as np

from client_server.model_client import WsModelClient


def validate_actions(actions):
    if not isinstance(actions, list) or len(actions) != 50:
        raise ValueError('Expected a 50-step ARX X5 joint action chunk')
    expected = {
        'left_arm_joint_state': (6,),
        'left_ee_joint_state': (1,),
        'right_arm_joint_state': (6,),
        'right_ee_joint_state': (1,),
    }
    for action in actions:
        if not isinstance(action, dict) or set(action) != set(expected):
            raise ValueError('Expected both arms and grippers in each action')
        for key, shape in expected.items():
            value = np.asarray(action[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid action field: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:6000')
    args = parser.parse_args()
    obs = {
        'images': {name: np.zeros((224, 224, 3), dtype=np.uint8)
                   for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
        # Checkpoint order: left arm, left gripper, right arm, right gripper.
        'state': np.zeros(14, dtype=np.float32),
        'instruction': 'stack the bowls',
    }
    with WsModelClient(url=args.url, evaluation_id='container-check',
                       trial_id='synthetic-input', request_timeout_s=600) as client:
        client.call('reset')
        client.call('update_obs', obs=obs)
        validate_actions(client.call('get_action'))
        client.call('reset')
    print('PASS: real server returned a finite 50-step joint action chunk for synthetic input.')
    print('No simulation, task score, or robot execution was performed.')


if __name__ == '__main__':
    main()
