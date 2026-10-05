"""Send a synthetic observation to a real GR00T server; never execute actions."""
import argparse
import numpy as np
from client_server.model_client import WsModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:6000')
    args = parser.parse_args()
    obs = {
        'video': {name: np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)
                  for name in ('image', 'wrist_image')},
        'state': {name: np.zeros((1, 1, 1), dtype=np.float32)
                  for name in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')},
        'language': {'annotation.human.action.task_description': [['pick up the bowl']]},
    }
    obs['state']['gripper'] = np.zeros((1, 1, 2), dtype=np.float32)
    with WsModelClient(url=args.url, request_timeout_s=180) as client:
        client.call('reset')
        client.call('update_obs', obs=obs)
        actions = client.call('get_action')
        if not isinstance(actions, list) or len(actions) != 16:
            raise ValueError('Expected a 16-step native LIBERO action chunk')
        for action in actions:
            for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'):
                value = np.asarray(action[key])
                if value.size != 1 or not np.isfinite(value).all():
                    raise ValueError(f'Invalid action field: {key}')
        client.call('reset')
    print('PASS: real server returned a finite action chunk for synthetic input.')
    print('No simulation, task score, or robot execution was performed.')


if __name__ == '__main__':
    main()
