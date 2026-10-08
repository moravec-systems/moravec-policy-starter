"""Check a running synthetic demo container without executing any actions."""
import argparse
import numpy as np
from client_server.model_client import WsModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:6000')
    args = parser.parse_args()
    with WsModelClient(url=args.url, evaluation_id="container-check",
                       trial_id="synthetic-input", request_timeout_s=15) as client:
        client.call('reset')
        client.call('update_obs', obs={'env_idx': 0, 'instruction': 'Transport check'})
        actions = client.call('get_action')
        if len(actions) != 2:
            raise ValueError('Expected two synthetic action steps')
        for action in actions:
            for side in ('left', 'right'):
                for channel, size in (('arm', 6), ('ee', 1)):
                    value = np.asarray(action[f'{side}_{channel}_joint_state'])
                    if value.shape != (size,) or not np.all(value == 0):
                        raise ValueError('Unexpected synthetic action')
        client.call('reset')
    print('PASS: observation/action transport and reset. Synthetic policy only.')


if __name__ == '__main__':
    main()
