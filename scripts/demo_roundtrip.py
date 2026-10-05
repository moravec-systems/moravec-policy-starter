"""Hardware-free demo: python scripts/demo_roundtrip.py (after pip install -e .).

Uses the demo and transport in a temporary evaluator layout.
Images and state are synthetic; zero actions are never sent to hardware.
"""
import asyncio
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def check_actions(actions, arm_dims, action_type):
    assert len(actions) == 2, 'Expected a two-step action chunk'
    for action in actions:
        for side, dim in zip(('left', 'right'), arm_dims):
            key = f'{side}_arm_joint_state' if action_type == 'joint' else f'{side}_ee_pose'
            assert action[key].shape == ((dim,) if action_type == 'joint' else (7,))
            assert np.isfinite(action[key]).all()
            assert action[f'{side}_ee_joint_state'].shape == (1,)


async def main():
    # Upstream helpers require a sibling env_cfg directory. Isolate the fixture
    # so this example cannot overwrite a user's evaluator configuration.
    with tempfile.TemporaryDirectory(prefix='moravec-demo-') as temporary:
        root = Path(temporary)
        package = root / 'moravec_policy'
        package.mkdir()
        shutil.copy2(REPO / 'model_template.py', package / 'model_template.py')
        for name in ('utils', 'client_server'):
            shutil.copytree(REPO / name, package / name,
                            ignore=shutil.ignore_patterns('__pycache__'))
        (package / 'policy').mkdir()
        shutil.copy2(REPO / 'policy/__init__.py', package / 'policy/__init__.py')
        shutil.copytree(REPO / 'policy/demo_policy', package / 'policy/demo_policy',
                        ignore=shutil.ignore_patterns('__pycache__'))
        env = root / 'env_cfg'
        (env / 'robot').mkdir(parents=True)
        (env / 'sim').mkdir()
        registry = REPO / 'utils/robot/_robot_info.json'
        shutil.copy2(registry, env / 'robot/_robot_info.json')
        (env / 'sim/demo.yml').write_text('scene:\n  num_envs: 2\n')
        sys.path[:0] = [str(root), str(package)]
        from moravec_policy.policy.demo_policy.model import Model
        from client_server.model_server import PolicyServer, PolicyServerConfig
        from client_server.model_client import WsModelClient

        for robot, dimensions in json.loads(registry.read_text()).items():
            (env / f'{robot}.yml').write_text(f'config:\n  robot: {robot}\n  sim: demo\n')
            state = {}
            for side, dim in zip(('left', 'right'), dimensions['arm_dim']):
                state[f'{side}_arm_joint_state'] = np.zeros(dim, dtype=np.float32)
                state[f'{side}_ee_joint_state'] = np.zeros(1, dtype=np.float32)
            obs = {'instruction': 'Check the policy connection.', 'state': state,
                   'vision': {camera: {'color': np.zeros((64, 64, 3), dtype=np.uint8)}
                              for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}}
            for mode in ('joint', 'ee'):
                model = Model({'env_cfg_type': robot, 'action_type': mode})
                try:
                    Model({'env_cfg_type': robot, 'action_type': 'invalid'})
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid action type accepted')
                try:
                    model.get_action_batch([99])
                except ValueError:
                    pass
                else:
                    raise AssertionError('Unknown environment accepted')
                server = PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=0))
                await server.start()

                def exercise():
                    with WsModelClient(url=server.url, evaluation_id='demo',
                                       trial_id=f'{robot}-{mode}') as client:
                        client.call('reset')
                        client.call('update_obs', obs=obs)
                        actions = client.call('get_action')
                        check_actions(actions, dimensions['arm_dim'], mode)
                        client.call('update_obs_batch', obs=[dict(obs, env_idx=4), dict(obs, env_idx=7)])
                        batch = client.call('get_action_batch', obs=[7, 4])
                        assert len(batch) == 2
                        assert len(client.call('get_action_batch')) == 2
                        assert len(client.call('get_action_batch', obs=[7])) == 1
                        assert client.call('get_action_batch', obs=[]) == []
                        for chunk in batch:
                            check_actions(chunk, dimensions['arm_dim'], mode)
                        client.call('reset')
                        assert len(client.call('get_action_batch')) == 1
                        check_actions(client.call('get_action'), dimensions['arm_dim'], mode)
                        print(f'PASS {robot}/{mode}: observations, actions, batch, reset')
                try:
                    await asyncio.to_thread(exercise)
                finally:
                    await server.stop()
    print('Six wiring checks passed. No learned inference, simulation, or hardware was run.')


if __name__ == '__main__':
    asyncio.run(main())
