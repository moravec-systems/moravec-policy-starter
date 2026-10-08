"""Synthetic actions for offline client/server wiring; never drive hardware."""

import numpy as np

from moravec_policy.model_template import ModelTemplate
from moravec_policy.utils.process_data import get_robot_action_dim_info


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.action_type = model_cfg['action_type']
        if self.action_type not in {'joint', 'ee'}:
            raise ValueError('action_type must be joint or ee')
        self.dimensions = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        arms = self.dimensions['arm_dim']
        grippers = self.dimensions['ee_dim']
        if len(arms) not in (1, 2) or len(arms) != len(grippers):
            raise ValueError('Expected matching arm/gripper dimensions for one or two arms')
        prefixes = [''] if len(arms) == 1 else ['left_', 'right_']
        self.channels = list(zip(prefixes, arms, grippers))
        self.reset()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        # A real policy would preprocess/store the images and state here.
        indices = [obs.get('env_idx', index) for index, obs in enumerate(obs_list)]
        if len(set(indices)) != len(indices):
            raise ValueError('Observation environment indices must be unique')
        self.env_indices = indices

    def get_action(self):
        # Replace this synthetic chunk with model inference and action conversion.
        chunk = []
        for _ in range(2):
            action = {}
            for prefix, arm_dim, gripper_dim in self.channels:
                if self.action_type == 'joint':
                    action[prefix + 'arm_joint_state'] = np.zeros(arm_dim, dtype=np.float32)
                else:
                    action[prefix + 'ee_pose'] = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
                action[prefix + 'ee_joint_state'] = np.zeros(gripper_dim, dtype=np.float32)
            chunk.append(action)
        return chunk

    def get_action_batch(self, env_idx_list=None):
        indices = self.env_indices if env_idx_list is None else env_idx_list
        if any(index not in self.env_indices for index in indices):
            raise ValueError('Update observations for each requested environment first')
        return [self.get_action() for _ in indices]

    def reset(self):
        self.env_indices = [0]
