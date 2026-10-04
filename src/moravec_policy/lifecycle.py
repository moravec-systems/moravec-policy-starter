"""Shared single-session lifecycle. Batching is explicitly unsupported by this starter."""


class Policy:
    def configure(self, profile):
        self.profile = profile
        self.reset()

    def reset(self):
        self.observation = None
        self._reset()

    def _reset(self):
        pass

    def update_obs(self, obs):
        self.observation = obs

    def get_action(self):
        if self.observation is None:
            raise ValueError("update_obs required before get_action")
        return self._infer(self.observation, self.profile)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("batching unsupported; use isolated replicas")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("batching unsupported; use isolated replicas")
