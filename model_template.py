class ModelTemplate:
    """
    Base template for a policy/model class.

    This template defines the minimal interface required for:
    1. loading or building a model
    2. updating observations from the environment
    3. generating actions
    4. resetting internal states between episodes

    Notes:
    - You should override the methods in your own subclass or implementation.
    - `self.model` can be a neural network, a planner, or any policy object.
    """

    RED = "\033[31m"
    RESET = "\033[0m"

    def __init__(self):
        """
        Initialize the template model.

        Attributes:
            self.model:
                The actual model/policy object.
        """
        self.model = None

    def _error_msg(self, msg: str) -> str:
        return f"{self.RED}[ERROR] {msg}{self.RESET}"

    def update_obs(self, obs):
        """
        Update the current observation used by the model.

        The policy server decodes camera colors before calling this, so
        obs["vision"][<camera>]["color"] is always an image array.
        """
        raise NotImplementedError(
            self._error_msg("update_obs() must be implemented by the user.")
        )

    def update_obs_batch(self, obs_list):
        """
        Update the current observation used by the model.

        Same decoding guarantee as update_obs().
        """
        raise NotImplementedError(
            self._error_msg("update_obs_batch() must be implemented by the user.")
        )

    def get_action(self):
        """
        Predict or generate an action from the current observation/state.
        """
        raise NotImplementedError(
            self._error_msg("get_action() must be implemented by the user.")
        )

    def get_action_batch(self, env_idx_list=None):
        """
        Predict or generate an action from the current observation/state.
        """
        raise NotImplementedError(
            self._error_msg("get_action_batch() must be implemented by the user.")
        )

    def reset(self):
        """
        Reset the internal state of the model at the start of a new episode.
        """
        pass

    def prepare_case(self, case_meta=None):
        """
        Optional hook called before an action case starts.
        """
        pass

    def on_trial_end(self, result=None):
        """
        Optional hook called after a trial finishes.
        """
        pass
