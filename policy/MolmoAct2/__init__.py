"""MolmoAct2 policy entry points; model dependencies load when requested."""
from .deploy import eval_one_episode, eval_one_episode_batch


def get_model(deploy_cfg):
    from .model import Model

    return Model(deploy_cfg)
