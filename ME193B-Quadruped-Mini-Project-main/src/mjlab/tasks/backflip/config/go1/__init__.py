from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import UNITREE_GO1_BACKFLIP_ENV_CFG
from .rl_cfg import UNITREE_GO1_BACKFLIP_RL_CFG

register_mjlab_task(
    task_id="Mjlab-Backflip-Unitree-Go1",
    env_cfg=UNITREE_GO1_BACKFLIP_ENV_CFG,
    rl_cfg=UNITREE_GO1_BACKFLIP_RL_CFG,
)
