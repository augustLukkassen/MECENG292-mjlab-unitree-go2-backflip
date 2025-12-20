from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import UNITREE_GO2_BACKFLIP_ENV_CFG
from .simple_env_cfg import SIMPLE_GO2_BACKFLIP_ENV_CFG
from .retarget_env_cfg import RETARGET_GO2_BACKFLIP_ENV_CFG
from .rl_cfg import UNITREE_GO2_BACKFLIP_RL_CFG

register_mjlab_task(
    task_id="Mjlab-Backflip-Unitree-Go2",
    env_cfg=UNITREE_GO2_BACKFLIP_ENV_CFG,
    rl_cfg=UNITREE_GO2_BACKFLIP_RL_CFG,
)

register_mjlab_task(
    task_id="Mjlab-Backflip-Unitree-Go2-Simple",
    env_cfg=SIMPLE_GO2_BACKFLIP_ENV_CFG,
    rl_cfg=UNITREE_GO2_BACKFLIP_RL_CFG,
)

register_mjlab_task(
    task_id="Mjlab-Backflip-Unitree-Go2-Retarget",
    env_cfg=RETARGET_GO2_BACKFLIP_ENV_CFG,
    rl_cfg=UNITREE_GO2_BACKFLIP_RL_CFG,
)
