from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import UNITREE_GO2_BACKFLIP_ENV_CFG
from .rl_cfg import UNITREE_GO2_PPO_CFG

register_mjlab_task(
  task_id="Mjlab-Backflip-Go2",
  env_cfg=UNITREE_GO2_BACKFLIP_ENV_CFG,
  rl_cfg=UNITREE_GO2_PPO_CFG,
  runner_cls=None,
)


