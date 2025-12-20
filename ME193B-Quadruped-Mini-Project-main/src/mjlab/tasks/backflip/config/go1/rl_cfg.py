"""RL configuration for Go1 backflip task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

UNITREE_GO1_BACKFLIP_RL_CFG = RslRlOnPolicyRunnerCfg(
  policy=RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=(256, 256, 128),
    critic_hidden_dims=(256, 256, 128),
    activation="elu",
  ),
  algorithm=RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  ),
  experiment_name="backflip_go1",
  save_interval=50,
  num_steps_per_env=24,
  max_iterations=1000,
)
