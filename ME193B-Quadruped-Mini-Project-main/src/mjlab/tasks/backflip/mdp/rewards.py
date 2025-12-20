from __future__ import annotations

import math

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_height(env, command_name, std, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  current_height = asset.data.root_link_pos_w[:, 2]
  target_height = command[:, 1]
  height_error = torch.square(current_height - target_height)

  return torch.exp(-height_error / (std**2))


def track_x_position(env, command_name, std, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  current_x = asset.data.root_link_pos_w[:, 0]
  target_x = command[:, 5]
  x_error = torch.square(current_x - target_x)

  return torch.exp(-x_error / (std**2))


def track_pitch_velocity(env, command_name, std, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  current_pitch_vel = asset.data.root_link_ang_vel_b[:, 1]
  target_pitch_vel = command[:, 3]

  pitch_vel_error = torch.square(current_pitch_vel - target_pitch_vel)
  return torch.exp(-pitch_vel_error / (std**2))


def phase_progress(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 0]


def backflip_completion_bonus(env, command_name, bonus=10.0, min_rotation=1.8 * math.pi):
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  cumulative_rotation = command[:, 4]

  phase_done = phase >= 0.9
  has_rotated = cumulative_rotation <= -min_rotation
  completed = (phase_done & has_rotated).float()

  return completed * bonus


def upright_at_landing(env, command_name, std, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  projected_gravity = asset.data.projected_gravity_b

  uprightness_error = torch.sum(projected_gravity[:, :2] ** 2, dim=1)
  landing_mask = (phase > 0.8).float()

  reward = torch.exp(-uprightness_error / (std**2))
  return reward * landing_mask


def rotation_consistency(env, command_name=None, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward maintaining backward pitch velocity during flip phase."""
  asset = env.scene[asset_cfg.name]
  pitch_vel = asset.data.root_link_ang_vel_b[:, 1]
  # Need ~6 rad/s to complete flip in 1 second
  rotation_reward = torch.clamp(-pitch_vel / 6.0, 0.0, 1.0)

  # If command_name provided, only apply during flip phase
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    phase = command[:, 0]
    flip_mask = ((phase >= 0.1) & (phase < 0.9)).float()
    return rotation_reward * flip_mask

  return rotation_reward


def insufficient_rotation_penalty(env, command_name, min_pitch_vel=3.0, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  pitch_vel = asset.data.root_link_ang_vel_b[:, 1]

  mid_flip = ((phase > 0.1) & (phase < 0.8)).float()
  too_slow = (pitch_vel > -min_pitch_vel).float()

  return mid_flip * too_slow


def air_time_during_flip(env, command_name, sensor_name):
  sensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  cumulative_rotation = command[:, 4]  # Track actual rotation progress
  in_contact = torch.any(sensor.data.found > 0, dim=1)

  # Only reward if actually rotating (at least 0.5 rad of rotation)
  has_rotation = (cumulative_rotation < -0.5).float()
  mid_flip_mask = ((phase > 0.2) & (phase < 0.8)).float()
  airborne = (~in_contact).float()

  return mid_flip_mask * airborne * has_rotation


def foot_height_during_flip(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward feet being high off the ground during flip phase."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  # Get foot heights (z position of foot sites)
  foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_feet)
  avg_foot_height = torch.mean(foot_heights, dim=1)

  # Only reward during flip phase (after crouch)
  flip_mask = (phase >= 0.1).float()

  # Reward feet being higher (normalized, max reward at ~0.5m)
  height_reward = torch.clamp(avg_foot_height / 0.5, 0.0, 1.0)

  return flip_mask * height_reward


def ground_contact_at_landing(env, command_name, sensor_name):
  sensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  in_contact = torch.any(sensor.data.found > 0, dim=1)

  landing_mask = (phase > 0.85).float()
  return landing_mask * in_contact.float()


def takeoff_impulse(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward exploding backwards: upward velocity AND backward pitch velocity together."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  vertical_vel = asset.data.root_link_lin_vel_w[:, 2]
  pitch_vel = asset.data.root_link_ang_vel_b[:, 1]  # negative = backward rotation
  backward_vel = -asset.data.root_link_lin_vel_w[:, 0]  # negative x = backward

  # Active after crouch (phi > 0.1) through early flip (phi < 0.5)
  takeoff_mask = ((phase >= 0.1) & (phase < 0.5)).float()

  # Reward upward velocity - need 2+ m/s for enough air time
  upward_reward = torch.clamp(vertical_vel / 2.0, 0.0, 1.0)

  # Reward backward pitch velocity - need 6+ rad/s to get past vertical
  # (2π rad / 1s flip = 6.28 rad/s needed on average)
  rotation_reward = torch.clamp(-pitch_vel / 6.0, 0.0, 1.0)

  # Reward backward linear velocity
  backward_reward = torch.clamp(backward_vel / 1.0, 0.0, 1.0)

  # Additive structure: reward each component, with bonus for combination
  base_reward = 0.3 * upward_reward + 0.5 * rotation_reward + 0.2 * backward_reward
  # Bonus for having both upward AND rotation together
  combo_bonus = upward_reward * rotation_reward

  return takeoff_mask * (base_reward + combo_bonus)


def crouch_incentive(env, standing_height=0.35, crouch_depth=0.25, crouch_steps=30, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  current_height = asset.data.root_link_pos_w[:, 2]

  early_mask = (env.episode_length_buf < crouch_steps).float()

  # Reward proportional to how much we've crouched toward target depth
  crouch_reward = torch.clamp((standing_height - current_height) / crouch_depth, 0.0, 1.0)

  return early_mask * crouch_reward


def smooth_actions(env):
  action = env.action_manager.action
  return torch.sum(torch.square(action), dim=1)


def action_rate_penalty(env):
  action = env.action_manager.action
  last_action = env.action_manager.prev_action
  action_diff = action - last_action
  return torch.sum(torch.square(action_diff), dim=1)


def off_axis_penalty(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  proj_grav = asset.data.projected_gravity_b
  roll_error = torch.abs(proj_grav[:, 1])
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
  # Strong penalty on both roll and yaw - we only want pitch rotation
  return roll_error + 1.0 * yaw_rate


def joint_velocity_penalty(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel
  return torch.sum(torch.square(joint_vel), dim=1)


def joint_torque_penalty(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  joint_torque = asset.data.applied_torque
  if joint_torque is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.square(joint_torque), dim=1)
