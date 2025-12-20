"""Termination conditions for backflip task."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def backflip_completed(env, command_name, uprightness_threshold=0.3, min_rotation=1.8 * 3.14159, asset_cfg=_DEFAULT_ASSET_CFG):
  """Terminate (success) when backflip is complete."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase_complete = command[:, 0] >= 0.9
  has_rotated = command[:, 4] <= -min_rotation
  uprightness_error = torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=1)
  is_upright = uprightness_error < uprightness_threshold

  return phase_complete & has_rotated & is_upright


def body_contact_during_flip(env, command_name, sensor_name):
  """Terminate if body contacts ground mid-flip."""
  sensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]

  has_contact = torch.any(sensor.data.found > 0, dim=1)
  mid_flip = (phase > 0.2) & (phase < 0.8)

  return mid_flip & has_contact


def height_too_low(env, command_name, min_height=0.15, asset_cfg=_DEFAULT_ASSET_CFG):
  """Terminate if height drops too low mid-flip."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]

  mid_flip = (phase > 0.15) & (phase < 0.75)
  too_low = asset.data.root_link_pos_w[:, 2] < min_height

  return mid_flip & too_low


def bad_landing_orientation(env, command_name, limit_angle=1.2, asset_cfg=_DEFAULT_ASSET_CFG):
  """Terminate if orientation is bad during landing phase."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]

  angle = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0))
  return (phase > 0.85) & (angle > limit_angle)


def no_progress(env, command_name, min_phase_by_time=0.5, time_threshold=1.0):
  """Terminate if not making progress."""
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]
  episode_time = env.episode_length_buf * env.step_dt
  expected = episode_time / time_threshold * min_phase_by_time

  return (episode_time > time_threshold) & (phase < expected * 0.5)


def excessive_rotation(env, max_roll=1.0, max_yaw_rate=5.0, asset_cfg=_DEFAULT_ASSET_CFG):
  """Terminate if excessive roll or yaw."""
  asset = env.scene[asset_cfg.name]
  roll = torch.abs(asset.data.projected_gravity_b[:, 1])
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
  return (roll > max_roll) | (yaw_rate > max_yaw_rate)


def failed_takeoff(env, command_name, check_phase=0.45, min_height=0.50):
  """Terminate if robot hasn't jumped high enough by check_phase."""
  command_term = env.command_manager.get_term(command_name)
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  max_height = command_term.metrics["max_height"]

  return (phase >= check_phase) & (max_height < min_height)


def landed_upside_down(env, sensor_name, min_height=0.25, asset_cfg=_DEFAULT_ASSET_CFG):
  """Terminate if robot lands inverted (grav_z > 0.5)."""
  asset = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]

  height = asset.data.root_link_pos_w[:, 2]
  any_contact = (sensor.data.found > 0).any(dim=-1)
  on_ground = (height < min_height) | any_contact

  grav_z = asset.data.projected_gravity_b[:, 2]
  is_inverted = grav_z > 0.5

  return on_ground & is_inverted


def insufficient_rotation(env, command_name, check_phase=0.65, min_rotation_progress=0.5):
  """Terminate if rotation progress is below threshold at check_phase."""
  command_term = env.command_manager.get_term(command_name)
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  progress = command_term.metrics["max_rotation_progress"]

  return (phase >= check_phase) & (progress < min_rotation_progress)


def wrong_direction_takeoff(env, command_name, check_phase=0.35, max_forward_pitch=0.3):
  """Terminate if pitching forward (frontflip) instead of backward."""
  command_term = env.command_manager.get_term(command_name)
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  cumulative_pitch = command_term.cumulative_pitch

  return (phase >= check_phase) & (cumulative_pitch > max_forward_pitch)


def bad_landing(env, sensor_name, command_name, min_uprightness=-0.7, min_height=0.20, asset_cfg=_DEFAULT_ASSET_CFG):
  """Terminate if robot lands poorly (not upright or body too low)."""
  asset = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  in_landing = phase >= 0.9

  any_contact = (sensor.data.found > 0).any(dim=-1)
  height = asset.data.root_link_pos_w[:, 2]
  on_ground = any_contact | (height < min_height)

  grav_z = asset.data.projected_gravity_b[:, 2]
  not_upright = grav_z > min_uprightness

  return in_landing & on_ground & not_upright
