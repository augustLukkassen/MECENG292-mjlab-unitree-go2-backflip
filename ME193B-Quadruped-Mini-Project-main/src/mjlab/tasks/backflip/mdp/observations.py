from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def backflip_phase(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 0:1]


def backflip_target_height(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 1:2]


def backflip_target_pitch(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 2:3]


def backflip_target_pitch_vel(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 3:4]


def backflip_target_x(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 5:6]


def backflip_target_grav_x(env, command_name):
  """Target gravity x-component (for simple backflip command)."""
  command = env.command_manager.get_command(command_name)
  return command[:, 2:3]


def backflip_target_grav_z(env, command_name):
  """Target gravity z-component (for simple backflip command)."""
  command = env.command_manager.get_command(command_name)
  return command[:, 3:4]


def simple_backflip_target_x(env, command_name):
  """Target x-position (for simple backflip command)."""
  command = env.command_manager.get_command(command_name)
  return command[:, 4:5]


def base_x(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """Current x-position of the robot base."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 0:1]


def base_height(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def base_pitch(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  proj_grav = asset.data.projected_gravity_b
  pitch = torch.asin(-proj_grav[:, 0:1])
  return pitch


def base_pitch_rate(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, 1:2]


def base_orientation_quat(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_quat_w


def base_lin_vel_vertical(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w[:, 2:3]


def height_error(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  current_height = asset.data.root_link_pos_w[:, 2:3]
  target_height = command[:, 1:2]
  return target_height - current_height


def x_error(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  current_x = asset.data.root_link_pos_w[:, 0:1]
  target_x = command[:, 5:6]
  return target_x - current_x


def cumulative_pitch(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, 1:2]


# Privileged critic observations (foot-related)

def foot_height(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """Height of each foot site relative to world frame."""
  asset = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_air_time(env, sensor_name):
  """Time each foot has been in the air."""
  sensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env, sensor_name):
  """Binary contact state for each foot."""
  sensor = env.scene[sensor_name]
  return (sensor.data.found > 0).float()


def all_feet_airborne(env, sensor_name):
  """1.0 if all feet are off ground, 0.0 otherwise."""
  sensor = env.scene[sensor_name]
  any_contact = (sensor.data.found > 0).any(dim=-1, keepdim=True)
  return (~any_contact).float()


def rotation_progress(env, command_name):
  """Actual rotation progress from command metrics (0 to 1)."""
  command_term = env.command_manager.get_term(command_name)
  return command_term.metrics["max_rotation_progress"].unsqueeze(-1)


def cumulative_pitch_rotation(env, command_name):
  """Cumulative pitch rotation in radians (negative = backflip)."""
  command_term = env.command_manager.get_term(command_name)
  return command_term.cumulative_pitch.unsqueeze(-1)


def foot_contact_forces(env, sensor_name):
  """Contact forces on each foot (log-scaled)."""
  sensor = env.scene[sensor_name]
  forces_flat = sensor.data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
