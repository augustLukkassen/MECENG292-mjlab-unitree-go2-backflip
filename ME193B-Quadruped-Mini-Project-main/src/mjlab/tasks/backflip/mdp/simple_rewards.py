"""Rewards for backflip task."""

from __future__ import annotations

import math
import os
import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Reference trajectory for motion imitation (optional)
_REF_DATA_PATH = os.path.join(os.path.dirname(__file__), "backflip_reference_phase.npz")
_REF_DATA = None
_REF_POSITIONS = None
_REF_POSITIONS_DEVICE = {}


def _load_reference():
  global _REF_DATA, _REF_POSITIONS
  if _REF_DATA is None and os.path.exists(_REF_DATA_PATH):
    _REF_DATA = np.load(_REF_DATA_PATH)
    _REF_POSITIONS = torch.from_numpy(_REF_DATA["positions"]).float()


def _get_ref_positions(device):
  _load_reference()
  if _REF_POSITIONS is None:
    return None
  if device not in _REF_POSITIONS_DEVICE:
    _REF_POSITIONS_DEVICE[device] = _REF_POSITIONS.to(device)
  return _REF_POSITIONS_DEVICE[device]


# === Trajectory Tracking ===

def track_height_simple(env, command_name, std=0.1, asset_cfg=_DEFAULT_ASSET_CFG):
  """Gaussian reward for tracking target height."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.square(asset.data.root_link_pos_w[:, 2] - command[:, 1])
  return torch.exp(-error / (std**2))


def track_x_simple(env, command_name, std=0.15, asset_cfg=_DEFAULT_ASSET_CFG):
  """Gaussian reward for tracking target x-position."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.square(asset.data.root_link_pos_w[:, 0] - command[:, 4])
  return torch.exp(-error / (std**2))


def track_orientation_simple(env, command_name, std=0.3, asset_cfg=_DEFAULT_ASSET_CFG):
  """Gaussian reward for tracking target orientation via projected gravity."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  grav = asset.data.projected_gravity_b
  error = torch.square(grav[:, 0] - command[:, 2]) + torch.square(grav[:, 2] - command[:, 3])
  return torch.exp(-error / (std**2))


# === Progress Rewards ===

def backflip_progress_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward based on phase progress (0→1)."""
  command_term = env.command_manager.get_term(command_name)
  return command_term.metrics["max_rotation_progress"]


def max_height_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward for maximum height achieved (normalized 0→1)."""
  command_term = env.command_manager.get_term(command_name)
  standing = command_term.cfg.standing_height
  peak = command_term.cfg.peak_height
  return torch.clamp((command_term.metrics["max_height"] - standing) / (peak - standing), 0, 1)


# === Phase-Gated Rewards ===

def crouch_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward for crouching during phase 0→0.15."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]
  height = asset.data.root_link_pos_w[:, 2]

  crouch_amount = torch.clamp((0.35 - height) / (0.35 - 0.18), 0, 1)
  return torch.where(phase < 0.15, crouch_amount, torch.zeros_like(crouch_amount))


def upward_velocity_reward(env, command_name, scale=2.0, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward upward velocity during takeoff phase (0.15→0.40)."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]
  z_vel = asset.data.root_link_lin_vel_w[:, 2]

  in_takeoff = (phase >= 0.15) & (phase < 0.40)
  reward = torch.clamp(z_vel / scale, 0, 1)
  return torch.where(in_takeoff, reward, torch.zeros_like(reward))


def backward_takeoff_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward backward pitch rate during takeoff (0.10→0.35)."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]
  pitch_rate = asset.data.root_link_ang_vel_b[:, 1]

  in_takeoff = (phase >= 0.10) & (phase < 0.35)
  reward = torch.clamp(-pitch_rate / 5.0, 0, 1)
  return torch.where(in_takeoff, reward, torch.zeros_like(reward))


def airborne_bonus(env, sensor_name, min_height=0.40, asset_cfg=_DEFAULT_ASSET_CFG):
  """Bonus for being airborne with body elevated above min_height."""
  asset = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]
  feet_airborne = ~(sensor.data.found > 0).any(dim=-1)
  height_ok = asset.data.root_link_pos_w[:, 2] > min_height
  return (feet_airborne & height_ok).float()


def pitch_velocity_reward(env, sensor_name, scale=10.0, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward backward pitch rate while airborne."""
  asset = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]
  is_airborne = ~(sensor.data.found > 0).any(dim=-1)
  pitch_rate = asset.data.root_link_ang_vel_b[:, 1]

  reward = torch.clamp(-pitch_rate / scale, 0, 1)
  return torch.where(is_airborne, reward, torch.zeros_like(reward))


def landing_preparation_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward leg extension during descent phase (0.60→0.90)."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]

  in_descent = (phase >= 0.60) & (phase < 0.90)
  calf_positions = asset.data.joint_pos[:, [2, 5, 8, 11]]
  avg_calf = calf_positions.mean(dim=1)
  extension = torch.clamp((avg_calf + 2.6) / 1.6, 0, 1)

  return torch.where(in_descent, extension, torch.zeros_like(extension))


def stand_up_reward(env, command_name, standing_height=0.35, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward being upright and tall during landing phase (>0.85)."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]

  in_landing = phase >= 0.85
  grav_z = asset.data.projected_gravity_b[:, 2]
  height = asset.data.root_link_pos_w[:, 2]

  upright = torch.clamp((-grav_z - 0.5) / 0.5, 0, 1)
  tall = torch.clamp(height / standing_height, 0, 1)
  reward = upright * tall

  return torch.where(in_landing, reward, torch.zeros_like(reward))


def leg_extension_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward leg extension during takeoff phase (0.15→0.35)."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  phase = command[:, 0]

  in_takeoff = (phase >= 0.15) & (phase < 0.35)
  calf_positions = asset.data.joint_pos[:, [2, 5, 8, 11]]
  avg_calf = calf_positions.mean(dim=1)
  extension = torch.clamp((avg_calf + 2.6) / 2.3, 0, 1)

  return torch.where(in_takeoff, extension, torch.zeros_like(extension))


# === Regularization ===

def action_rate_simple(env):
  """Penalize rapid action changes."""
  action = env.action_manager.action
  prev = env.action_manager.prev_action
  return torch.sum(torch.square(action - prev), dim=1)


def off_axis_simple(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """Penalize roll and yaw (want pure pitch rotation)."""
  asset = env.scene[asset_cfg.name]
  roll = torch.abs(asset.data.projected_gravity_b[:, 1])
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
  return roll + 0.5 * yaw_rate


def joint_velocity_penalty(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """Penalize high joint velocities."""
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, :12]), dim=1)


# === Motion Imitation (optional) ===

def motion_imitation_reward(env, command_name, std=0.5, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward for matching reference joint trajectory."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  device = command.device

  ref_positions = _get_ref_positions(device)
  if ref_positions is None:
    return torch.zeros(env.num_envs, device=device)

  phase = command[:, 0]
  n_phases = ref_positions.shape[0]
  phase_idx = torch.clamp((phase * (n_phases - 1)).long(), 0, n_phases - 1)

  ref_joint_pos = ref_positions[phase_idx]
  current_joint_pos = asset.data.joint_pos[:, :12]
  error = torch.sum(torch.square(current_joint_pos - ref_joint_pos), dim=1)

  return torch.exp(-error / (std**2))
