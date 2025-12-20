from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math 

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)

def track_phase_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base height.
  
  Only active during phase 0-0.8 (jump and flip).
  Disabled during phase 0.8-1.0 to avoid rewarding standing still.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  phase = command[:, 0]

  # Only active during first 80% of phase
  active = (phase < 0.8).float()

  base_height = 0.3
  jump_height = 1.0  # Peak = 1.3m - more air time to complete flip!
  target_height = base_height + jump_height * torch.sin(phase*math.pi) 

  actual = asset.data.root_link_pos_w[:, 2]
  height_error = torch.square(actual - target_height)

  return torch.exp(-height_error / std**2) * active

def track_phase_pitch(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking rotation progress using projected gravity.
  
  Uses projected gravity Z component which is more robust than Euler angles.
  - Upright: proj_grav_z = -1
  - Inverted (half flip): proj_grav_z = +1
  - Back to upright: proj_grav_z = -1
  
  Target: -cos(phase * 2π) follows this pattern exactly.
  Only active during phase < 0.8 to avoid rewarding standing still.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  phase = command[:, 0]
  
  # Only active during first 80% of phase (avoid rewarding standing still at end)
  active = (phase < 0.8).float()
  
  # Target Z component of projected gravity
  # phase=0: -cos(0) = -1 (upright)
  # phase=0.5: -cos(π) = +1 (inverted)
  # phase=0.8: -cos(1.6π) ≈ -0.81 (almost upright again)
  target_grav_z = -torch.cos(phase * 2 * math.pi)
  
  # Actual projected gravity Z component
  actual_grav_z = asset.data.projected_gravity_b[:, 2]
  
  # Normalize (gravity magnitude should be ~1 but let's be safe)
  grav_norm = torch.norm(asset.data.projected_gravity_b, dim=-1, keepdim=False)
  actual_grav_z = actual_grav_z / torch.clamp(grav_norm, min=0.1)
  
  error = torch.square(target_grav_z - actual_grav_z)
  return torch.exp(-error / std**2) * active

def landing_upright(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for landing upright."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  phase = command[:, 0]

  phase_weight = torch.clamp((phase-0.8)/0.2, 0.0, 1.0)

  proj_gravity = asset.data.projected_gravity_b
  proj_gravity_normalized = proj_gravity / torch.norm(proj_gravity, dim=-1, keepdim=True)
  proj_gravity_error = torch.square(proj_gravity_normalized[:, 2] + 1.0)

  actual_height = asset.data.root_link_pos_w[:, 2]
  height_error = torch.square(actual_height - 0.3)
  
  total_error = proj_gravity_error + height_error
  return phase_weight * torch.exp(-total_error / std**2)


def track_pitch_velocity(
  env: ManagerBasedRlEnv,
  target_velocity: float,
  std: float,
  command_name: str,
  axis: int = 1,  # 0=roll(X), 1=pitch(Y), 2=yaw(Z)
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for spinning at target pitch velocity (backward flip).
  
  Uses WORLD-FRAME angular velocity. Simple Gaussian reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  phase = command[:, 0]
  
  # Only reward velocity during the flip, not during landing
  phase_weight = torch.clamp(1.0 - (phase - 0.7) / 0.3, 0.0, 1.0)
  
  # Use WORLD-FRAME angular velocity
  actual_vel = asset.data.root_link_ang_vel_w[:, axis]
  
  # Simple Gaussian reward - let robot explore both directions
  error = torch.square(actual_vel - target_velocity)
  reward = torch.exp(-error / std**2)
  
  return phase_weight * reward


def simple_pitch_velocity(
  env: ManagerBasedRlEnv,
  min_height: float = 0.5,  # Must be airborne to get rotation reward!
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward negative Y angular velocity (nose UP for backflip).
  
  Only rewards rotation when robot is above min_height (must jump first!).
  This prevents the robot from just tipping over on the ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Height gate - must be airborne to get rotation reward
  height = asset.data.root_link_pos_w[:, 2]
  height_gate = (height > min_height).float()
  
  # World Y angular velocity - negative = nose UP
  pitch_vel = asset.data.root_link_ang_vel_w[:, 1]
  
  # Reward negative velocity (nose UP), only when airborne
  return torch.clamp(-pitch_vel, min=0.0, max=10.0) * height_gate


def vertical_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward upward velocity during jump phase (phase < 0.5)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  phase = command[:, 0]
  
  # Only reward upward velocity during first half (jump phase)
  jump_phase = (phase < 0.5).float()
  
  # Vertical velocity (positive = up)
  vert_vel = asset.data.root_link_lin_vel_w[:, 2]
  
  # Reward upward velocity
  return torch.clamp(vert_vel, min=0.0, max=5.0) * jump_phase


def penalize_yaw_roll(
  env: ManagerBasedRlEnv,
  pitch_axis: int = 1,  # Which axis is the backflip axis (exclude from penalty)
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize rotation around non-pitch axes in WORLD frame."""
  asset: Entity = env.scene[asset_cfg.name]
  # Use WORLD-FRAME angular velocity
  ang_vel = asset.data.root_link_ang_vel_w  # [B, 3]
  
  # Penalize all axes except the pitch axis
  penalty = torch.zeros(ang_vel.shape[0], device=ang_vel.device)
  for i in range(3):
    if i != pitch_axis:
      penalty += torch.square(ang_vel[:, i])
  
  return penalty


def penalize_wrong_pitch(
  env: ManagerBasedRlEnv,
  axis: int = 1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize positive pitch velocity (wrong direction for backflip).
  
  For this robot, backflip = NEGATIVE Y rotation.
  This penalizes any positive pitch velocity.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pitch_vel = asset.data.root_link_ang_vel_w[:, axis]
  
  # Only penalize positive velocity (wrong direction)
  wrong_direction = torch.clamp(pitch_vel, min=0.0)  # Positive when pitch_vel is positive
  return torch.square(wrong_direction)


def takeoff_impulse(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward explosive takeoff: upward velocity + backward pitch together.
  
  Only active during early phase (< 0.3) to encourage proper launch.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  phase = command[:, 0]
  
  # Only during early phase
  early_phase = (phase < 0.3).float()
  
  # Upward velocity (positive = up)
  upward_vel = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0, max=5.0)
  
  # Backward pitch velocity (negative Y = nose up)
  backward_pitch = torch.clamp(-asset.data.root_link_ang_vel_b[:, 1], min=0.0, max=8.0)
  
  # Bonus for doing BOTH together
  combo_bonus = upward_vel * backward_pitch * 0.1
  
  return (upward_vel + backward_pitch + combo_bonus) * early_phase


def inverted_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for progress toward inverted - NO phase restriction.
  
  Projected gravity Z: -1 (upright) -> 0 (vertical) -> +1 (inverted)
  Reward = (grav_z + 1) / 2:
    - Upright: 0.0
    - Vertical: 0.5  
    - Inverted: 1.0
  Active anytime - directly rewards being more tilted.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Projected gravity Z: -1 upright, 0 vertical, +1 inverted
  grav_z = asset.data.projected_gravity_b[:, 2]
  
  # Smooth reward: 0 at upright, 0.5 at vertical, 1.0 at inverted
  progress = (grav_z + 1.0) / 2.0
  
  return progress


def legs_forward(
  env: ManagerBasedRlEnv,
  min_height: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward swinging legs forward when airborne.
  
  For backflip, legs should swing forward (NEGATIVE thigh velocity) 
  to help carry the rotation. Only rewards when above min_height.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Height gate - only when airborne
  height = asset.data.root_link_pos_w[:, 2]
  airborne = (height > min_height).float()
  
  # Get thigh joint velocities
  joint_vel = asset.data.joint_vel  # [B, num_joints]
  
  # Thigh joint indices: FR, FL, RR, RL
  thigh_indices = [1, 4, 7, 10]
  thigh_vel = joint_vel[:, thigh_indices].mean(dim=1)
  
  # Reward NEGATIVE velocity (legs swinging forward), clamped
  forward_reward = torch.clamp(-thigh_vel, min=0.0, max=5.0)
  
  return forward_reward * airborne


def default_joint_position(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  asset: Entity = env.scene[asset_cfg.name]
  current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  desired_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  error_squared = torch.square(current_joint_pos - desired_joint_pos)
  return torch.sum(torch.abs(current_joint_pos - desired_joint_pos), dim=1)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def base_z(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

    # Use root link projected gravity.
  z_error = torch.square(asset.data.root_link_pos_w[:, 2] - 0.3)
  return torch.exp(-z_error / std**2)



def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose, with tighter constraints when standing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
