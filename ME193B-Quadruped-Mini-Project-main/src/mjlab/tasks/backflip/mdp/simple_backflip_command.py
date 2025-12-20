"""Time-based backflip command generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class SimpleBackflipCommand(CommandTerm):
  """
  Command output: [phase, target_height, target_grav_x, target_grav_z, target_x]

  Trajectory phases:
    0.00-0.15: crouch
    0.15-0.30: takeoff
    0.30-0.70: airborne (rotation)
    0.70-1.00: landing

  Progress metric (achievement-based, 0→1):
    0.0=start, 0.1=crouched, 0.25=airborne, 0.5=inverted, 0.75=270°, 1.0=landed
  """

  cfg: SimpleBackflipCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.asset_name]
    self.env = env
    self._command = torch.zeros(self.num_envs, 5, device=self.device)
    self.dt = env.step_dt
    self.flip_steps = int(cfg.flip_duration / self.dt)

    self.metrics["max_rotation_progress"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["max_height"] = torch.zeros(self.num_envs, device=self.device)

    self.start_x = torch.zeros(self.num_envs, device=self.device)
    self.cumulative_pitch = torch.zeros(self.num_envs, device=self.device)

    self.feet_sensor = None
    if cfg.feet_sensor_name is not None:
      self.feet_sensor = env.scene[cfg.feet_sensor_name]

  @property
  def command(self):
    return self._command

  def _update_metrics(self):
    # Integrate pitch rate (negative = backflip direction)
    pitch_rate = self.robot.data.root_link_ang_vel_b[:, 1]
    self.cumulative_pitch += pitch_rate * self.dt

    # Convert cumulative pitch to rotation fraction (0→1 for full 360°)
    backflip_rotation = torch.clamp(-self.cumulative_pitch, 0.0, 2 * math.pi)
    rotation_frac = backflip_rotation / (2 * math.pi)

    current_height = self.robot.data.root_link_pos_w[:, 2]
    self.metrics["max_height"] = torch.maximum(self.metrics["max_height"], current_height)

    # Compute phase-based progress
    progress = self.metrics["max_rotation_progress"].clone()
    standing = self.cfg.standing_height
    crouch = self.cfg.crouch_height

    # Airborne check
    if self.feet_sensor is not None:
      is_airborne = ~(self.feet_sensor.data.found > 0).any(dim=-1)
    else:
      is_airborne = current_height > standing + 0.15

    # Phase 0→0.1: Crouching
    crouch_frac = torch.clamp((standing - current_height) / (standing - crouch), 0, 1)
    progress = torch.where(progress < 0.1, torch.maximum(progress, 0.1 * crouch_frac), progress)

    # Phase 0.1→0.25: Takeoff
    takeoff_target = standing + 0.5 * (self.cfg.peak_height - standing)
    takeoff_frac = torch.clamp((current_height - crouch) / (takeoff_target - crouch), 0, 1)
    takeoff_progress = torch.where(is_airborne, torch.tensor(0.25, device=self.device), 0.1 + 0.14 * takeoff_frac)
    in_takeoff = (progress >= 0.1) & (progress < 0.25)
    progress = torch.where(in_takeoff, torch.maximum(progress, takeoff_progress), progress)

    # Phase 0.25→0.5: First half of rotation (0°→180°)
    height_bonus = torch.clamp((current_height - standing) / (self.cfg.peak_height - standing), 0, 1)
    credit_scale = 0.5 + 0.5 * height_bonus
    rotation_progress = 0.25 + 0.25 * torch.clamp(rotation_frac / 0.5, 0, 1) * credit_scale
    in_rotation = (progress >= 0.25) & (progress < 0.5) & is_airborne
    progress = torch.where(in_rotation, torch.maximum(progress, rotation_progress), progress)

    # Phase 0.5→0.75: Second half of rotation (180°→270°)
    max_height_bonus = torch.clamp((self.metrics["max_height"] - standing) / (self.cfg.peak_height - standing), 0, 1)
    flip_credit = 0.5 + 0.5 * max_height_bonus
    flip_frac = torch.clamp((rotation_frac - 0.5) / 0.5, 0, 1)
    flip_progress = 0.5 + 0.25 * flip_frac * flip_credit
    in_flip = (progress >= 0.5) & (progress < 0.75) & is_airborne
    progress = torch.where(in_flip, torch.maximum(progress, flip_progress), progress)

    # Phase 0.75→1.0: Landing
    grav_z = self.robot.data.projected_gravity_b[:, 2]
    upright_frac = torch.clamp((-grav_z - 0.5) / 0.5, 0, 1)
    landing_progress = 0.75 + 0.25 * upright_frac
    in_landing = (progress >= 0.75) & (progress < 1.0) & ~is_airborne
    progress = torch.where(in_landing, torch.maximum(progress, landing_progress), progress)

    self.metrics["max_rotation_progress"] = progress

  def _resample_command(self, env_ids):
    self.metrics["max_rotation_progress"][env_ids] = 0.0
    self.metrics["max_height"][env_ids] = 0.0
    self.cumulative_pitch[env_ids] = 0.0
    self.start_x[env_ids] = self.robot.data.root_link_pos_w[env_ids, 0]
    self._update_reference(env_ids)

  def _update_command(self):
    self._update_reference(torch.arange(self.num_envs, device=self.device))

  def _update_reference(self, env_ids):
    time_steps = self.env.episode_length_buf[env_ids].float()
    phase = torch.clamp(time_steps / self.flip_steps, 0, 1)

    self._command[env_ids, 0] = phase
    self._command[env_ids, 1] = self._height_trajectory(phase)
    grav_x, grav_z = self._orientation_trajectory(phase)
    self._command[env_ids, 2] = grav_x
    self._command[env_ids, 3] = grav_z
    self._command[env_ids, 4] = self.start_x[env_ids] + self._x_trajectory(phase)

  def _height_trajectory(self, phase):
    standing, crouch, peak = self.cfg.standing_height, self.cfg.crouch_height, self.cfg.peak_height
    height = torch.zeros_like(phase)

    # Crouch
    m1 = phase < 0.15
    height = torch.where(m1, standing - (standing - crouch) * (phase / 0.15), height)

    # Takeoff
    m2 = (phase >= 0.15) & (phase < 0.30)
    t2 = (phase - 0.15) / 0.15
    height = torch.where(m2, crouch + (peak - crouch) * 0.8 * t2, height)

    # Airborne (parabolic arc)
    m3 = (phase >= 0.30) & (phase < 0.70)
    t3 = (phase - 0.30) / 0.40
    base = crouch + (peak - crouch) * 0.8
    height = torch.where(m3, base + (peak - base) * torch.sin(math.pi * t3), height)

    # Landing
    m4 = phase >= 0.70
    t4 = (phase - 0.70) / 0.30
    height = torch.where(m4, base * (1 - t4) + standing * t4, height)

    return height

  def _orientation_trajectory(self, phase):
    # Full 360° rotation between phase 0.15 and 0.70
    rotation_phase = torch.clamp((phase - 0.15) / 0.55, 0, 1)
    angle = 2 * math.pi * rotation_phase
    return -torch.sin(angle), -torch.cos(angle)

  def _x_trajectory(self, phase):
    """Robot moves backward during flip, returns near start."""
    backward = self.cfg.backward_displacement
    x = torch.zeros_like(phase)

    # Move backward (0.15→0.50)
    m2 = (phase >= 0.15) & (phase < 0.50)
    t2 = (phase - 0.15) / 0.35
    x = torch.where(m2, -backward * torch.sin(math.pi / 2 * t2), x)

    # Return toward start (0.50→1.00)
    m3 = phase >= 0.50
    t3 = (phase - 0.50) / 0.50
    x = torch.where(m3, -backward * torch.cos(math.pi / 2 * t3), x)

    return x

  def _debug_vis_impl(self, visualizer):
    idx = visualizer.env_idx
    if idx >= self.num_envs:
      return
    pos = self.robot.data.root_link_pos_w[idx].cpu().numpy()
    pos[2] = self._command[idx, 1].item()
    visualizer.add_sphere(pos, radius=0.05, color=(0.2, 0.8, 0.2, 0.5))


@dataclass(kw_only=True)
class SimpleBackflipCommandCfg(CommandTermCfg):
  asset_name: str
  feet_sensor_name: str | None = None
  flip_duration: float = 1.0
  standing_height: float = 0.35
  crouch_height: float = 0.18
  peak_height: float = 0.90
  backward_displacement: float = 0.3
  class_type: type = SimpleBackflipCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
