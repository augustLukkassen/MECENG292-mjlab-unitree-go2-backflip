from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class BackflipPhaseCommand(CommandTerm):
  """
  Backflip trajectory with three phases:
    1. Crouch (time-based): lower height to wind up
    2. Flip (rotation-based): explode backwards and rotate
    3. Land (rotation-based): complete rotation upright
  """

  cfg: BackflipPhaseCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)

    self.robot = env.scene[cfg.asset_name]
    self.env = env
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self._command = torch.zeros(self.num_envs, 6, device=self.device)
    self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
    self.prev_pitch = torch.zeros(self.num_envs, device=self.device)
    self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.initial_x = torch.zeros(self.num_envs, device=self.device)

    self.metrics["max_phase_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completed"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["cumulative_rotation"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return self._command

  def _update_metrics(self):
    self.metrics["max_phase_reached"] = torch.maximum(
      self.metrics["max_phase_reached"], self.phase
    )
    self.metrics["completed"] = self.completed.float()
    self.metrics["cumulative_rotation"] = self.cumulative_rotation.abs()

  def _resample_command(self, env_ids):
    self.phase[env_ids] = 0.0
    self.cumulative_rotation[env_ids] = 0.0
    self.prev_pitch[env_ids] = 0.0
    self.completed[env_ids] = False
    self.initial_x[env_ids] = self.robot.data.root_link_pos_w[env_ids, 0]
    self._update_reference(env_ids)

  def _update_command(self):
    # Track pitch rotation
    quat = self.robot.data.root_link_quat_w
    sinp = 2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1])
    sinp = torch.clamp(sinp, -1.0, 1.0)
    current_pitch = torch.asin(sinp)

    pitch_delta = current_pitch - self.prev_pitch
    pitch_delta = torch.where(pitch_delta > math.pi, pitch_delta - 2 * math.pi, pitch_delta)
    pitch_delta = torch.where(pitch_delta < -math.pi, pitch_delta + 2 * math.pi, pitch_delta)
    self.cumulative_rotation += pitch_delta
    self.prev_pitch = current_pitch

    # Get current height
    current_height = self.robot.data.root_link_pos_w[:, 2]

    # Crouch target height
    crouch_target = self.cfg.standing_height - self.cfg.crouch_depth

    # Crouch is complete ONLY when height is below crouch target (no time fallback)
    # This ensures the robot actually crouches before we start tracking rotation
    crouch_complete = current_height < crouch_target + 0.02

    # Crouch progress based on height (how far we've descended)
    height_drop = self.cfg.standing_height - current_height
    crouch_progress = torch.clamp(height_drop / self.cfg.crouch_depth, 0.0, 1.0)

    # Rotation progress for flip phase
    rotation_progress = torch.clamp(-self.cumulative_rotation / (2 * math.pi), 0.0, 1.0)

    # Phase: crouch (0-0.1) based on height, flip (0.1-1.0) based on rotation
    self.phase = torch.where(
      ~crouch_complete,
      crouch_progress * 0.1,  # Crouch phase: 0 to 0.1 based on height
      0.1 + rotation_progress * 0.9  # Flip phase: 0.1 to 1.0 based on rotation
    )
    self.completed = self.phase >= 0.95

    all_envs = torch.arange(self.num_envs, device=self.device)
    self._update_reference(all_envs)

  def _update_reference(self, env_ids):
    phi = self.phase[env_ids]

    standing_height = self.cfg.standing_height
    peak_height = self.cfg.peak_height
    crouch_depth = self.cfg.crouch_depth
    backward_distance = self.cfg.backward_distance

    # Height trajectory:
    # - During crouch (phi < 0.1): dip down
    # - During flip (phi >= 0.1): sine curve from crouch to peak to landing
    in_crouch = phi < 0.1
    crouch_progress = phi / 0.1  # 0 to 1 during crouch
    flip_progress = (phi - 0.1) / 0.9  # 0 to 1 during flip

    # Crouch: standing -> standing - crouch_depth
    crouch_height = standing_height - crouch_depth * crouch_progress

    # Flip: start from crouch depth, peak at middle, return to standing
    # Use sine that starts low, peaks high, returns to standing
    flip_height = (standing_height - crouch_depth) + \
                  (peak_height - standing_height + crouch_depth) * torch.sin(math.pi * flip_progress)

    target_height = torch.where(in_crouch, crouch_height, flip_height)

    # X trajectory (backward arc):
    # - During crouch: stay at initial position
    # - During flip: arc backwards (negative x), peak at mid-flip, return near start
    crouch_x = self.initial_x[env_ids]
    flip_x = self.initial_x[env_ids] - backward_distance * torch.sin(math.pi * flip_progress)
    target_x = torch.where(in_crouch, crouch_x, flip_x)

    # Pitch targets (only relevant during flip phase)
    target_pitch = -2.0 * math.pi * torch.clamp(flip_progress, 0.0, 1.0)
    target_pitch_vel = torch.full_like(phi, -2.0 * math.pi / self.cfg.flip_duration)

    # Slow down rotation near landing
    landing_mask = phi > 0.8
    target_pitch_vel[landing_mask] *= 0.5

    # During crouch, target pitch vel is 0 (not rotating yet)
    target_pitch_vel[in_crouch] = 0.0

    self._command[env_ids, 0] = phi
    self._command[env_ids, 1] = target_height
    self._command[env_ids, 2] = target_pitch
    self._command[env_ids, 3] = target_pitch_vel
    self._command[env_ids, 4] = self.cumulative_rotation[env_ids]
    self._command[env_ids, 5] = target_x

  def _debug_vis_impl(self, visualizer):
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    target_height = self._command[batch, 1].item()
    base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    target_pos = base_pos.copy()
    target_pos[2] = target_height

    visualizer.add_sphere(target_pos, radius=0.05, color=(0.2, 0.8, 0.2, 0.5))


@dataclass(kw_only=True)
class BackflipPhaseCommandCfg(CommandTermCfg):
  asset_name: str
  flip_duration: float = 0.8
  standing_height: float = 0.35
  peak_height: float = 0.7
  crouch_depth: float = 0.25  # How much to lower during crouch (meters) - deeper for more power
  backward_distance: float = 0.15  # How far backward the arc goes (meters)
  class_type: type = BackflipPhaseCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
