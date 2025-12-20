from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class BackflipCommand(CommandTerm):
    """Rotation-based phase command for backflip task.
    
    Key insight: Phase advances based on ACTUAL ROTATION, not time.
    This forces the robot to actually flip to make progress.
    
    Phases:
      - Crouch (0.0-0.1): Lower height to wind up
      - Flip (0.1-1.0): Rotate backwards, phase tracks rotation progress
    """
    cfg: BackflipCommandCfg

    def __init__(self, cfg: BackflipCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]
        
        # Phase goes from 0 (start) to 1 (full rotation complete)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        
        # Track cumulative rotation (negative = backward flip)
        self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
        self.prev_pitch = torch.zeros(self.num_envs, device=self.device)
        
        # Command output: [phase, target_height, target_pitch, target_pitch_vel]
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Metrics for logging
        self.metrics["max_phase_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cumulative_rotation"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return command as shape [num_envs, 4]."""
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Reset phase and rotation tracking at start of episode."""
        self.phase[env_ids] = 0.0
        self.cumulative_rotation[env_ids] = 0.0
        self.prev_pitch[env_ids] = 0.0
        self._update_reference(env_ids)

    def _update_command(self) -> None:
        """Update phase based on actual rotation, not time."""
        # Get current pitch from quaternion
        quat = self.robot.data.root_link_quat_w  # (w, x, y, z)
        sinp = 2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1])
        sinp = torch.clamp(sinp, -1.0, 1.0)
        current_pitch = torch.asin(sinp)
        
        # Track pitch delta (handle wrap-around)
        pitch_delta = current_pitch - self.prev_pitch
        pitch_delta = torch.where(pitch_delta > math.pi, pitch_delta - 2 * math.pi, pitch_delta)
        pitch_delta = torch.where(pitch_delta < -math.pi, pitch_delta + 2 * math.pi, pitch_delta)
        self.cumulative_rotation += pitch_delta
        self.prev_pitch = current_pitch
        
        # Get current height
        current_height = self.robot.data.root_link_pos_w[:, 2]
        
        # Crouch detection: height below threshold
        crouch_target = self.cfg.standing_height - self.cfg.crouch_depth
        crouch_complete = current_height < crouch_target + 0.02
        
        # Crouch progress based on height
        height_drop = self.cfg.standing_height - current_height
        crouch_progress = torch.clamp(height_drop / self.cfg.crouch_depth, 0.0, 1.0)
        
        # Rotation progress: -cumulative_rotation / 2π (negative because backward flip)
        rotation_progress = torch.clamp(-self.cumulative_rotation / (2 * math.pi), 0.0, 1.0)
        
        # Phase: crouch (0-0.1) based on height, flip (0.1-1.0) based on rotation
        self.phase = torch.where(
            ~crouch_complete,
            crouch_progress * 0.1,  # Crouch phase: 0 to 0.1 based on height
            0.1 + rotation_progress * 0.9  # Flip phase: 0.1 to 1.0 based on rotation
        )
        
        # Update reference targets for all environments
        all_envs = torch.arange(self.num_envs, device=self.device)
        self._update_reference(all_envs)

    def _update_reference(self, env_ids: torch.Tensor) -> None:
        """Generate target height, pitch, pitch_vel based on phase."""
        phi = self.phase[env_ids]
        
        standing_height = self.cfg.standing_height
        peak_height = self.cfg.peak_height
        crouch_depth = self.cfg.crouch_depth
        
        # Phase regions
        in_crouch = phi < 0.1
        crouch_progress = phi / 0.1  # 0 to 1 during crouch
        flip_progress = (phi - 0.1) / 0.9  # 0 to 1 during flip
        
        # Height trajectory:
        # Crouch: standing -> standing - crouch_depth
        crouch_height = standing_height - crouch_depth * crouch_progress
        
        # Flip: sine curve from crouch to peak and back to standing
        flip_height = (standing_height - crouch_depth) + \
                      (peak_height - standing_height + crouch_depth) * torch.sin(math.pi * flip_progress)
        
        target_height = torch.where(in_crouch, crouch_height, flip_height)
        
        # Pitch targets (2π rotation over flip phase)
        target_pitch = -2.0 * math.pi * torch.clamp(flip_progress, 0.0, 1.0)
        
        # Target pitch velocity (negative = backward rotation)
        target_pitch_vel = torch.full_like(phi, -2.0 * math.pi / self.cfg.flip_duration)
        target_pitch_vel[in_crouch] = 0.0  # No rotation during crouch
        target_pitch_vel[phi > 0.8] *= 0.5  # Slow down near landing
        
        self._command[env_ids, 0] = phi
        self._command[env_ids, 1] = target_height
        self._command[env_ids, 2] = target_pitch
        self._command[env_ids, 3] = target_pitch_vel

    def _update_metrics(self) -> None:
        """Track backflip metrics for logging."""
        self.metrics["max_phase_reached"] = torch.maximum(
            self.metrics["max_phase_reached"], self.phase
        )
        self.metrics["cumulative_rotation"] = self.cumulative_rotation.abs()


@dataclass(kw_only=True)
class BackflipCommandCfg(CommandTermCfg):
    """Configuration for rotation-based backflip command generator."""
    asset_name: str
    flip_duration: float = 1.0  # seconds for the flip portion
    standing_height: float = 0.35  # Go2 standing height
    peak_height: float = 0.85  # Target peak height during flip
    crouch_depth: float = 0.1  # How much to lower during crouch
    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    class_type: type[CommandTerm] = BackflipCommand
