from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class BackflipCommand(CommandTerm):
    """Phase-based command for backflip task.
    
    The command is a single scalar (phase) that goes from 0 to 1
    over the duration of the backflip.
    """
    cfg: BackflipCommandCfg

    def __init__(self, cfg: BackflipCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.asset_name]
        
        # Phase goes from 0 (start) to 1 (landed)
        self.phase = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return phase as shape [num_envs, 1]."""
        return self.phase.unsqueeze(-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Reset phase to 0 at start of episode."""
        self.phase[env_ids] = 0.0

    def _update_command(self) -> None:
        """Increment phase each timestep."""
        self.phase += self._env.step_dt / self.cfg.phase_duration
        self.phase = torch.clamp(self.phase, 0.0, 1.0)

    def _update_metrics(self) -> None:
        """Optional: track backflip metrics."""
        pass  # Can add logging later if needed


@dataclass(kw_only=True)
class BackflipCommandCfg(CommandTermCfg):
    """Configuration for backflip command generator."""
    asset_name: str
    phase_duration: float = 1.5  # seconds for one backflip
    resampling_time_range: tuple[float, float] = (0.0, 0.0)  # No resampling needed, phase is continuous
    class_type: type[CommandTerm] = BackflipCommand