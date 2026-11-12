"""Stage-0 velocity commands with dedicated sideways and turning practice."""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand


class XLLStage0VelocityCommand(UniformVelocityCommand):
    """Mix general motion with lateral, in-place turning, and standing commands."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        mode = torch.rand(ids.numel(), device=self.device)
        lateral = ids[mode < 0.15]
        turn = ids[(mode >= 0.15) & (mode < 0.35)]
        standing = ids[(mode >= 0.35) & (mode < 0.40)]

        if lateral.numel():
            speed = 0.2 + 0.3 * torch.rand(lateral.numel(), device=self.device)
            sign = torch.where(torch.rand(lateral.numel(), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[lateral, 0] = 0.0
            self.vel_command_b[lateral, 1] = speed * sign
            self.vel_command_b[lateral, 2] = 0.0

        if turn.numel():
            speed = 0.2 + 0.4 * torch.rand(turn.numel(), device=self.device)
            sign = torch.where(torch.rand(turn.numel(), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[turn, :2] = 0.0
            self.vel_command_b[turn, 2] = speed * sign

        if standing.numel():
            self.vel_command_b[standing] = 0.0
