"""Stage 3 pit-contact reward."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

def pit_foot_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    low_height: float = -0.1,
) -> torch.Tensor:
    """Return one while any foot contacts terrain below the stake tops."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > contact_threshold
    feet_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    low_contact = contact & (feet_z - env.scene.env_origins[:, 2:3] < low_height)

    return low_contact.any(dim=1).float()


def low_stake_contacts(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    low_height: float = -0.1,
    platform_half_width: float = 1.0,
) -> torch.Tensor:
    """Count feet contacting the lower ground, and record the episode's landing quality."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > contact_threshold
    foot_pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
    local_foot_pos = foot_pos - env.scene.env_origins[:, None, :]
    low_contact = contact & (local_foot_pos[..., 2] < low_height)

    if not hasattr(env, "_stage3_low_contacts"):
        env._stage3_low_contacts = torch.zeros(env.num_envs, device=env.device)
        env._stage3_stance_contacts = torch.zeros(env.num_envs, device=env.device)

    outside_platform = (local_foot_pos[..., :2].abs() > platform_half_width).any(dim=-1)
    env._stage3_low_contacts += (low_contact & outside_platform).sum(dim=1)
    env._stage3_stance_contacts += (contact & outside_platform).sum(dim=1)
    return low_contact.sum(dim=1).float()


def stake_terrain_levels(
    env,
    env_ids,
    max_low_contact_ratio: float = 0.1,
    min_stance_contacts: int = 20,
) -> torch.Tensor:
    """Advance only after sufficient travel with mostly stake-top foot contacts."""
    terrain: TerrainImporter = env.scene.terrain
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    if hasattr(env, "_stage3_stance_contacts"):
        stance = env._stage3_stance_contacts[env_ids]
        low = env._stage3_low_contacts[env_ids]
    else:
        stance = torch.zeros_like(distance)
        low = torch.zeros_like(distance)

    good_contacts = (stance >= min_stance_contacts) & (low <= max_low_contact_ratio * stance)
    move_up = (distance > terrain.cfg.terrain_generator.size[0] / 2) & good_contacts
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    if hasattr(env, "_stage3_stance_contacts"):
        env._stage3_stance_contacts[env_ids] = 0
        env._stage3_low_contacts[env_ids] = 0
    return torch.mean(terrain.terrain_levels.float())
