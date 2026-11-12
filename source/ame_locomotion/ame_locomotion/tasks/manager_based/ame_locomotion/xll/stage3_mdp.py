"""Stage 3 stake-lane constraints and curriculum for XLL."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _contact_totals(env) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(env, "_stage3_floor_contacts"):
        env._stage3_floor_contacts = torch.zeros(env.num_envs, device=env.device)
        env._stage3_stance_contacts = torch.zeros(env.num_envs, device=env.device)
    return env._stage3_floor_contacts, env._stage3_stance_contacts


def low_ground_foot_contacts(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    low_height_threshold: float = -0.1,
) -> torch.Tensor:
    """Count feet contacting the floor below the stake tops, and track episode ratios."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold
    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]
    outside_platform = (
        foot_pos[:, :, :2] - env.scene.env_origins[:, None, :2]
    ).abs().amax(dim=-1) > 1.0
    valid_contact = contact & outside_platform
    origin_z = env.scene.env_origins[:, 2:3]
    low_contact = valid_contact & (foot_pos[:, :, 2] - origin_z < low_height_threshold)

    floor_total, stance_total = _contact_totals(env)
    floor_total += low_contact.sum(dim=1)
    stance_total += valid_contact.sum(dim=1)
    return low_contact.float().sum(dim=1) / 4.0


def outside_stake_lane(env, half_width: float = 0.8) -> torch.Tensor:
    """End an episode when the base leaves the x-directed stake lane."""
    robot = env.scene["robot"]
    lateral_distance = robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return lateral_distance.abs() > half_width


def stake_lane_terrain_levels(
    env,
    env_ids: torch.Tensor,
    min_progress: float = 4.0,
    max_floor_ratio: float = 0.10,
    min_stance_contacts: int = 20,
    lane_half_width: float = 0.8,
) -> torch.Tensor:
    """Upgrade only after forward progress with most support on stake tops."""
    terrain = env.scene.terrain
    robot = env.scene["robot"]
    floor_total, stance_total = _contact_totals(env)
    progress = robot.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    floor_ratio = floor_total[env_ids] / stance_total[env_ids].clamp(min=1.0)
    enough_contacts = stance_total[env_ids] >= min_stance_contacts
    inside_lane = (
        robot.data.root_pos_w[env_ids, 1] - env.scene.env_origins[env_ids, 1]
    ).abs() <= lane_half_width

    move_up = (progress > min_progress) & enough_contacts & (floor_ratio <= max_floor_ratio) & inside_lane
    commanded_forward = env.command_manager.get_command("base_velocity")[env_ids, 0].clamp(min=0.0)
    move_down = (progress < commanded_forward * env.max_episode_length_s * 0.5) | (
        enough_contacts & (floor_ratio > 0.35)
    )
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    floor_total[env_ids] = 0
    stance_total[env_ids] = 0
    return terrain.terrain_levels.float().mean()
