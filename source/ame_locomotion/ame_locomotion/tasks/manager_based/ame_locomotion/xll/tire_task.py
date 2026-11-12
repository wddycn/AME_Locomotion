"""Route constraints and curriculum for the fixed tire array."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def tire_course_success(env) -> torch.Tensor:
    """Finish after clearing the array and continuing along the forward exit lane."""
    local = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    # The largest array ends at x=3.40 m.  Requiring x>3.90 m confirms that
    # the robot has fully cleared it and keeps moving along the +X route.
    return (local[:, 0] > 3.9) & (local[:, 1].abs() <= 0.35)


def outside_tire_lane(env) -> torch.Tensor:
    """End an episode when the body bypasses the array from its side."""
    local = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return (local[:, 1].abs() > 0.35) | (local[:, 0] < -0.5)


def _tire_foot_contact_masks(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float,
    tire_surface_height: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify support as ground in an opening or elevated tire rubber."""
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
    local_feet = feet - env.scene.env_origins[:, None, :]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > contact_threshold
    root_x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    in_array = (root_x > 0.7) & (root_x < 3.8)
    within_array_length = (local_feet[..., 0] > 0.95) & (local_feet[..., 0] < 3.75)
    relevant_contact = contact & within_array_length & in_array[:, None]
    on_tire = relevant_contact & (local_feet[..., 2] > tire_surface_height)
    on_ground = relevant_contact & ~on_tire
    return on_ground, on_tire, contact


def safe_tire_array_foot_contacts(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    tire_surface_height: float = 0.07,
    min_forward_speed: float = 0.20,
) -> torch.Tensor:
    """Reward forward-moving support on ground visible through holes or diamond gaps."""
    on_ground, _, contact = _tire_foot_contact_masks(
        env, asset_cfg, sensor_cfg, contact_threshold, tire_surface_height
    )
    forward_speed = env.scene["robot"].data.root_lin_vel_b[:, 0]
    forward_scale = (forward_speed / min_forward_speed).clamp_(0.0, 1.0)
    return on_ground.float().sum(dim=1) / contact.sum(dim=1).clamp_min(1) * forward_scale


def tire_surface_foot_contacts(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    tire_surface_height: float = 0.07,
) -> torch.Tensor:
    """Penalize support on the elevated tire rubber, while allowing all ground openings."""
    _, on_tire, contact = _tire_foot_contact_masks(
        env, asset_cfg, sensor_cfg, contact_threshold, tire_surface_height
    )
    return on_tire.float().sum(dim=1) / contact.sum(dim=1).clamp_min(1)


def tire_terrain_levels(env, env_ids: torch.Tensor, required_streak: int = 3) -> torch.Tensor:
    """Change terrain level only after three consecutive episode outcomes."""
    terrain = env.scene.terrain
    success = tire_course_success(env)[env_ids]
    if getattr(env, "_tire_course_streak", None) is None:
        env._tire_course_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    streak = env._tire_course_streak[env_ids]
    streak = torch.where(success, streak.clamp_min(0) + 1, streak.clamp_max(0) - 1)
    move_up = streak >= required_streak
    move_down = streak <= -required_streak
    terrain.update_env_origins(env_ids, move_up, move_down)
    streak[move_up | move_down] = 0
    env._tire_course_streak[env_ids] = streak
    return terrain.terrain_levels.float().mean()
