"""Stage 4 full-course constraints and curriculum."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def course_floor_contacts(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    course_num_columns: int = 8,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize stepping on the lower floor between the entry and exit platforms."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
    local = feet - env.scene.env_origins[:, None, :]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold
    in_stakes = (local[..., 0] > 2.0) & (local[..., 0] < 3.9) & (local[..., 1].abs() < 0.75)
    below_tops = local[..., 2] < 0.10
    course = (env.scene.terrain.terrain_types < course_num_columns).unsqueeze(1)
    return (contact & in_stakes & below_tops & course).float().sum(dim=1) / 4.0


def outside_course_lane(env, course_num_columns: int = 8, half_width: float = 0.55) -> torch.Tensor:
    """Keep full-course episodes from bypassing the platforms and stakes."""
    local = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    course = env.scene.terrain.terrain_types < course_num_columns
    return course & (local[:, 0] > 1.1) & (local[:, 0] < 4.8) & (local[:, 1].abs() > half_width)


def course_reached_goal(env, course_num_columns: int = 8) -> torch.Tensor:
    """Finish the route after landing beyond the exit platform."""
    local = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    course = env.scene.terrain.terrain_types < course_num_columns
    landed = (local[:, 0] > 5.1) & (local[:, 1].abs() < 0.55) & (local[:, 2] < 0.55)
    retained_stakes = local[:, 0] > 5.1
    return torch.where(course, landed, retained_stakes)


def course_terrain_levels(env, env_ids, course_num_columns: int = 8) -> torch.Tensor:
    """Advance after crossing the exit step onto the landing ground."""
    terrain = env.scene.terrain
    root = env.scene["robot"].data.root_pos_w[env_ids]
    local = root - env.scene.env_origins[env_ids]
    move_up = course_reached_goal(env, course_num_columns)[env_ids]
    move_down = (local[:, 0] < 2.0) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()
