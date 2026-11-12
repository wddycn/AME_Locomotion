"""Quadruped contact and gait penalties for the XLL locomotion task."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def same_end_airborne(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize simultaneous swing of both front or both rear feet while moving."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > contact_threshold
    front_air = ~contact[:, 0] & ~contact[:, 1]
    rear_air = ~contact[:, 2] & ~contact[:, 3]
    moving = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return (front_air.float() + rear_air.float()) * moving.float()


def same_end_feet_too_near(env, asset_cfg: SceneEntityCfg, threshold: float = 0.2) -> torch.Tensor:
    """Penalize front-pair or rear-pair feet that approach too closely in XY."""
    asset = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    front_distance = torch.linalg.norm(foot_xy[:, 0] - foot_xy[:, 1], dim=-1)
    rear_distance = torch.linalg.norm(foot_xy[:, 2] - foot_xy[:, 3], dim=-1)
    return (threshold - front_distance).clamp(min=0.0) + (threshold - rear_distance).clamp(min=0.0)


def diagonal_trot_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.25,
    velocity_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward alternating diagonal support, with feet ordered FL, FR, RL, RR."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    left_diagonal = contact[:, 0] & contact[:, 3] & ~contact[:, 1] & ~contact[:, 2]
    right_diagonal = contact[:, 1] & contact[:, 2] & ~contact[:, 0] & ~contact[:, 3]
    commanded = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    speed = torch.linalg.norm(env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
    moving = (commanded > command_threshold) & (speed > velocity_threshold)
    return (left_diagonal | right_diagonal).float() * moving.float()


def lateral_turn_step_air_time(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.15,
) -> torch.Tensor:
    """Reward actual swing-and-touchdown during pure lateral or yaw commands."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    pure_lateral_or_turn = (command[:, 0].abs() < 0.1) & (
        (command[:, 1].abs() > 0.15) | (command[:, 2].abs() > 0.15)
    )
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    steps = ((air_time - threshold).clamp(min=0.0, max=0.35) * first_contact).sum(dim=1)
    return steps * pure_lateral_or_turn.float()
