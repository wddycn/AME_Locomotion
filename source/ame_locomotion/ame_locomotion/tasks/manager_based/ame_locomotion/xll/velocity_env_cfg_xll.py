"""AME rough-terrain locomotion task for the 12-DoF XLL quadruped."""

from copy import deepcopy
from importlib import import_module

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp
from ame_locomotion.tasks.manager_based.ame_locomotion.assets.robots.xll import XLL_CFG, XLL_JOINT_NAMES
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.terrain_cfg import ROUGH_TERRAINS_CFG
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.stage3_terrain_cfg import STAGE3_STAKES_TERRAINS_CFG
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.stage4_terrain_cfg import STAGE4_COURSE_TERRAINS_CFG

from . import rewards as xll_rewards
from .commands import XLLStage0VelocityCommand
from . import stage3 as xll_stage3
from . import stage4 as xll_stage4
from . import tire_task as xll_tires
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.tire_terrain_cfg import TIRE_ARRAY_TERRAINS_CFG


STAGE_0_TERRAINS = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
)
STAGE_1_TERRAINS = ("hf_steppingstones", "hf_gaps")


_g1_cfg = import_module(
    "ame_locomotion.tasks.manager_based.ame_locomotion.29dof.velocity_env_cfg_29dof"
)
BaseActionsCfg = _g1_cfg.ActionsCfg
G1RoughEnvCfg = _g1_cfg.G1RoughEnvCfg


@configclass
class XLLActionsCfg(BaseActionsCfg):
    """Position actions in the XLL URDF joint order."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=XLL_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class XLLAMERoughEnvCfg(G1RoughEnvCfg):
    """XLL stage 1: stepping stones and ring gaps."""

    actions: XLLActionsCfg = XLLActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # XLL_CFG supplies the measured standing pose: root height 0.36 m and
        # the twelve nominal leg angles used by XLL-Rough-Teleop-v0.
        self.scene.robot = XLL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # The XLL URDF calls this fixed sensor link "radar". There is no
        # "ladar_link" in the model, so using the actual link is required.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/radar"
        self.actions.joint_pos.scale = 0.25

        joint_cfg = SceneEntityCfg("robot", joint_names=XLL_JOINT_NAMES, preserve_order=True)
        for obs_group in (self.observations.policy, self.observations.critic):
            obs_group.joint_pos.params["asset_cfg"] = joint_cfg
            obs_group.joint_vel.params["asset_cfg"] = joint_cfg
        self.observations.policy.enable_corruption = False
        for term_name in ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"):
            getattr(self.observations.policy, term_name).noise = None
        self.observations.policy.height_scan.params["noise"] = False

        # Use the XLL PD-baseline reset/domain-randomization schedule from
        # XLL-Rough-Teleop-v0 before introducing robustness randomization.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Match the G1 stage-1 objectives, with foot ordering and joint pairs
        # adapted to the quadruped morphology.
        feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        feet_cfg = SceneEntityCfg("contact_forces", body_names=feet_names, preserve_order=True)
        feet_asset_cfg = SceneEntityCfg("robot", body_names=feet_names, preserve_order=True)
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        # Stage 0/1 use the same velocity tracking objective.
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        self.rewards.dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
        self.rewards.dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
        self.rewards.dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
        self.rewards.dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
        self.rewards.dof_torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.25,
            params={"command_name": "base_velocity", "sensor_cfg": feet_cfg, "threshold": 0.3},
        )
        self.rewards.feet_air_time_variance = RewTerm(
            func=mdp.air_time_variance_penalty, weight=-0.7, params={"sensor_cfg": feet_cfg}
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": feet_cfg,
                "asset_cfg": feet_asset_cfg,
            },
        )
        self.rewards.feet_stumble = RewTerm(
            func=mdp.feet_stumble, weight=-2.0, params={"sensor_cfg": feet_cfg}
        )
        self.rewards.same_end_airborne = RewTerm(
            func=xll_rewards.same_end_airborne,
            weight=-0.5,
            params={"sensor_cfg": feet_cfg, "command_name": "base_velocity"},
        )
        self.rewards.feet_too_near = RewTerm(
            func=xll_rewards.same_end_feet_too_near,
            weight=-1.0,
            params={"asset_cfg": feet_asset_cfg, "threshold": 0.2},
        )
        self.rewards.joint_coordination = RewTerm(
            func=mdp.joint_coordination_rel,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "coord_joints": [
                    ["FL_thigh_joint", "RR_thigh_joint"],
                    ["FR_thigh_joint", "RL_thigh_joint"],
                ],
                "coord_signs": [[1.0, -1.0], [1.0, -1.0]],
            },
        )
        self.rewards.diagonal_trot_contact = RewTerm(
            func=xll_rewards.diagonal_trot_contact,
            weight=0.2,
            params={"sensor_cfg": feet_cfg, "command_name": "base_velocity"},
        )
        self.rewards.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None

        # Like XLL-Rough-Teleop-v0, only a base collision terminates the episode.
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="base")

        self._set_training_terrains(STAGE_1_TERRAINS)

    def _enable_domain_randomization(self):
        """Shared physical perturbations and policy noise for training."""
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.6, 1.2),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-0.5, 1.0),
                "operation": "add",
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}},
        )
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.1, 0.1)
        self.observations.policy.enable_corruption = True
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-2.0, n_max=2.0)
        self.observations.policy.height_scan.params["noise"] = True

    def _set_training_terrains(self, terrain_names: tuple[str, ...]):
        terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
        terrain_cfg.curriculum = self.curriculum.terrain_levels is not None
        terrain_cfg.sub_terrains = {
            name: terrain_cfg.sub_terrains[name] for name in terrain_names
        }
        for sub_terrain in terrain_cfg.sub_terrains.values():
            sub_terrain.proportion = 1.0 / len(terrain_names)
        self.scene.terrain.terrain_generator = terrain_cfg


@configclass
class XLLAMEStage0EnvCfg(XLLAMERoughEnvCfg):
    """XLL stage 0: stairs, boxes, rough ground, and slopes."""

    def __post_init__(self):
        super().__post_init__()
        self._set_training_terrains(STAGE_0_TERRAINS)
        self._enable_domain_randomization()
        # Reserve 15% for pure lateral motion and 20% for low-speed in-place
        # turning; 5% is standing and 60% remains general motion.
        self.commands.base_velocity.class_type = XLLStage0VelocityCommand
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.lin_vel_z_l2.weight = -3.0

        # First learn to move with the compact XLL-Rough-Teleop-v0 reward set.
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.lateral_turn_step_air_time = RewTerm(
            func=xll_rewards.lateral_turn_step_air_time,
            weight=0.2,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                ),
                "command_name": "base_velocity",
                "threshold": 0.15,
            },
        )

        self.rewards.termination_penalty = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_vel_l2 = None
        self.rewards.dof_pos_limits = None
        self.rewards.dof_torques_limits = None
        self.rewards.feet_air_time_variance = None
        self.rewards.feet_stumble = None
        self.rewards.same_end_airborne = None
        self.rewards.feet_too_near = None
        self.rewards.joint_coordination = None


@configclass
class XLLAMEStage1EnvCfg(XLLAMEStage0EnvCfg):
    """XLL stage 1: original tracking objectives on stones and gaps, with randomization."""

    def __post_init__(self):
        super().__post_init__()
        self._set_training_terrains(STAGE_1_TERRAINS)
        self.commands.base_velocity.class_type = UniformVelocityCommand
        self.rewards.lateral_turn_step_air_time = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -2.0


@configclass
class XLLAMEStage2EnvCfg(XLLAMEStage1EnvCfg):
    """XLL stage 2: harder terrains and stronger robustness training."""

    def __post_init__(self):
        super().__post_init__()
        terrain_cfg = deepcopy(FINETUNE_ROUGH_TERRAINS_CFG)
        terrain_cfg.curriculum = self.curriculum.terrain_levels is not None
        self.scene.terrain.terrain_generator = terrain_cfg
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.10, 0.10), "y": (-0.10, 0.10), "yaw": (-0.15, 0.15),
        }
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4

        # Tighten the proven stage-0/1 objectives without re-enabling the
        # extra penalties that previously stalled rough-terrain progress.
        weights = {
            "track_lin_vel_xy_exp": 2.5,
            "lin_vel_z_l2": -3.0,
            "ang_vel_xy_l2": -0.1,
            "undesired_contacts": -1.5,
            "action_rate_l2": -0.02,
            "feet_slide": -0.2,
            "diagonal_trot_contact": 0.3,
        }
        for name, weight in weights.items():
            getattr(self.rewards, name).weight = weight

        self._enable_domain_randomization()


@configclass
class XLLStage3RewardsCfg(_g1_cfg.RewardsCfg):
    pit_foot_contact = RewTerm(
        func=xll_stage3.pit_foot_contact,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
        },
    )


@configclass
class XLLAMEStage3EnvCfg(XLLAMEStage2EnvCfg):
    """XLL stage 3: Stage 2 objectives on stakes3 with progressively shallower pits."""

    rewards: XLLStage3RewardsCfg = XLLStage3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        terrain_cfg = deepcopy(STAGE3_STAKES_TERRAINS_CFG)
        terrain_cfg.curriculum = self.curriculum.terrain_levels is not None
        self.scene.terrain.terrain_generator = terrain_cfg


@configclass
class XLLStage4RewardsCfg(XLLStage3RewardsCfg):
    course_floor_contacts = RewTerm(
        func=xll_stage4.course_floor_contacts,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "course_num_columns": 8,
        },
    )


@configclass
class XLLStage4TerminationsCfg(_g1_cfg.TerminationsCfg):
    course_success = DoneTerm(
        func=xll_stage4.course_reached_goal,
        params={"course_num_columns": 8},
    )
    outside_course_lane = DoneTerm(
        func=xll_stage4.outside_course_lane,
        params={"course_num_columns": 8, "half_width": 0.55},
    )


@configclass
class XLLStage4CurriculumCfg(_g1_cfg.CurriculumCfg):
    terrain_levels = CurrTerm(func=xll_stage4.course_terrain_levels, params={"course_num_columns": 8})


@configclass
class XLLAMEStage4EnvCfg(XLLAMEStage3EnvCfg):
    """Fine-tune Stage 3 on the complete entry-step, stakes, exit-step course."""

    rewards: XLLStage4RewardsCfg = XLLStage4RewardsCfg()
    terminations: XLLStage4TerminationsCfg = XLLStage4TerminationsCfg()
    curriculum: XLLStage4CurriculumCfg = XLLStage4CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        terrain_cfg = deepcopy(STAGE4_COURSE_TERRAINS_CFG)
        terrain_cfg.curriculum = self.curriculum.terrain_levels is not None
        self.scene.terrain.terrain_generator = terrain_cfg
        self.scene.terrain.max_init_terrain_level = 1
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.10, 0.10), "y": (-0.10, 0.10), "yaw": (-0.15, 0.15),
        }
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class XLLAMERoughPlayEnvCfg(XLLAMEStage1EnvCfg):
    """Deterministic, single-environment-friendly variant for policy playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class XLLAMEStage0PlayEnvCfg(XLLAMEStage0EnvCfg):
    """Stage 0 playback with the same rewards and terrains as training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class XLLAMEStage2PlayEnvCfg(XLLAMEStage2EnvCfg):
    """Stage 2 playback without training perturbations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.scene.terrain.terrain_generator.curriculum = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class XLLAMEStakes02TestEnvCfg(XLLAMEStage2PlayEnvCfg):
    """Playback-only test of the Stage 2 policy on 0.2 m-deep stakes3 pits."""

    def __post_init__(self):
        super().__post_init__()
        terrain_cfg = deepcopy(STAGE3_STAKES_TERRAINS_CFG)
        terrain_cfg.num_rows = 1
        terrain_cfg.num_cols = 1
        terrain_cfg.curriculum = False
        terrain_cfg.seed = 42
        stakes = terrain_cfg.sub_terrains["stakes3"]
        stakes.holes_depth_range = None
        stakes.holes_depth = -0.2
        self.scene.terrain.terrain_generator = terrain_cfg
        self.scene.terrain.max_init_terrain_level = 0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class XLLAMEStage3PlayEnvCfg(XLLAMEStage3EnvCfg):
    """Stage 3 playback on a compact four-level stake terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        terrain_cfg = self.scene.terrain.terrain_generator
        terrain_cfg.num_rows = 4
        terrain_cfg.num_cols = 4
        terrain_cfg.border_width = 2.0
        terrain_cfg.sub_terrains["stakes3"].holes_depth_levels = terrain_cfg.num_rows
        terrain_cfg.curriculum = True
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class XLLTireTerminationsCfg(_g1_cfg.TerminationsCfg):
    tire_success = DoneTerm(func=xll_tires.tire_course_success)
    outside_tire_lane = DoneTerm(func=xll_tires.outside_tire_lane)


@configclass
class XLLTireCurriculumCfg(_g1_cfg.CurriculumCfg):
    terrain_levels = CurrTerm(func=xll_tires.tire_terrain_levels, params={"required_streak": 3})


@configclass
class XLLAMETireArrayEnvCfg(XLLAMEStage2EnvCfg):
    """Stage 2 policy fine-tuning on fixed, randomized tire arrays."""

    terminations: XLLTireTerminationsCfg = XLLTireTerminationsCfg()
    curriculum: XLLTireCurriculumCfg = XLLTireCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = deepcopy(TIRE_ARRAY_TERRAINS_CFG)
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.num_envs = 256
        self.episode_length_s = 15.0
        self.commands.base_velocity.class_type = UniformVelocityCommand
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.10, 0.10), "y": (-0.10, 0.10), "yaw": (-0.15, 0.15),
        }
        feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.rewards.safe_tire_array_foot_contacts = RewTerm(
            func=xll_tires.safe_tire_array_foot_contacts,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=feet_names, preserve_order=True),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet_names, preserve_order=True),
                "contact_threshold": 1.0,
                "tire_surface_height": 0.07,
                "min_forward_speed": 0.20,
            },
        )
        self.rewards.tire_surface_foot_contacts = RewTerm(
            func=xll_tires.tire_surface_foot_contacts,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=feet_names, preserve_order=True),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet_names, preserve_order=True),
                "contact_threshold": 1.0,
                "tire_surface_height": 0.07,
            },
        )
        # Make the required exit segment valuable, rather than relying only
        # on dense velocity tracking after the final row of tires.
        self.rewards.tire_success = RewTerm(func=xll_tires.tire_course_success, weight=25.0)
        self.rewards.outside_tire_lane = RewTerm(func=xll_tires.outside_tire_lane, weight=-100.0)


@configclass
class XLLAMETireArrayPlayEnvCfg(XLLAMETireArrayEnvCfg):
    """Tire-array playback with physical perturbations and scan noise disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.seed = 42
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
