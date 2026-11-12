"""Self-contained XLL articulation and nominal standing configuration."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from .delayed_motor import DelayedDCMotorCfg


_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
XLL_URDF = _DATA_DIR / "xll_description" / "urdf" / "XLL.urdf"

XLL_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


XLL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(XLL_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": 0.8,
            "FR_hip_joint": 0.0, "FR_thigh_joint": -0.8, "FR_calf_joint": -0.8,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 0.8, "RL_calf_joint": 0.8,
            "RR_hip_joint": 0.0, "RR_thigh_joint": -0.8, "RR_calf_joint": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=31.7,
            saturation_effort=31.7,
            velocity_limit=52.4,
            stiffness=20.0,
            damping=0.1,
            friction=0.0,
            armature=0.0042,
            min_delay=0,
            max_delay=1,
        ),
    },
)
