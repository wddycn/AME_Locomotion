"""XLL Stage 4: full course plus Stage 3 stake practice."""

from copy import deepcopy

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .loco_hf_terrains_cfg import HfXllCourseTerrainCfg
from .stage3_terrain_cfg import STAGE3_STAKES_TERRAINS_CFG


_stakes3 = deepcopy(STAGE3_STAKES_TERRAINS_CFG.sub_terrains["stakes3"])
_stakes3.proportion = 0.2
_stakes3.holes_depth_levels = 6

STAGE4_COURSE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(12.0, 8.0),
    border_width=3.0,
    num_rows=6,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "course": HfXllCourseTerrainCfg(
            proportion=0.8, border_width=0.25, step_height_range=(0.10, 0.22)
        ),
        "stakes3": _stakes3,
    },
)
