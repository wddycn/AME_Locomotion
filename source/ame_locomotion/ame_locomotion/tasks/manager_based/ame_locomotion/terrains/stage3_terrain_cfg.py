"""XLL Stage 3: Stage 2 stakes3 with a pit-depth-only curriculum."""

from copy import deepcopy

from .finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG


STAGE3_STAKES_TERRAINS_CFG = deepcopy(FINETUNE_ROUGH_TERRAINS_CFG)
stakes3 = deepcopy(FINETUNE_ROUGH_TERRAINS_CFG.sub_terrains["stakes3"])
stakes3.proportion = 1.0
stakes3.layout_difficulty = 1.0
stakes3.holes_depth_range = (-1.2, -0.2)
stakes3.holes_depth_levels = STAGE3_STAKES_TERRAINS_CFG.num_rows
STAGE3_STAKES_TERRAINS_CFG.sub_terrains = {"stakes3": stakes3}
