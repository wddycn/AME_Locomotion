from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from . import loco_hf_terrains

@configclass
class HfStonesBridgeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stones bridge height field terrain."""

    function = loco_hf_terrains.stones_bridge_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""
    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""
    stone_length_range: tuple[float, float] = MISSING
    """The minimum and maximum length of the stones (in m)."""
    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""
    stone_lateral_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum lateral distance between stones (in m)."""
    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfConcentricGapTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a concentric gaps height field terrain."""

    function = loco_hf_terrains.concentric_gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gaps (in m)."""
    ground_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the ground (in m)."""
    ground_height_max: float = MISSING
    """The maximum height of the ground (in m).""" 
    gap_depth: float = -2.0
    """The depth of the gaps (negative obstacles). Defaults to -2.0."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfDoubleColumnStakesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a two-column plum-blossom stakes height field terrain."""

    function = loco_hf_terrains.double_column_stakes_terrain

    stake_height_max: float = MISSING
    """The maximum height variation of the stakes (in m)."""
    stake_side_range: tuple[float, float] = MISSING
    """The minimum and maximum side length of the square stakes (in m)."""
    stake_gap_range: tuple[float, float] = MISSING
    """The minimum and maximum clear gap between successive stakes along the extension axis (in m)."""
    column_gap_range: tuple[float, float] = MISSING
    """The minimum and maximum lateral clear gap between the two stake columns (in m)."""
    column_jitter: float = 0.0
    """Maximum lateral jitter applied to each stake center (in m). Defaults to 0.0."""
    holes_depth: float = -2.0
    """The base depth around the stakes (negative obstacles). Defaults to -2.0."""
    platform_width: float = 1.0
    """Width of the central platform patch (in m). Defaults to 1.0."""


@configclass
class HfAlternateColumnStakesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a two-column plum-blossom stakes height field terrain."""

    function = loco_hf_terrains.alternate_column_stakes_terrain

    stake_height_max: float = MISSING
    """The maximum height variation of the stakes (in m)."""
    stake_side_range: tuple[float, float] = MISSING
    """The minimum and maximum side length of the square stakes (in m)."""
    stake_gap_range: tuple[float, float] = MISSING
    """The minimum and maximum clear gap between successive stakes along the extension axis (in m)."""
    column_gap_range: tuple[float, float] = MISSING
    """The lateral center-to-center distance between alternating stakes (in m)."""
    column_jitter: float = 0.0
    """Maximum lateral jitter applied to each stake center (in m). Defaults to 0.0."""
    snap_to_nearest_grid: bool = False
    """Round sizes to the nearest grid cell and preserve odd-cell column spacing."""
    preserve_column_spacing: bool = False
    """Round only the column distance and retain odd-cell spacing without changing stake size or forward gap."""
    holes_depth: float = -2.0
    """The base depth around the stakes (negative obstacles). Defaults to -2.0."""
    layout_difficulty: float | None = None
    """Fixed difficulty for stake size and spacing; None follows the terrain curriculum."""
    holes_depth_range: tuple[float, float] | None = None
    """Pit depths at the first and last curriculum levels; None uses holes_depth."""
    holes_depth_levels: int = 10
    """Number of discrete pit-depth levels when holes_depth_range is set."""
    platform_width: float = 1.0
    """Width of the central platform patch (in m). Defaults to 1.0."""


@configclass
class HfXllCourseTerrainCfg(HfTerrainBaseCfg):
    """Ground, entry step, 0.2 m stakes, exit step, and landing ground."""

    function = loco_hf_terrains.xll_course_terrain

    step_height_range: tuple[float, float] = (0.10, 0.22)
    """Entry/exit platform and stake height from easiest to hardest row."""
