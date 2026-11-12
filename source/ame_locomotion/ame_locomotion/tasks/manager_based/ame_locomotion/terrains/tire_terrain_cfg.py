"""Static, ray-castable 2 x 4 tire arrays for Stage 2 fine-tuning."""

import numpy as np
import trimesh

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def tire_array_terrain(difficulty, cfg):
    """Build closed hollow-cylinder meshes; dimensions are sampled per array.

    Both center spacings equal the sampled outer diameter. Thus all horizontal
    and vertical neighbors are tangent, leaving only the natural diamond gaps
    at the intersections of each 2 x 2 group.
    """
    # Difficulty is the terrain-row curriculum level.  At level 0 every
    # dimension is at its easiest permitted value; later rows progressively
    # widen the sampled range until the final rows cover the full spec.
    outer = np.random.uniform(cfg.outer_diameter_range[0], np.interp(difficulty, (0.0, 1.0), cfg.outer_diameter_range))
    inner = np.random.uniform(cfg.inner_diameter_range[0], np.interp(difficulty, (0.0, 1.0), cfg.inner_diameter_range))
    height = np.random.uniform(cfg.width_range[0], np.interp(difficulty, (0.0, 1.0), cfg.width_range))
    origin = np.array([cfg.size[0] / 2 - 2.0, cfg.size[1] / 2, 0.0])
    ground = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.1))
    ground.apply_translation((cfg.size[0] / 2, cfg.size[1] / 2, -0.05))
    meshes = [ground]
    # Flat annular cylinder: vertical inner/outer walls and horizontal top
    # and bottom faces. The central hole remains open down to the ground.
    angles = np.arange(cfg.ring_segments) * 2 * np.pi / cfg.ring_segments
    cos, sin = np.cos(angles), np.sin(angles)
    outer_radius, inner_radius = outer / 2, inner / 2
    vertices = np.empty((cfg.ring_segments * 4, 3), dtype=float)
    vertices[0::4] = np.column_stack((outer_radius * cos, outer_radius * sin, np.zeros(cfg.ring_segments)))
    vertices[1::4] = np.column_stack((outer_radius * cos, outer_radius * sin, np.full(cfg.ring_segments, height)))
    vertices[2::4] = np.column_stack((inner_radius * cos, inner_radius * sin, np.zeros(cfg.ring_segments)))
    vertices[3::4] = np.column_stack((inner_radius * cos, inner_radius * sin, np.full(cfg.ring_segments, height)))
    faces = []
    for i in range(cfg.ring_segments):
        next_i = (i + 1) % cfg.ring_segments
        outer_bottom, outer_top, inner_bottom, inner_top = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        next_outer_bottom, next_outer_top = 4 * next_i, 4 * next_i + 1
        next_inner_bottom, next_inner_top = 4 * next_i + 2, 4 * next_i + 3
        # Outer wall, inner wall, top annulus, and bottom annulus.
        faces.extend(
            (
                (outer_bottom, next_outer_bottom, next_outer_top),
                (outer_bottom, next_outer_top, outer_top),
                (inner_bottom, inner_top, next_inner_top),
                (inner_bottom, next_inner_top, next_inner_bottom),
                (outer_top, next_outer_top, next_inner_top),
                (outer_top, next_inner_top, inner_top),
                (outer_bottom, next_inner_bottom, next_outer_bottom),
                (outer_bottom, inner_bottom, next_inner_bottom),
            )
        )
    for col in range(4):
        for row in range(2):
            mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces, process=False)
            mesh.apply_translation(origin + [1.0 + outer / 2 + col * outer, (row - 0.5) * outer, 0])
            meshes.append(mesh)
    return meshes, origin


@configclass
class TireArrayTerrainCfg(SubTerrainBaseCfg):
    function = tire_array_terrain
    outer_diameter_range: tuple[float, float] = (0.5, 0.6)
    inner_diameter_range: tuple[float, float] = (0.3, 0.36)
    width_range: tuple[float, float] = (0.145, 0.185)
    ring_segments: int = 48


TIRE_ARRAY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 5.0), border_width=2.0,
    # Rows are difficulty levels; columns provide several randomized arrays
    # at each level.  Terrain levels select rows during resets.
    num_rows=8, num_cols=8, curriculum=True, use_cache=False,
    sub_terrains={"tires": TireArrayTerrainCfg(proportion=1.0)},
)
