"""Stage 1: the canopy and the light inside it.

Builds a 2D slice through a citrus canopy as a grid of square cells, works out
the share of full sunlight that reaches each cell (Beer-Lambert law), and
marks the cells in deep shade. Design decisions D1 and D3 are implemented here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

import params

NEXT_TO_FRUIT_CELLS = 2


@dataclass
class Fruit:
    position_across: float
    depth: float
    radius: float = params.FRUIT_RADIUS.value


@dataclass
class Canopy:
    cells_across: int
    cells_deep: int
    cell_size: float
    column_boundaries: np.ndarray
    row_boundaries: np.ndarray
    column_positions: np.ndarray
    row_depths: np.ndarray
    leaf_area_density: np.ndarray
    light_blocking: np.ndarray
    share_of_full_light: np.ndarray
    deep_shade: np.ndarray
    contains_fruit: np.ndarray
    light_extinction_coefficient: float
    fruits: List[Fruit] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.column_boundaries[-1]

    @property
    def depth(self) -> float:
        return self.row_boundaries[-1]

    def cell_containing(self, position_across: float, depth: float) -> Tuple[int, int]:
        column = int(position_across / self.cell_size)
        row = int(depth / self.cell_size)
        return column, row

    def is_inside(self, position_across: float, depth: float) -> bool:
        return 0.0 <= position_across < self.width and 0.0 <= depth < self.depth


def default_fruits(slice_width: float, slice_depth: float) -> List[Fruit]:
    """Fruit placed through the outer and middle canopy, where citrus bears it.

    Without fruit the light would depend on depth alone, and the model could
    not tell a shaded spot beside a fruit from open inner canopy, which
    research question 1 needs it to do. The fruit sit in evenly spaced rows,
    with every second row shifted along by half a spacing.
    """
    spacing = params.FRUIT_SPACING.value
    positions_across = np.arange(params.FIRST_FRUIT_FROM_LEFT_EDGE_M, slice_width, spacing)
    fruits = []
    for fruit_row_number, depth in enumerate(params.FRUIT_ROW_DEPTHS_M):
        shift = 0.0 if fruit_row_number % 2 == 0 else spacing / 2.0
        for position in positions_across:
            if position + shift < slice_width:
                fruits.append(Fruit(position_across=float(position + shift), depth=float(depth)))
    return fruits


def build_canopy(leaf_area_index: float = None,
                 light_extinction_coefficient: float = None,
                 slice_width: float = None,
                 slice_depth: float = None,
                 cell_size: float = None,
                 fruits: List[Fruit] = None) -> Canopy:
    leaf_area_index = params.LEAF_AREA_INDEX.value if leaf_area_index is None else leaf_area_index
    light_extinction_coefficient = (params.LIGHT_EXTINCTION_COEFFICIENT.value
                                    if light_extinction_coefficient is None
                                    else light_extinction_coefficient)
    slice_width = params.SLICE_WIDTH.value if slice_width is None else slice_width
    slice_depth = params.SLICE_DEPTH.value if slice_depth is None else slice_depth
    cell_size = params.CELL_SIZE.value if cell_size is None else cell_size

    cells_across = int(round(slice_width / cell_size))
    cells_deep = int(round(slice_depth / cell_size))
    column_boundaries = np.linspace(0.0, slice_width, cells_across + 1)
    row_boundaries = np.linspace(0.0, slice_depth, cells_deep + 1)
    column_positions = 0.5 * (column_boundaries[:-1] + column_boundaries[1:])
    row_depths = 0.5 * (row_boundaries[:-1] + row_boundaries[1:])

    leaf_area_per_cubic_metre = leaf_area_index / slice_depth
    leaf_area_density = np.full((cells_across, cells_deep), leaf_area_per_cubic_metre)

    leaf_area_above = np.tile(row_depths * leaf_area_per_cubic_metre, (cells_across, 1))
    light_blocked_by_leaves = light_extinction_coefficient * leaf_area_above

    if fruits is None:
        fruits = default_fruits(slice_width, slice_depth)

    light_blocked_by_fruit = np.zeros((cells_across, cells_deep))
    contains_fruit = np.zeros((cells_across, cells_deep), dtype=bool)
    for fruit in fruits:
        fruit_columns = np.abs(column_positions - fruit.position_across) <= fruit.radius
        rows_below_fruit = row_depths > fruit.depth
        light_blocked_by_fruit[np.ix_(fruit_columns, rows_below_fruit)] += params.FRUIT_LIGHT_BLOCKING.value
        fruit_rows = np.abs(row_depths - fruit.depth) <= fruit.radius
        contains_fruit[np.ix_(fruit_columns, fruit_rows)] = True

    light_blocking = light_blocked_by_leaves + light_blocked_by_fruit
    share_of_full_light = np.exp(-light_blocking)
    deep_shade = share_of_full_light <= params.DEEP_SHADE_LIGHT_LIMIT.value

    return Canopy(
        cells_across=cells_across, cells_deep=cells_deep, cell_size=cell_size,
        column_boundaries=column_boundaries, row_boundaries=row_boundaries,
        column_positions=column_positions, row_depths=row_depths,
        leaf_area_density=leaf_area_density, light_blocking=light_blocking,
        share_of_full_light=share_of_full_light, deep_shade=deep_shade,
        contains_fruit=contains_fruit,
        light_extinction_coefficient=light_extinction_coefficient, fruits=fruits,
    )


def canopy_zones(canopy: Canopy) -> dict:
    """The outer, middle and inner thirds of the canopy, plus the deep-shade
    cells next to fruit."""
    rows_per_third = canopy.cells_deep // 3
    outer = np.zeros((canopy.cells_across, canopy.cells_deep), dtype=bool)
    middle = np.zeros((canopy.cells_across, canopy.cells_deep), dtype=bool)
    inner = np.zeros((canopy.cells_across, canopy.cells_deep), dtype=bool)
    outer[:, :rows_per_third] = True
    middle[:, rows_per_third:2 * rows_per_third] = True
    inner[:, 2 * rows_per_third:] = True
    near_fruit = _cells_near(canopy.contains_fruit, NEXT_TO_FRUIT_CELLS)
    shaded_next_to_fruit = canopy.deep_shade & near_fruit
    return {"outer": outer, "middle": middle, "inner": inner,
            "shaded_next_to_fruit": shaded_next_to_fruit}


def _cells_near(marked: np.ndarray, distance_in_cells: int) -> np.ndarray:
    result = marked.copy()
    for _ in range(distance_in_cells):
        grown = result.copy()
        grown[1:, :] |= result[:-1, :]
        grown[:-1, :] |= result[1:, :]
        grown[:, 1:] |= result[:, :-1]
        grown[:, :-1] |= result[:, 1:]
        result = grown
    return result
