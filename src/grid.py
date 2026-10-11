"""Stage 1 - grid construction.

Builds the 2D canopy cross-section, the Beer-Lambert UV field and the
refuge classification. Design decisions D1 and D3 are implemented here.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

import params as P



@dataclass
class Fruit:
    x: float
    z: float
    radius: float = P.FRUIT_RADIUS.value


@dataclass
class Canopy:
    nx: int
    nz: int
    dx: float
    x_edges: np.ndarray
    z_edges: np.ndarray
    x_centres: np.ndarray
    z_centres: np.ndarray
    lad: np.ndarray
    optical_depth: np.ndarray
    intensity_ratio: np.ndarray
    refuge: np.ndarray
    fruit_mask: np.ndarray
    k: float
    fruits: List[Fruit] = field(default_factory=list)

    def cell_of(self, x: float, z: float) -> Tuple[int, int]:
        i = int(x / self.dx)
        j = int(z / self.dx)
        return i, j

    def in_domain(self, x: float, z: float) -> bool:
        return 0.0 <= x < self.x_edges[-1] and 0.0 <= z < self.z_edges[-1]


def default_fruits(width: float, depth: float) -> List[Fruit]:
    """Discrete fruit bodies giving the canopy horizontal structure.

    Without these the UV field is a function of depth alone and the model
    cannot distinguish a calyx recess from open inner canopy, which RQ1
    requires it to do. Positions are a regular staggered arrangement in the
    outer and mid canopy where fruit is borne.
    """
    spacing = P.FRUIT_SPACING.value
    xs = np.arange(P.FRUIT_FIRST_X_M, width, spacing)
    fruits = []
    for row, z in enumerate(P.FRUIT_ROW_DEPTHS_M):
        offset = 0.0 if row % 2 == 0 else spacing / 2.0
        for x in xs:
            if x + offset < width:
                fruits.append(Fruit(x=float(x + offset), z=float(z)))
    return fruits


def build(lai_total: float = None,
          k: float = None,
          width: float = None,
          depth: float = None,
          dx: float = None,
          fruits: List[Fruit] = None) -> Canopy:
    lai_total = P.LAI_TOTAL.value if lai_total is None else lai_total
    k = P.EXTINCTION_COEFF.value if k is None else k
    width = P.DOMAIN_WIDTH.value if width is None else width
    depth = P.DOMAIN_DEPTH.value if depth is None else depth
    dx = P.CELL_SIZE.value if dx is None else dx

    nx = int(round(width / dx))
    nz = int(round(depth / dx))
    x_edges = np.linspace(0.0, width, nx + 1)
    z_edges = np.linspace(0.0, depth, nz + 1)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])

    lad_value = lai_total / depth
    lad = np.full((nx, nz), lad_value)

    cumulative_lai = np.tile(z_centres * lad_value, (nx, 1))
    tau_leaf = k * cumulative_lai

    if fruits is None:
        fruits = default_fruits(width, depth)

    tau_fruit = np.zeros((nx, nz))
    fruit_mask = np.zeros((nx, nz), dtype=bool)
    for f in fruits:
        shadow = np.abs(x_centres - f.x) <= f.radius
        below = z_centres > f.z
        tau_fruit[np.ix_(shadow, below)] += P.FRUIT_OPTICAL_DEPTH.value
        occupied_z = np.abs(z_centres - f.z) <= f.radius
        fruit_mask[np.ix_(shadow, occupied_z)] = True

    optical_depth = tau_leaf + tau_fruit
    intensity_ratio = np.exp(-optical_depth)
    refuge = intensity_ratio <= P.REFUGE_THRESHOLD.value

    return Canopy(
        nx=nx, nz=nz, dx=dx,
        x_edges=x_edges, z_edges=z_edges,
        x_centres=x_centres, z_centres=z_centres,
        lad=lad, optical_depth=optical_depth,
        intensity_ratio=intensity_ratio, refuge=refuge,
        fruit_mask=fruit_mask, k=k, fruits=fruits,
    )


def zones(canopy: Canopy) -> dict:
    """Canopy depth zones plus the fruit-adjacent microhabitat."""
    nz = canopy.nz
    third = nz // 3
    outer = np.zeros((canopy.nx, nz), dtype=bool)
    mid = np.zeros((canopy.nx, nz), dtype=bool)
    inner = np.zeros((canopy.nx, nz), dtype=bool)
    outer[:, :third] = True
    mid[:, third:2 * third] = True
    inner[:, 2 * third:] = True
    microhabitat = canopy.refuge & _dilate(canopy.fruit_mask, 2)
    return {"outer": outer, "mid": mid, "inner": inner,
            "microhabitat": microhabitat}


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(cells):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    return out
