"""Single source of truth for every model input.

Every value carries its citation as data, not as a comment, so that the
parameter table in the dissertation appendix can be generated directly from
this module and can never drift from what the code actually ran.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Sequence


SOURCED = "sourced"
UNSOURCED = "unsourced"
ASSUMPTION = "assumption"
STANDARD = "standard"


@dataclass(frozen=True)
class Param:
    symbol: str
    value: float
    unit: str
    source: str
    status: str = SOURCED
    low: Optional[float] = None
    high: Optional[float] = None
    note: str = ""

    def bounds(self):
        return (self.low if self.low is not None else self.value,
                self.high if self.high is not None else self.value)


DROPLET_RADIUS = Param(
    symbol="r_p", value=70.0e-6, unit="m",
    source="Yan et al. (2023); Grant et al. (2022); Chen et al. (2020)",
    low=52.5e-6, high=263.5e-6,
    note="Diameters 105 / 140 / 527 um: the lowest, median and highest volume "
         "median diameters reported for drone sprays, under a rule set before "
         "searching. Radius is half the diameter.",
)

SPRAY_DENSITY = Param(
    symbol="rho_p", value=1000.0, unit="kg/m3",
    source="Betancourt et al. (2024)",
)

AIR_DENSITY = Param(
    symbol="rho_a", value=1.225, unit="kg/m3",
    source="U.S. Standard Atmosphere (NOAA, NASA and USAF, 1976), sea level, 15 C",
    status=STANDARD,
    note="Not given in Betancourt et al. (2024), to whom the submitted "
         "chapter attributed it. Correction sheet item N4.",
)

AIR_VISCOSITY = Param(
    symbol="mu_a", value=1.7894e-5, unit="Pa.s",
    source="U.S. Standard Atmosphere (NOAA, NASA and USAF, 1976), sea level, 15 C",
    status=STANDARD,
    note="Required by the Reynolds number in the piecewise drag law. Absent "
         "from the submitted parameter table; must be added to Ch3 s3.6.",
)

GRAVITY = Param(
    symbol="g", value=9.81, unit="m/s2",
    source="Standard value", status=STANDARD,
)

LAI_TOTAL = Param(
    symbol="LAI", value=2.8, unit="m2/m2",
    source="Almenares-Garlobo et al. (2015)",
    low=2.0, high=3.6,
    note="Central value is the midpoint of the reported citrus range.",
)

EXTINCTION_COEFF = Param(
    symbol="k", value=0.575, unit="dimensionless",
    source="Zhang et al. (2014)",
    low=0.55, high=0.60,
    note="Broadleaf forest analogy adopted for citrus.",
)

REFUGE_THRESHOLD = Param(
    symbol="tau", value=0.10, unit="dimensionless",
    source="Study definition, Ch3 s3.3.1 and s3.6.2",
    status=ASSUMPTION,
    note="A cell is classified refuge when I(x,z)/I0 <= 0.10, i.e. at or "
         "beyond 90 percent attenuation. With solid fruit, any threshold below "
         "the leaves-only floor gives the same refuge map.",
)

HALF_LIFE_EXPOSED = Param(
    symbol="T_half_exposed", value=144.0, unit="h",
    source="Moore (2002), cited in Moore et al. (2015)",
    note="Sunny side of citrus trees: activity below half within 3-6 days. "
         "Sunny-side estimates span about 1.4-6 days; the conservative rule "
         "takes 6 days. lambda = ln(2)/144 = 0.00481 /h.",
)

HALF_LIFE_SHADED = Param(
    symbol="T_half_shaded", value=504.0, unit="h",
    source="Moore (2002), cited in Moore et al. (2015)",
    note="Shady side of citrus trees: still above half at 21 days. The "
         "conservative rule takes the 21-day lower bound. "
         "lambda = ln(2)/504 = 0.00138 /h.",
)

UAV_ALTITUDE = Param(
    symbol="h_uav", value=1.25, unit="m",
    source="Han et al. (2026); Meng et al. (2022); Chen et al. (2025); "
           "Martinez-Guanter et al. (2020)",
    low=0.5, high=2.0,
    note="Height above the canopy top. Low, central and high are the lowest, "
         "median and highest heights flown in citrus UAV field studies; see "
         "the vault note Drone Settings Sources (D6).",
)

UAV_VELOCITY = Param(
    symbol="v_uav", value=3.0, unit="m/s",
    source="Han et al. (2026); Meng et al. (2022); Yan et al. (2023); "
           "Chen et al. (2025); Martinez-Guanter et al. (2020)",
    low=1.0, high=4.0,
    note="Lowest, median and highest flight speeds in citrus UAV field "
         "studies; see the vault note Drone Settings Sources (D6).",
)

DOWNWASH_VELOCITY = Param(
    symbol="v_dw", value=8.5, unit="m/s",
    source="Yallappa et al. (2024), Figure 12",
    low=6.5, high=10.0,
    note="Air speed straight below the centre of a six-rotor spraying UAV "
         "at 1, 2 and 3 m below the rotors with 0, 5 and 10 kg payloads, "
         "read from the bar chart to 0.1 m/s. Low, central and high are the "
         "lowest, median and highest of the nine readings, rounded to "
         "0.5 m/s. Measured in hover over a test rig; see the vault note "
         "Drone Settings Sources (D6).",
)

DOMAIN_WIDTH = Param(
    symbol="L_x", value=4.0, unit="m",
    source="Study definition, within the citrus canopy widths in Larbi et al. "
           "(2026) and Yan et al. (2023)", status=ASSUMPTION,
)

DOMAIN_DEPTH = Param(
    symbol="L_z", value=3.0, unit="m",
    source="Study definition, within the citrus tree heights in Larbi et al. "
           "(2026) and Han et al. (2026)", status=ASSUMPTION,
)

CELL_SIZE = Param(
    symbol="dx", value=0.05, unit="m",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

TIME_STEP = Param(
    symbol="dt", value=0.01, unit="s",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
    note="Drag relaxation time is about 0.03 s for the smallest droplets "
         "(r_p = 52.5 um), so forward Euler at 0.01 s stays inside its "
         "stability limit of twice the relaxation time.",
)

N_DROPLETS = Param(
    symbol="N", value=10000, unit="droplets/run",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

INTERCEPTION_EFFICIENCY = Param(
    symbol="eta", value=0.575, unit="dimensionless",
    source="Equal to k: the same leaf geometry as the light model "
           "(Campbell and Norman, 1998)",
    note="Spray capture per metre of path through foliage is k x LAD, using "
         "the extinction coefficient of the run's own canopy.",
)

FRUIT_INTERCEPTION = Param(
    symbol="k_fruit", value=float("inf"), unit="1/m",
    source="Physical: the fruit is solid", status=STANDARD,
    note="Every droplet whose path enters a fruit cell is captured.",
)

FRUIT_RADIUS = Param(
    symbol="r_fruit", value=0.04, unit="m",
    source="NSW DPI (2020): Palmer navel, average diameter 83 mm",
)

FRUIT_OPTICAL_DEPTH = Param(
    symbol="tau_fruit", value=float("inf"), unit="dimensionless",
    source="Physical: the fruit is solid", status=STANDARD,
    note="Fruit blocks all direct light; every cell below a fruit is fully shaded.",
)

FRUIT_SPACING = Param(
    symbol="s_fruit", value=0.55, unit="m",
    source="Model choice; fruit kept in the outer canopy after Whitney and "
           "Wheaton (1984)", status=ASSUMPTION,
    note="Spacing within a row, starting 0.35 m from the edge; alternate "
         "rows are offset by half a spacing.",
)

DOWNWASH_DECAY_LENGTH = Param(
    symbol="L_dw", value=0.5, unit="m",
    source="Zhang et al. (2022): fade depths of 0.4-1.3 m measured inside "
           "pear canopies",
    note="Downwash falls as exp(-z / L_dw) with depth below the canopy top.",
)

FRUIT_FIRST_X_M = 0.35
FRUIT_ROW_DEPTHS_M: Sequence[float] = (0.45, 0.95, 1.45)
CHECKPOINTS_H: Sequence[float] = (24.0, 48.0, 72.0)

ALL_PARAMS = [
    DROPLET_RADIUS, SPRAY_DENSITY, AIR_DENSITY, AIR_VISCOSITY, GRAVITY,
    LAI_TOTAL, EXTINCTION_COEFF, REFUGE_THRESHOLD,
    HALF_LIFE_EXPOSED, HALF_LIFE_SHADED,
    UAV_ALTITUDE, UAV_VELOCITY, DOWNWASH_VELOCITY,
    DOMAIN_WIDTH, DOMAIN_DEPTH, CELL_SIZE, TIME_STEP, N_DROPLETS,
    INTERCEPTION_EFFICIENCY, FRUIT_INTERCEPTION, FRUIT_RADIUS,
    FRUIT_OPTICAL_DEPTH, FRUIT_SPACING, DOWNWASH_DECAY_LENGTH,
]

SWEEP_PARAMS = [UAV_ALTITUDE, UAV_VELOCITY, DOWNWASH_VELOCITY, DROPLET_RADIUS]


def unsourced():
    return [p for p in ALL_PARAMS if p.status == UNSOURCED]


def as_table():
    return [asdict(p) for p in ALL_PARAMS]


def baseline():
    return {p.symbol: p.value for p in ALL_PARAMS}
