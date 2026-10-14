"""Every model input in one place, with its source.

Each input carries its citation as data rather than as a comment, so the
input table in the dissertation appendix can be generated straight from this
file and always matches what the model actually ran with.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Sequence


SOURCED = "sourced"
UNSOURCED = "unsourced"
ASSUMPTION = "assumption"
STANDARD = "standard"


@dataclass(frozen=True)
class ModelInput:
    symbol: str
    value: float
    unit: str
    source: str
    status: str = SOURCED
    low: Optional[float] = None
    high: Optional[float] = None
    note: str = ""

    def low_and_high(self):
        return (self.low if self.low is not None else self.value,
                self.high if self.high is not None else self.value)


DROPLET_RADIUS = ModelInput(
    symbol="r_p", value=70.0e-6, unit="m",
    source="Yan et al. (2023); Grant et al. (2022); Chen et al. (2020)",
    low=52.5e-6, high=263.5e-6,
    note="Diameters 105 / 140 / 527 um: the lowest, median and highest volume "
         "median diameters reported for drone sprays, under a rule set before "
         "searching. Radius is half the diameter.",
)

SPRAY_LIQUID_DENSITY = ModelInput(
    symbol="rho_p", value=1000.0, unit="kg/m3",
    source="Betancourt et al. (2024)",
)

AIR_DENSITY = ModelInput(
    symbol="rho_a", value=1.225, unit="kg/m3",
    source="U.S. Standard Atmosphere (NOAA, NASA and USAF, 1976), sea level, 15 C",
    status=STANDARD,
    note="Not given in Betancourt et al. (2024), to whom the submitted "
         "chapter attributed it. Correction sheet item N4.",
)

AIR_VISCOSITY = ModelInput(
    symbol="mu_a", value=1.7894e-5, unit="Pa.s",
    source="U.S. Standard Atmosphere (NOAA, NASA and USAF, 1976), sea level, 15 C",
    status=STANDARD,
    note="Required by the Reynolds number in the piecewise drag law. Absent "
         "from the submitted parameter table; must be added to Ch3 s3.6.",
)

GRAVITY = ModelInput(
    symbol="g", value=9.81, unit="m/s2",
    source="Standard value", status=STANDARD,
)

LEAF_AREA_INDEX = ModelInput(
    symbol="LAI", value=2.8, unit="m2/m2",
    source="Almenares-Garlobo et al. (2015)",
    low=2.0, high=3.6,
    note="Central value is the midpoint of the reported citrus range.",
)

LIGHT_EXTINCTION_COEFFICIENT = ModelInput(
    symbol="k", value=0.575, unit="dimensionless",
    source="Zhang et al. (2014)",
    low=0.55, high=0.60,
    note="Broadleaf forest analogy adopted for citrus.",
)

DEEP_SHADE_LIGHT_LIMIT = ModelInput(
    symbol="tau", value=0.10, unit="dimensionless",
    source="Study definition, Ch3 s3.3.1 and s3.6.2",
    status=ASSUMPTION,
    note="A cell is classified refuge when I(x,z)/I0 <= 0.10, i.e. at or "
         "beyond 90 percent attenuation. With solid fruit, any threshold below "
         "the leaves-only floor gives the same refuge map.",
)

VIRUS_HALF_LIFE_IN_FULL_SUN = ModelInput(
    symbol="T_half_exposed", value=144.0, unit="h",
    source="Moore (2002), cited in Moore et al. (2015)",
    note="Sunny side of citrus trees: activity below half within 3-6 days. "
         "Sunny-side estimates span about 1.4-6 days; the conservative rule "
         "takes 6 days. lambda = ln(2)/144 = 0.00481 /h.",
)

VIRUS_HALF_LIFE_IN_FULL_SHADE = ModelInput(
    symbol="T_half_shaded", value=504.0, unit="h",
    source="Moore (2002), cited in Moore et al. (2015)",
    note="Shady side of citrus trees: still above half at 21 days. The "
         "conservative rule takes the 21-day lower bound. "
         "lambda = ln(2)/504 = 0.00138 /h.",
)

DRONE_HEIGHT_ABOVE_CANOPY = ModelInput(
    symbol="h_uav", value=1.25, unit="m",
    source="Han et al. (2026); Meng et al. (2022); Chen et al. (2025); "
           "Martinez-Guanter et al. (2020)",
    low=0.5, high=2.0,
    note="Height above the canopy top. Low, central and high are the lowest, "
         "median and highest heights flown in citrus UAV field studies; see "
         "the vault note Drone Settings Sources (D6).",
)

DRONE_SPEED = ModelInput(
    symbol="v_uav", value=3.0, unit="m/s",
    source="Han et al. (2026); Meng et al. (2022); Yan et al. (2023); "
           "Chen et al. (2025); Martinez-Guanter et al. (2020)",
    low=1.0, high=4.0,
    note="Lowest, median and highest flight speeds in citrus UAV field "
         "studies; see the vault note Drone Settings Sources (D6).",
)

DOWNWASH_SPEED = ModelInput(
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

SLICE_WIDTH = ModelInput(
    symbol="L_x", value=4.0, unit="m",
    source="Study definition, within the citrus canopy widths in Larbi et al. "
           "(2026) and Yan et al. (2023)", status=ASSUMPTION,
)

SLICE_DEPTH = ModelInput(
    symbol="L_z", value=3.0, unit="m",
    source="Study definition, within the citrus tree heights in Larbi et al. "
           "(2026) and Han et al. (2026)", status=ASSUMPTION,
)

CELL_SIZE = ModelInput(
    symbol="dx", value=0.05, unit="m",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

TIME_STEP = ModelInput(
    symbol="dt", value=0.01, unit="s",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
    note="Drag relaxation time is about 0.03 s for the smallest droplets "
         "(r_p = 52.5 um), so forward Euler at 0.01 s stays inside its "
         "stability limit of twice the relaxation time.",
)

DROPLETS_PER_RUN = ModelInput(
    symbol="N", value=10000, unit="droplets/run",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

LEAF_SPRAY_CAPTURE = ModelInput(
    symbol="eta", value=0.575, unit="dimensionless",
    source="Equal to k: the same leaf geometry as the light model "
           "(Campbell and Norman, 1998)",
    note="Spray capture per metre of path through foliage is k x LAD, using "
         "the extinction coefficient of the run's own canopy.",
)

FRUIT_SPRAY_CAPTURE = ModelInput(
    symbol="k_fruit", value=float("inf"), unit="1/m",
    source="Physical: the fruit is solid", status=STANDARD,
    note="Every droplet whose path enters a fruit cell is captured.",
)

FRUIT_RADIUS = ModelInput(
    symbol="r_fruit", value=0.04, unit="m",
    source="NSW DPI (2020): Palmer navel, average diameter 83 mm",
)

FRUIT_LIGHT_BLOCKING = ModelInput(
    symbol="tau_fruit", value=float("inf"), unit="dimensionless",
    source="Physical: the fruit is solid", status=STANDARD,
    note="Fruit blocks all direct light; every cell below a fruit is fully shaded.",
)

FRUIT_SPACING = ModelInput(
    symbol="s_fruit", value=0.55, unit="m",
    source="Model choice; fruit kept in the outer canopy after Whitney and "
           "Wheaton (1984)", status=ASSUMPTION,
    note="Spacing within a row, starting 0.35 m from the edge; alternate "
         "rows are offset by half a spacing.",
)

DOWNWASH_FADE_DEPTH = ModelInput(
    symbol="L_dw", value=0.5, unit="m",
    source="Zhang et al. (2022): fade depths of 0.4-1.3 m measured inside "
           "pear canopies",
    note="Downwash falls as exp(-z / L_dw) with depth below the canopy top.",
)

FIRST_FRUIT_FROM_LEFT_EDGE_M = 0.35
FRUIT_ROW_DEPTHS_M: Sequence[float] = (0.45, 0.95, 1.45)
CHECKPOINT_HOURS: Sequence[float] = (24.0, 48.0, 72.0)

ALL_INPUTS = [
    DROPLET_RADIUS, SPRAY_LIQUID_DENSITY, AIR_DENSITY, AIR_VISCOSITY, GRAVITY,
    LEAF_AREA_INDEX, LIGHT_EXTINCTION_COEFFICIENT, DEEP_SHADE_LIGHT_LIMIT,
    VIRUS_HALF_LIFE_IN_FULL_SUN, VIRUS_HALF_LIFE_IN_FULL_SHADE,
    DRONE_HEIGHT_ABOVE_CANOPY, DRONE_SPEED, DOWNWASH_SPEED,
    SLICE_WIDTH, SLICE_DEPTH, CELL_SIZE, TIME_STEP, DROPLETS_PER_RUN,
    LEAF_SPRAY_CAPTURE, FRUIT_SPRAY_CAPTURE, FRUIT_RADIUS,
    FRUIT_LIGHT_BLOCKING, FRUIT_SPACING, DOWNWASH_FADE_DEPTH,
]

SETTINGS_TESTED = {
    "drone_height": DRONE_HEIGHT_ABOVE_CANOPY,
    "drone_speed": DRONE_SPEED,
    "downwash_speed": DOWNWASH_SPEED,
    "droplet_radius": DROPLET_RADIUS,
}


def inputs_without_a_source():
    return [model_input for model_input in ALL_INPUTS if model_input.status == UNSOURCED]


def as_table():
    return [asdict(model_input) for model_input in ALL_INPUTS]


def central_values():
    return {model_input.symbol: model_input.value for model_input in ALL_INPUTS}
