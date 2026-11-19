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
        low = self.value if self.low is None else self.low
        high = self.value if self.high is None else self.high
        return low, high


DROPLET_RADIUS = ModelInput(
    symbol="r_p", value=70.0e-6, unit="m",
    source="Yan et al. (2023); Grant et al. (2022); Chen et al. (2020)",
    low=52.5e-6, high=263.5e-6,
    note="Droplet sizes of 105, 140 and 527 um across: the smallest, middle "
         "and largest sizes reported for drone sprays. The radius is half of that.",
)

SPRAY_LIQUID_DENSITY = ModelInput(
    symbol="rho_p", value=1000.0, unit="kg/m3",
    source="Betancourt et al. (2024)",
)

AIR_DENSITY = ModelInput(
    symbol="rho_a", value=1.225, unit="kg/m3",
    source="U.S. Standard Atmosphere (NOAA, NASA and USAF, 1976), sea level, 15 C",
    status=STANDARD,
)

AIR_VISCOSITY = ModelInput(
    symbol="mu_a", value=1.7894e-5, unit="Pa.s",
    source="U.S. Standard Atmosphere (NOAA, NASA and USAF, 1976), sea level, 15 C",
    status=STANDARD,
    note="Needed to work out the air drag on a droplet.",
)

GRAVITY = ModelInput(
    symbol="g", value=9.81, unit="m/s2",
    source="Standard value", status=STANDARD,
)

LEAF_AREA_INDEX = ModelInput(
    symbol="LAI", value=2.8, unit="m2/m2",
    source="Almenares-Garlobo et al. (2015)",
    low=2.0, high=3.6,
    note="The middle of the range measured on citrus.",
)

LIGHT_EXTINCTION_COEFFICIENT = ModelInput(
    symbol="k", value=0.575, unit="none",
    source="Zhang et al. (2014)",
    low=0.55, high=0.60,
    note="How strongly the leaves block light. Measured on broadleaf trees "
         "and used here for citrus.",
)

DEEP_SHADE_LIGHT_LIMIT = ModelInput(
    symbol="tau", value=0.10, unit="none",
    source="Study definition", status=ASSUMPTION,
    note="A cell is in deep shade when it gets 10% of full sunlight or less.",
)

VIRUS_HALF_LIFE_IN_FULL_SUN = ModelInput(
    symbol="T_half_exposed", value=144.0, unit="h",
    source="Moore (2002), cited in Moore et al. (2015)",
    note="On the sunny side of citrus trees the virus lost half its activity "
         "in 3 to 6 days. Published values run from about 1.4 to 6 days, and "
         "the slowest, 6 days, was used.",
)

VIRUS_HALF_LIFE_IN_FULL_SHADE = ModelInput(
    symbol="T_half_shaded", value=504.0, unit="h",
    source="Moore (2002), cited in Moore et al. (2015)",
    note="On the shady side of citrus trees more than half the virus was "
         "still active after 21 days, so 21 days was used.",
)

DRONE_HEIGHT_ABOVE_CANOPY = ModelInput(
    symbol="h_uav", value=1.25, unit="m",
    source="Han et al. (2026); Meng et al. (2022); Chen et al. (2025); "
           "Martinez-Guanter et al. (2020)",
    low=0.5, high=2.0,
    note="Height above the top of the trees. Low, middle and high are the "
         "lowest, middle and highest heights flown in drone trials on citrus.",
)

DRONE_SPEED = ModelInput(
    symbol="v_uav", value=3.0, unit="m/s",
    source="Han et al. (2026); Meng et al. (2022); Yan et al. (2023); "
           "Chen et al. (2025); Martinez-Guanter et al. (2020)",
    low=1.0, high=4.0,
    note="The lowest, middle and highest flight speeds in drone trials on citrus.",
)

DOWNWASH_SPEED = ModelInput(
    symbol="v_dw", value=8.5, unit="m/s",
    source="Yallappa et al. (2024), Figure 12",
    low=6.5, high=10.0,
    note="Air speed straight below a six-rotor spraying drone, measured 1, 2 "
         "and 3 m below the rotors while carrying 0, 5 and 10 kg, and read "
         "from the bar chart. Low, middle and high are the lowest, middle and "
         "highest of the nine readings, rounded to 0.5 m/s. The drone was "
         "hovering over a test rig.",
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
    source="Study definition", status=ASSUMPTION,
)

TIME_STEP = ModelInput(
    symbol="dt", value=0.01, unit="s",
    source="Study definition", status=ASSUMPTION,
    note="The smallest droplets take about 0.03 s to catch up with the air "
         "around them, so steps of 0.01 s are small enough to follow them.",
)

DROPLETS_PER_RUN = ModelInput(
    symbol="N", value=10000, unit="droplets",
    source="Study definition", status=ASSUMPTION,
)

LEAF_SPRAY_CAPTURE = ModelInput(
    symbol="eta", value=0.575, unit="none",
    source="Equal to k: leaves catch spray the same way they block light "
           "(Campbell and Norman, 1998)",
    note="The share of spray caught per metre of leaves is k times the leaf "
         "area density, using the k of the canopy being sprayed.",
)

FRUIT_SPRAY_CAPTURE = ModelInput(
    symbol="k_fruit", value=float("inf"), unit="1/m",
    source="Physical: the fruit is solid", status=STANDARD,
    note="Every droplet that reaches a fruit is caught by it.",
)

FRUIT_RADIUS = ModelInput(
    symbol="r_fruit", value=0.04, unit="m",
    source="NSW DPI (2020): Palmer navel, average diameter 83 mm",
)

FRUIT_LIGHT_BLOCKING = ModelInput(
    symbol="tau_fruit", value=float("inf"), unit="none",
    source="Physical: the fruit is solid", status=STANDARD,
    note="A fruit blocks all direct light, so every cell below it is fully shaded.",
)

FRUIT_SPACING = ModelInput(
    symbol="s_fruit", value=0.55, unit="m",
    source="Model choice; fruit kept in the outer canopy after Whitney and "
           "Wheaton (1984)", status=ASSUMPTION,
    note="Fruit are 0.55 m apart in a row, starting 0.35 m from the edge. "
         "Every second row is moved along by half that gap.",
)

DOWNWASH_FADE_DEPTH = ModelInput(
    symbol="L_dw", value=0.5, unit="m",
    source="Zhang et al. (2022): fade depths of 0.4-1.3 m measured inside "
           "pear trees",
    note="The downwash gets weaker below the top of the trees. Every 0.5 m "
         "down, it drops to about a third (37%) of its speed.",
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


def value_or_default(value, model_input):
    return model_input.value if value is None else value


def inputs_without_a_source():
    return [model_input for model_input in ALL_INPUTS if model_input.status == UNSOURCED]


def as_table():
    return [asdict(model_input) for model_input in ALL_INPUTS]


def central_values():
    return {model_input.symbol: model_input.value for model_input in ALL_INPUTS}
