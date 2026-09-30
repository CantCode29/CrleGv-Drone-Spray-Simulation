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
    symbol="r_p", value=0.5e-3, unit="m",
    source="Betancourt et al. (2024)",
    low=0.3e-3, high=0.8e-3,
    note="Sweep levels 0.3 / 0.5 / 0.8 mm. Reported as radius throughout; "
         "RQ2 wording corrected from diameter to radius to match.",
)

SPRAY_DENSITY = Param(
    symbol="rho_p", value=1000.0, unit="kg/m3",
    source="Betancourt et al. (2024)",
)

AIR_DENSITY = Param(
    symbol="rho_a", value=1.225, unit="kg/m3",
    source="Standard value for dry air at sea level, 15 C, 101.325 kPa",
    status=STANDARD,
    note="Not given in Betancourt et al. (2024), to whom the submitted "
         "chapter attributed it. Correction sheet item N4.",
)

AIR_VISCOSITY = Param(
    symbol="mu_a", value=1.81e-5, unit="Pa.s",
    source="Standard value for dry air at 25 C, 101.325 kPa",
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
         "beyond 90 percent attenuation.",
)

HALF_LIFE_EXPOSED = Param(
    symbol="T_half_exposed", value=24.0, unit="h",
    source="Mwanza et al. (2022)",
    note="Sunny exposed canopy surface. lambda = ln(2)/24 = 0.0289 /h.",
)

HALF_LIFE_SHADED = Param(
    symbol="T_half_shaded", value=72.0, unit="h",
    source="Fritsch and Huber (1989), cited in Moore et al. (2015)",
    note="Conservative shaded inner-canopy upper bound. "
         "lambda = ln(2)/72 = 0.0096 /h. NOT attributable to Mwanza et al. "
         "(2022) - this misattribution is error B1 in the correction log.",
)

LC50 = Param(
    symbol="LC50", value=2.29e4, unit="OB/mL",
    source="Mwanza et al. (2022)",
)

TEMPERATURE = Param(
    symbol="T", value=25.0, unit="degC",
    source="Model assumption, held constant", status=ASSUMPTION,
)

UAV_ALTITUDE = Param(
    symbol="h_uav", value=1.25, unit="m",
    source="Han et al. (2026); Meng et al. (2022); Chen et al. (2025); "
           "Martinez-Guanter et al. (2019)",
    low=0.5, high=2.0,
    note="Height above the canopy top. Low, central and high are the lowest, "
         "median and highest heights flown in citrus UAV field studies; see "
         "the vault note Drone Settings Sources (D6).",
)

UAV_VELOCITY = Param(
    symbol="v_uav", value=3.0, unit="m/s",
    source="Han et al. (2026); Meng et al. (2022); Yan et al. (2023); "
           "Chen et al. (2025); Martinez-Guanter et al. (2019)",
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
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

DOMAIN_DEPTH = Param(
    symbol="L_z", value=3.0, unit="m",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

CELL_SIZE = Param(
    symbol="dx", value=0.05, unit="m",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

TIME_STEP = Param(
    symbol="dt", value=0.01, unit="s",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
    note="Droplet drag relaxation time is approximately 3 s at r_p = 0.5 mm, "
         "so forward Euler at 0.01 s is stable by two orders of magnitude.",
)

N_DROPLETS = Param(
    symbol="N", value=10000, unit="droplets/run",
    source="Study definition, Ch3 s3.7.3", status=ASSUMPTION,
)

CHECKPOINTS_H: Sequence[float] = (24.0, 48.0, 72.0)

ALL_PARAMS = [
    DROPLET_RADIUS, SPRAY_DENSITY, AIR_DENSITY, AIR_VISCOSITY, GRAVITY,
    LAI_TOTAL, EXTINCTION_COEFF, REFUGE_THRESHOLD,
    HALF_LIFE_EXPOSED, HALF_LIFE_SHADED, LC50, TEMPERATURE,
    UAV_ALTITUDE, UAV_VELOCITY, DOWNWASH_VELOCITY,
    DOMAIN_WIDTH, DOMAIN_DEPTH, CELL_SIZE, TIME_STEP, N_DROPLETS,
]

SWEEP_PARAMS = [UAV_ALTITUDE, UAV_VELOCITY, DOWNWASH_VELOCITY, DROPLET_RADIUS]


def unsourced():
    return [p for p in ALL_PARAMS if p.status == UNSOURCED]


def as_table():
    return [asdict(p) for p in ALL_PARAMS]


def baseline():
    return {p.symbol: p.value for p in ALL_PARAMS}
