from py_wake.deficit_models import FugaDeficit, NoWakeDeficit, ZongGaussianDeficit
from py_wake.deflection_models import FugaDeflection, JimenezWakeDeflection
from py_wake.examples.data.hornsrev1 import (
    Hornsrev1Site,
    V80,
    wt9_x,
    wt9_y,
    wt16_x,
    wt16_y,
    wt_x,
    wt_y,
)
from py_wake.rotor_avg_models import CGIRotorAvg
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import STF2017TurbulenceModel
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind

# loads cost values
# capex and opex https://atb.nrel.gov/electricity/2023/index
# technology lifespans - https://atb.nrel.gov/electricity/2023/definitions#costrecoveryperiod

CAPEX_GW = (3150 + 3901) / 2 * 1e6
OPEX_GWy = (102 + 116) / 2 * 1e6
LIFESPAN = 30

# load wind farm model
def get_wind_farm_model(nt: int = 9):
    # load turbine locations
    nt_opt = {
        9: (wt9_x, wt9_y),
        16: (wt16_x, wt16_y),
        80: (wt_x, wt_y),
    }
    if nt in nt_opt:
        x, y = nt_opt[nt]
    else:
        raise ValueError(
            f"Number of turbines must be one these options: {nt_opt.keys()}. Value provided: {nt}"
        )

    # load farm models
    wfm_high = All2AllIterative(
        site=Hornsrev1Site(),
        windTurbines=V80(),
        wake_deficitModel=FugaDeficit(),
        superpositionModel=LinearSum(),
        deflectionModel=FugaDeflection(),
        turbulenceModel=STF2017TurbulenceModel(),
        rotorAvgModel=CGIRotorAvg(7),
    )
    wfm_low = PropagateDownwind(
        site=Hornsrev1Site(),
        windTurbines=V80(),
        wake_deficitModel=ZongGaussianDeficit(use_effective_ws=True),
        superpositionModel=LinearSum(),
        deflectionModel=JimenezWakeDeflection(),
        turbulenceModel=STF2017TurbulenceModel(),
        rotorAvgModel=None,
    )
    wfm_lossless = PropagateDownwind(
        site=Hornsrev1Site(),
        windTurbines=V80(),
        wake_deficitModel=NoWakeDeficit(),
        superpositionModel=LinearSum(),
        deflectionModel=None,
        turbulenceModel=None,
        rotorAvgModel=None,
    )

    return x, y, wfm_high, wfm_low, wfm_lossless
