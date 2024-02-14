import numpy as np
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
# technology lifespans https://atb.nrel.gov/electricity/2023/definitions#costrecoveryperiod
# downtime https://blog.windurance.com/the-truth-about-pitch-system-downtime and https://energyfollower.com/how-long-do-wind-turbines-last/
CAPEX_GW = (3150 + 3901) / 2 * 1e6
_opex1 = (0.29, 102)
_opex2 = (0.46, 116)
_grad = (_opex2[1] - _opex1[1]) / (_opex2[0] - _opex1[0])
OPEX_FIXED_GWy = (_opex1[1] - _grad * _opex1[0]) * 1e6
OPEX_VAR_GWh = _grad / (365.25 * 24)
LIFESPAN = 30
DOWNTIME = 0.02

# define default ranges
WS_DEFAULT = np.arange(0, 31)
WD_DEFAULT = np.arange(0, 360, 15)

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


# run simulation
def run_sim(wfm=wfm_low, x=wt9_x, y=wt9_y, yaw=0, ws=WS_DEFAULT, wd=WD_DEFAULT):
    sim_res = wfm(
        x=x,
        y=y,
        tilt=0,
        yaw=yaw,
        n_cpu=1,
        ws=ws,
        wd=wd,
    )
    return sim_res


# calculate metrics
def calc_metrics(sim_res, sim_res_base, show=False):
    # unpack values
    rated_power = (
        sim_res.windFarmModel.windTurbines.powerCtFunction.power_ct_tab[0].max() / 1e9
    )

    # scaled probabilities
    P = sim_res.P / sim_res.P.sum()
    P_base = sim_res_base.P / sim_res_base.P.sum()
    Sector_frequency = sim_res.Sector_frequency / sim_res.Sector_frequency.sum()

    # calculate turbulent kinetic energy ratio
    tke_vel = ((sim_res.TI_eff * sim_res.ws) ** 2 * P).sum(["wd", "ws"])
    tke_vel_base = ((sim_res_base.TI_eff * sim_res_base.ws) ** 2 * P_base).sum(
        ["wd", "ws"]
    )
    tke_ratio = tke_vel / tke_vel_base

    # calculate metrics of interest
    uptime = 1 - DOWNTIME * tke_ratio
    aep = sim_res.aep().sum("ws") * uptime
    fixed_cost = (OPEX_FIXED_GWy + CAPEX_GW / LIFESPAN) * rated_power * Sector_frequency
    variable_cost = OPEX_VAR_GWh * aep
    lcoe = (fixed_cost + variable_cost) / (aep * 1000)
    cap_fac = aep / (
        Sector_frequency * rated_power * (sim_res.wt * 0 + 1) * 365.25 * 24
    )

    # print values
    if show:
        print(f"AEP [GWh]: {aep.sum():,.3f}")
        print(f"LCoE [USD/MWh]: {(lcoe * aep * 1000).sum()/(aep * 1000).sum():,.3f}")
        print(
            f"Capacity factor [%]: {100*cap_fac.weighted(Sector_frequency).mean():,.3f}"
        )

    return aep, lcoe, cap_fac
