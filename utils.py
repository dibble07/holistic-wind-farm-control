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
OPEX_GWy = (102 + 116) / 2 * 1e6
LIFESPAN = 30
DOWNTIME = 0.02


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
def run_sim(wfm=wfm_low, x=wt9_x, y=wt9_y, yaw=0, ws=np.arange(0, 31)):
    sim_res = wfm(
        x=x,
        y=y,
        tilt=0,
        yaw=yaw,
        n_cpu=None,
        ws=ws,
        wd=270,
    )
    return sim_res


# calculate metrics
def calc_metrics(sim_res, sim_res_base, show=False):
    # unpack values
    nt = len(sim_res.wt)
    power_installed = (
        sim_res.windFarmModel.windTurbines.powerCtFunction.power_ct_tab[0].max()
        / 1e9
        * nt
    )

    # calculate turbulent kinetic energy ratio
    tke_vel = (
        ((sim_res.TI_eff * sim_res.ws) ** 2 * sim_res.P / sim_res.P.sum()).sum().values
    )
    tke_vel_base = (
        (
            (sim_res_base.TI_eff * sim_res_base.ws) ** 2
            * sim_res_base.P
            / sim_res_base.P.sum()
        )
        .sum()
        .values
    )
    tke_ratio = tke_vel / tke_vel_base

    # calculate metrics of interest
    lcoe = (
        (OPEX_GWy + CAPEX_GW / LIFESPAN)
        * power_installed
        / (sim_res.aep().sum().values * (1 - DOWNTIME * tke_ratio) * 1000)
    )
    cap_fac = (
        sim_res.aep().sum().values
        * (1 - DOWNTIME * tke_ratio)
        / (power_installed * 365.25 * 24)
        * 100
    )

    # print values
    if show:
        print(f"LCoE [USD/MWh]: {lcoe:,.3f}")
        print(f"Capacity factor [%]: {cap_fac:,.3f}")

    return lcoe, cap_fac
