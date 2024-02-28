import logging

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from py_wake.deficit_models import FugaDeficit, NoWakeDeficit, ZongGaussianDeficit
from py_wake.deflection_models import JimenezWakeDeflection
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
from scipy import optimize

# define colormap that shows bad, neutral and good intuitively
cmap = LinearSegmentedColormap.from_list(
    name="mymap", colors=["tab:red", "w", "tab:green"]
)

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
WS_DEFAULT = np.arange(0, 25.01, 1)
WD_DEFAULT = np.arange(0, 360, 30)

# load farm models
wfm_high = All2AllIterative(
    site=Hornsrev1Site(),
    windTurbines=V80(),
    wake_deficitModel=FugaDeficit(),
    superpositionModel=LinearSum(),
    deflectionModel=JimenezWakeDeflection(),
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


# calculate metrics
def run_sim(
    wfm, x, y, yaw, ws, wd, sim_res_ref=None, Sector_frequency=None, P=None, show=False
):
    # run simulation
    sim_res = wfm(x=x, y=y, tilt=0, yaw=yaw, n_cpu=1, ws=ws, wd=wd)

    # ensure probabilities (wind direction and speed) total 1
    if Sector_frequency is None:
        Sector_frequency = sim_res.Sector_frequency
        if not np.isclose(Sector_frequency.sum(), 1):
            logging.warning(
                f"Sector frequency renormalised as total probability was {Sector_frequency.sum().values}"
            )
            Sector_frequency = Sector_frequency / Sector_frequency.sum()
    if P is None:
        P = sim_res.P
        if not np.isclose(P.sum(), 1):
            logging.warning(f"P renormalised as total probability was {P.sum().values}")
            P = P / P.sum()

    # unpack values
    rated_power = wfm.windTurbines.powerCtFunction.power_ct_tab[0].max() / 1e9
    Sector_frequency = Sector_frequency.sel(wd=wd)
    P = P.sel(wd=wd)

    # subset baseline simulation
    if sim_res_ref is None:
        logging.warning("no reference simulation provided")
        sim_res_ref = sim_res.sel(wd=wd)
    else:
        for attr in ["wd", "ws", "wt"]:
            curr = getattr(sim_res, attr).values.tolist()
            curr = curr if isinstance(curr, list) else [curr]
            ref = getattr(sim_res_ref, attr).values.tolist()
            ref = ref if isinstance(ref, list) else [ref]
            if curr != ref:
                msg = f"Simulations have different {attr} values. {curr} vs {ref}"
                raise ValueError(msg)
        sim_res_ref = sim_res_ref.sel(wd=wd)

    # calculate turbulent kinetic energy ratio
    tke_vel = ((sim_res.TI_eff * sim_res.ws) ** 2 * P).sum(["wd", "ws"])
    tke_vel_base = ((sim_res_ref.TI_eff * sim_res_ref.ws) ** 2 * P).sum(["wd", "ws"])
    tke_ratio = tke_vel / tke_vel_base

    # calculate metrics of interest
    uptime = 1 - DOWNTIME * tke_ratio
    sim_res["energy"] = (sim_res.Power * P).sum("ws") * 8760 / 1e9 * uptime
    fixed_cost = (OPEX_FIXED_GWy + CAPEX_GW / LIFESPAN) * rated_power * Sector_frequency
    variable_cost = OPEX_VAR_GWh * sim_res.energy
    sim_res["lcoe"] = (fixed_cost + variable_cost) / (sim_res.energy * 1000)
    sim_res["cap_fac"] = sim_res.energy / (
        Sector_frequency * rated_power * (sim_res.wt * 0 + 1) * 365.25 * 24
    )

    # aggregate metrics
    sim_res["lcoe_direction"] = (sim_res.lcoe * (sim_res.energy * 1000)).sum("wt") / (
        sim_res.energy * 1000
    ).sum("wt")
    sim_res["lcoe_overall"] = (sim_res.lcoe * (sim_res.energy * 1000)).sum(
        ["wt", "wd"]
    ) / (sim_res.energy * 1000).sum(["wt", "wd"])
    sim_res["cap_fac_direction"] = sim_res.cap_fac.weighted(Sector_frequency).mean("wt")
    sim_res["cap_fac_overall"] = sim_res.cap_fac.weighted(Sector_frequency).mean(
        ["wt", "wd"]
    )

    # print values
    if show:
        print(f"Annual energy [GWh]: {sim_res.energy.sum():,.3f}")
        print(f"LCoE [USD/MWh]: {sim_res.lcoe_overall:,.3f}")
        print(f"Capacity factor [%]: {sim_res.cap_fac_overall*100:,.3f}")

    return sim_res, Sector_frequency, P


# define constants
YAW_SCALE = 15


# optimise for a single direction
def optimise_direction(wd, sim_res_ref_low, sim_res_ref_high, Sector_frequency, P):
    # unpack values
    wfm_low = sim_res_ref_low.windFarmModel
    wfm_high = sim_res_ref_high.windFarmModel
    sim_res_ref_low = sim_res_ref_low.sel(wd=[wd])
    sim_res_ref_high = sim_res_ref_high.sel(wd=[wd])
    x = sim_res_ref_low.x.values.tolist()
    y = sim_res_ref_low.y.values.tolist()
    ws = sim_res_ref_low.ws.values.tolist()
    wt = sim_res_ref_low.wt.values.tolist()
    yaw_shape = (len(wt), 1, len(ws))

    # check matching parameters in reference simulations
    for attr in ["wd", "ws", "wt"]:
        low = getattr(sim_res_ref_low, attr).values.tolist()
        low = low if isinstance(low, list) else [low]
        high = getattr(sim_res_ref_high, attr).values.tolist()
        high = high if isinstance(high, list) else [high]
        if low != high:
            msg = f"Simulations have different {attr} values. {low} vs {high}"
            raise ValueError(msg)

    # optimise for power output across independent wind speeds
    logging.info("starting power based optimisation")
    yaw_opt_power = np.full(yaw_shape, np.nan)
    next_x0 = np.ones(len(wt)) / YAW_SCALE
    for i, ws_ in enumerate(ws):
        # define objective function for power
        def obj_power_single(yaw_norm):
            sim_res, _, _ = run_sim(
                wfm=wfm_low,
                x=x,
                y=y,
                yaw=yaw_norm * YAW_SCALE,
                ws=[ws_],
                wd=[wd],
                sim_res_ref=sim_res_ref_low.sel(ws=[ws_]),
                Sector_frequency=Sector_frequency,
                P=P,
            )
            power = sim_res.Power.sum("wt")
            power_ref = sim_res_ref_low.sel(ws=[ws_]).Power.sum("wt")
            obj = -(power / power_ref).sel(ws=ws_, wd=wd).values.tolist()
            return obj

        assert np.isclose(obj_power_single(np.zeros(len(wt))), -1)

        res = optimize.minimize(
            fun=obj_power_single,
            x0=next_x0,
            method="SLSQP",
            options=dict(ftol=1e-8),
        )
        next_x0 = res.x
        yaw_opt_power[:, :, i] = res.x.reshape(-1, 1) * YAW_SCALE

    # define objective function for lcoe
    def obj_lcoe_single(yaw_norm):
        sim_res, _, _ = run_sim(
            wfm=wfm_high,
            x=x,
            y=y,
            yaw=yaw_norm.reshape(yaw_shape) * YAW_SCALE,
            ws=ws,
            wd=[wd],
            sim_res_ref=sim_res_ref_high,
            Sector_frequency=Sector_frequency,
            P=P,
        )
        obj = (
            sim_res.lcoe_direction / sim_res_ref_high.lcoe_direction
        ).values.tolist()[0]
        return obj

    assert np.isclose(obj_lcoe_single(np.zeros(yaw_shape)), 1)

    # optimise for lcoe across all wind speeds
    logging.info("starting LCoE based optimisation")
    res = optimize.minimize(
        fun=obj_lcoe_single,
        x0=yaw_opt_power.ravel() / YAW_SCALE,
        method="SLSQP",
        options=dict(ftol=1e-8),
    )
    yaw_opt_lcoe = res.x.reshape(yaw_shape) * YAW_SCALE

    # assess result
    obj_power = obj_lcoe_single(yaw_opt_power.ravel() / YAW_SCALE)
    obj_lcoe = obj_lcoe_single(yaw_opt_lcoe.ravel() / YAW_SCALE)
    if obj_power < obj_lcoe:
        logging.warning(
            f"optimising based on power resulted in a better lcoe reduction: {obj_power:.6f} vs {obj_lcoe:.6f}"
        )
    if not res.success:
        logging.warning(f"optimisation was not successful: {res.message}")
    logging.info(f"optimisation complete (fun = {res.fun:.6f}, nit = {res.nit:.0f})")

    return np.squeeze(yaw_opt_power), np.squeeze(yaw_opt_lcoe)
