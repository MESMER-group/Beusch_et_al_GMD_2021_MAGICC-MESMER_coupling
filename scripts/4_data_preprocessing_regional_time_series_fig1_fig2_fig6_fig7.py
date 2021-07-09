import sys

sys.path.append("../")


import joblib
import numpy as np
import scmdata
import scmdata.database

# load in configurations used in this script
import configs.config_across_scen_T_cmip6ng_sspall_default as cfg

# load in the MESMER tools
from mesmer.io import load_cmipng, load_mesmer_output, load_regs_ls_wgt_lon_lat
from mesmer.utils import convert_dict_to_arr, extract_land

print("Get config information sorted")
magicc_scenarios_to_run = ["ssp*"]
time = np.arange(1850, 2101)
scenarios = cfg.scenarios
esms = cfg.esms
nr_esms = len(esms)
nr_emus_lv = cfg.nr_emus_v
q = [0.05, 0.5, 0.95]  # quantiles to derive

print("Load in MAGICC output")
db = scmdata.database.ScmDatabase(cfg.dir_magicc_probab_emus)
magicc_gsat_output = scmdata.run_append(
    [
        db.load(scenario=s, variable="Surface Air Temperature Change")
        for s in magicc_scenarios_to_run
    ]
)


print("Preprocess MAGICC probabilistic forced warming output")
# ATTENTION: requires scmdata v>0.9.0 and <0.10.0, otherwise time filter doesn't work
gt_T = {}
for scen in scenarios:
    # cut out 1850-2100
    gt_T[scen] = magicc_gsat_output.filter(scenario=scen[2:], year=time).values
    # rebaseline to 1850-1900
    gt_T[scen] = gt_T[scen] - np.mean(gt_T[scen][:, :51], axis=1).reshape([600, 1])
nr_gt_T, nr_ts = gt_T[scen].shape
nr_emus_per_esm = nr_gt_T * nr_emus_lv

print("Load in the global variability emulations from MESMER")
emus_gv = {}
for esm in esms:
    emus_gv[esm] = load_mesmer_output("emus_gv", cfg, esm_str=esm)


print("Derive the global temperature change quantiles")
g_quantiles = {}
g_all = np.zeros([nr_emus_per_esm * nr_esms, nr_ts])
for scen in scenarios:
    print("start with scen", scen)
    g_quantiles[scen] = {}
    for s, esm in enumerate(esms):
        # collection of full ensemble of global temperatures
        g_all[
            s * nr_emus_per_esm : s * nr_emus_per_esm + nr_emus_per_esm
        ] = np.concatenate(
            [gt_T[scen][k] + emus_gv[esm]["all"] for k in np.arange(nr_gt_T)]
        )
    (
        g_quantiles[scen][q[0]],
        g_quantiles[scen][q[1]],
        g_quantiles[scen][q[2]],
    ) = np.quantile(g_all, q=q, axis=0)
joblib.dump(g_quantiles, cfg.dir_stats + "global_quantiles.pkl")

del emus_gv, g_all, g_quantiles

print(
    "Load in the local forced warming parameters and the local variability emulations from MESMER"
)
params_lt = {}
emus_lv = {}
for esm in esms:
    print("start loading esm", esm)
    params_lt[esm] = load_mesmer_output("params_lt", cfg, esm_str=esm)
    emus_lv[esm] = load_mesmer_output("emus_lv", cfg, esm_str=esm)

print("Load in the regional information")
# need to load in single ESM file to be able to get regional dictionary that is needed
esm = esms[0]
scen = "h-ssp585"
targ = cfg.targs[0]
tas_g_dict = {}
tas_g = {}
tas_g_dict[esm] = {}
tas_g_dict[esm][scen], _, lon, lat, _ = load_cmipng(targ, esm, scen, cfg)
tas_g[esm] = convert_dict_to_arr(tas_g_dict[esm])
# load in the constant files
reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(cfg.reg_type, lon, lat)
# extract land
tas, reg_dict, ls = extract_land(
    tas_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
)


print("Derive the regional quantiles")
regs = ["global_land", "ENA"]
for reg in regs:
    if reg == "ENA":
        reg_idx = reg_dict["abbrevs"].index(reg)
        wgt_gps_l = reg_dict["wgt_gps_l"][reg_idx]
    elif reg == "global_land":
        wgt_gps_l = np.sum(reg_dict["wgt_gps_l"], axis=0)
    print("start with region:", reg)
    nr_gps = len(wgt_gps_l)
    print("create regional averages of local variability for all ESMs")
    reg_emus_lv = {}
    for esm in esms:
        reg_emus_lv[esm] = np.average(
            emus_lv[esm]["all"]["tas"], weights=wgt_gps_l, axis=2
        )
    reg_quantiles = {}
    for scen in scenarios:
        print("start with scenario:", scen)
        reg_quantiles[scen] = {}
        reg_all = np.zeros([nr_emus_per_esm * nr_esms, nr_ts])
        print(
            "compute local forced warming for all ESMs and merge with local variability for full emulations"
        )
        # slow since recompute full globe for every reg and use loops -> def. potential for cleaner solution
        # (if I had more memory space could also do this just once at beginning)
        for s, esm in enumerate(esms):
            # local forced warming for each grid point
            emus_lt = np.zeros([nr_gt_T, nr_ts, nr_gps])
            for gp in np.arange(nr_gps):
                emus_lt[:, :, gp] = (
                    params_lt[esm]["coef_gttas"]["tas"][gp] * gt_T[scen]
                    + params_lt[esm]["intercept"]["tas"][gp]
                )
            # regional average
            reg_emus_lt = np.average(emus_lt, weights=wgt_gps_l, axis=2)
            # collection of full ensemble of regional temperature changes
            reg_all[
                s * nr_emus_per_esm : s * nr_emus_per_esm + nr_emus_per_esm
            ] = np.concatenate(
                [reg_emus_lt[i] + reg_emus_lv[esm] for i in np.arange(nr_gt_T)]
            )
        print("derive quantiles")
        (
            reg_quantiles[scen][q[0]],
            reg_quantiles[scen][q[1]],
            reg_quantiles[scen][q[2]],
        ) = np.quantile(reg_all, q=q, axis=0)
    print("save quantiles and sample emulations")
    joblib.dump(reg_quantiles, cfg.dir_stats + reg + "_quantiles.pkl")
del reg_all, emus_lt, reg_quantiles

print("Derive grid point level quantiles")
gp = 1317  # gp in ENA
print("start with grid point", gp)
gp_quantiles = {}
for scen in scenarios:
    print("start with scenario:", scen)
    gp_quantiles[scen] = {}
    gp_all = np.zeros([nr_emus_per_esm * nr_esms, nr_ts])
    print(
        "compute local forced warming for all ESMs and merge with local variability for full emulations at this grid point"
    )
    # slow (if I had more memory space could also do this just once at beginning)
    for s, esm in enumerate(esms):
        # local forced warming at specific grid point
        gp_emus_lt = (
            params_lt[esm]["coef_gttas"]["tas"][gp] * gt_T[scen]
            + params_lt[esm]["intercept"]["tas"][gp]
        )
        # collection of full ensemble of temperature changes at this grid point
        gp_all[
            s * nr_emus_per_esm : s * nr_emus_per_esm + nr_emus_per_esm
        ] = np.concatenate(
            [
                gp_emus_lt[i] + emus_lv[esm]["all"]["tas"][:, :, gp]
                for i in np.arange(nr_gt_T)
            ]
        )

    print("derive quantiles")
    (
        gp_quantiles[scen][q[0]],
        gp_quantiles[scen][q[1]],
        gp_quantiles[scen][q[2]],
    ) = np.quantile(gp_all, q=q, axis=0)

print("save quantiles and sample emulations")
joblib.dump(gp_quantiles, cfg.dir_stats + "gp_level/" + str(gp) + "_quantiles.pkl")
