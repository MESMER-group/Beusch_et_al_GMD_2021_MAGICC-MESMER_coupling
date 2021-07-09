import sys

sys.path.append("../")


import joblib
import numpy as np
import scmdata
import scmdata.database

# load in configurations used in this script
import configs.config_across_scen_T_cmip6ng_sspall_default as cfg

# load in the MESMER tools
from mesmer.io import load_mesmer_output

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


print("Load in ESM-specific local forced response paramters from MESMER")

params_lt = {}
for esm in esms:
    params_lt[esm] = load_mesmer_output("params_lt", cfg, esm_str=esm)


print("Derive maps of specific quantiles for selected time slots and scenarios")

nr_gps = len(params_lt[esm]["intercept"]["tas"])  # nr grid points
nr_gt_T = gt_T[scen].shape[0]  # nr forced warming trajectories
nr_emus_per_esm = nr_gt_T * nr_emus_lv  # nr emulations per ESM


ts = 2014
print("start with ts", ts)
idx_ts = np.where(time == ts)[0][0]


scen = "h-ssp245"
print("start with scen", scen)

ts_all = np.zeros([nr_emus_per_esm * nr_esms, nr_gps])
print(ts_all.shape)
for s, esm in enumerate(esms):
    # derive the ESM-specific emulated local forced warming distribution
    ts_emus_lt = np.zeros([nr_gt_T, nr_gps])
    for gp in np.arange(nr_gps):
        ts_emus_lt[:, gp] = (
            params_lt[esm]["coef_gttas"]["tas"][gp] * gt_T[scen][:, idx_ts]
            + params_lt[esm]["intercept"]["tas"][gp]
        )

    # load in the ESM-specific emulated local variability
    emus_lv = load_mesmer_output("emus_lv", cfg, esm_str=esm)
    emus_lv = emus_lv["all"]["tas"][:, idx_ts]

    # derive the ESM-specific full emulations and fill them into the full MAGICC-MESMER probabilistic ensemble
    ts_all[
        s * nr_emus_per_esm : s * nr_emus_per_esm + nr_emus_per_esm
    ] = np.concatenate([ts_emus_lt[k] + emus_lv for k in np.arange(nr_gt_T)])

# derive quantiles
for q_sel in q:
    print("start with quantile", q_sel)
    ts_quantile = np.quantile(ts_all, q=q_sel, axis=0)
    joblib.dump(
        ts_quantile,
        cfg.dir_stats
        + "gp_level/fields/"
        + str(ts)
        + "_quantile"
        + str(q_sel)
        + "_"
        + scen
        + ".pkl",
    )


# repeat same code but with loop for 2050 and 2100 and h-ssp119 + h-ssp370
for ts in [2050, 2100]:
    print("start with ts", ts)
    idx_ts = np.where(time == ts)[0][0]
    for scen in ["h-ssp119", "h-ssp370"]:
        print("start with scen", scen)
        ts_all = np.zeros([nr_emus_per_esm * nr_esms, nr_gps])
        print(ts_all.shape)
        for s, esm in enumerate(esms):
            # derive the ESM-specific emulated local forced warming distribution
            ts_emus_lt = np.zeros([nr_gt_T, nr_gps])
            for gp in np.arange(nr_gps):
                ts_emus_lt[:, gp] = (
                    params_lt[esm]["coef_gttas"]["tas"][gp] * gt_T[scen][:, idx_ts]
                    + params_lt[esm]["intercept"]["tas"][gp]
                )

            # load in the ESM-specific emulated local variability
            emus_lv = load_mesmer_output("emus_lv", cfg, esm_str=esm)
            emus_lv = emus_lv["all"]["tas"][:, idx_ts]

            # derive the ESM-specific full emulations and fill them into the full MAGICC-MESMER probabilistic ensemble
            ts_all[
                s * nr_emus_per_esm : s * nr_emus_per_esm + nr_emus_per_esm
            ] = np.concatenate([ts_emus_lt[k] + emus_lv for k in np.arange(nr_gt_T)])

        # derive quantiles
        for q_sel in q:
            print("start with quantile", q_sel)
            ts_quantile = np.quantile(ts_all, q=q_sel, axis=0)
            joblib.dump(
                ts_quantile,
                cfg.dir_stats
                + "gp_level/fields/"
                + str(ts)
                + "_quantile"
                + str(q_sel)
                + "_"
                + scen
                + ".pkl",
            )
