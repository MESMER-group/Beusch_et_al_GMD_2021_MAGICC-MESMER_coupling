"""
Configuration file for cmip6-ng, tas, hist + all ssps

"""


# ---------------------------------------------------------------------------------

# directories:

basedir = "/your/path/"
basedir_cmip = "/your/cmip/path/"


# cmip-ng
gen = 6  # generations
dir_cmipng = basedir_cmip + "cmip" + str(gen) + "-ng/"

# observations
dir_obs = basedir + "mesmer/data/observations/"

# auxiliary data
dir_aux = basedir + "mesmer/data/auxiliary/"

# MESMER
dir_mesmer_params = basedir + "across_scen_T/mesmer/calibrated_parameters/"
dir_mesmer_emus = basedir + "across_scen_T/mesmer/emulations/"

# MAGICC
dir_magicc_probab_emus = (
    basedir
    + "across_scen_T/data/rcmip_phase_2/rcmip-phase-2-archive/rcmip-phase-2/data/processed/submission-database-hadcrut5-target/"
)
dir_magicc_esmspecific_emus = (
    basedir + "across_scen_T/data/rcmip_phase_1/data-for-lea-v20200518/"
)

# emulation statistics
dir_stats = basedir + "across_scen_T/statistics/"

# plots
dir_plots = basedir + "across_scen_T/plots/"

# configs that can be set for every run:

# esms
esms = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "CanESM5",
    "CESM2",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CNRM-CM6-1",
    "CNRM-CM6-1-HR",
    "CNRM-ESM2-1",
    "E3SM-1-1",
    "FGOALS-f3-L",
    "FGOALS-g3",
    "FIO-ESM-2-0",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "IPSL-CM6A-LR",
    "MCM-UA-1-0",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NESM3",
    "NorESM2-LM",
    "NorESM2-MM",
    "UKESM1-0-LL",
]

targs = ["tas"]  # emulated variables

reg_type = "ar6.land"

ref = {}
ref["type"] = "individ"  # alternatives: 'first','all'
ref["start"] = "1850"  # first included year
ref["end"] = "1900"  # last included year

threshold_land = 1 / 3

wgt_scen_tr_eq = True  # if True weigh each scenario equally (ie less weight to individ runs of scens with more ic members)

nr_emus_v = 600  # trade-off between stable quantiles for individ ESMs and a not too large probab ensemble
scen_seed_offset_v = 0  # 0 meaning same emulations drawn for each scen, if put a number will have different ones for each scen
max_iter_cv = 15  # max. nr of iterations in cross validation

# predictors (for global module)
preds = {}
preds["tas"] = {}  # predictors for the target variable tas
preds["hfds"] = {}
preds["tas"]["gt"] = ["saod"]
preds["hfds"]["gt"] = []
preds["tas"]["gv"] = []
preds["tas"]["g_all"] = preds["tas"]["gt"] + preds["tas"]["gv"]

# methods (for all modules)
methods = {}
methods["tas"] = {}  # methods for the target variable tas
methods["hfds"] = {}
methods["tas"]["gt"] = "LOWESS_OLSVOLC"  # global trend emulation method
methods["hfds"]["gt"] = "LOWESS"
methods["tas"]["gv"] = "AR"  # global variability emulation method
methods["tas"]["lt"] = "OLS"  # local trends emulation method
method_lt_each_gp_sep = True  # method local trends applied to each gp separately
methods["tas"]["lv"] = "OLS_AR1_sci"  # local variability emulation method

# ---------------------------------------------------------------------------------

# configs that should remain untouched:

# full list of esms (to have unique seed for each esm no matter how)
all_esms = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "AWI-CM-1-1-MR",
    "CAMS-CSM1-0",
    "CanESM5",
    "CESM2",
    "CESM2-WACCM",
    "CIESM",
    "CMCC-CM2-SR5",
    "CNRM-CM6-1",
    "CNRM-CM6-1-HR",
    "CNRM-ESM2-1",
    "E3SM-1-1",
    "EC-Earth3",
    "EC-Earth3-Veg",
    "EC-Earth3-Veg-LR",
    "FGOALS-f3-L",
    "FGOALS-g3",
    "FIO-ESM-2-0",
    "GFDL-CM4",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "IPSL-CM6A-LR",
    "MCM-UA-1-0",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NESM3",
    "NorESM2-LM",
    "NorESM2-MM",
    "UKESM1-0-LL",
]

# full list of scenarios that could be considered
scenarios = [
    "h-ssp585",
    "h-ssp370",
    "h-ssp460",
    "h-ssp245",
    "h-ssp534-over",
    "h-ssp434",
    "h-ssp126",
    "h-ssp119",
]

if scen_seed_offset_v == 0:
    scenarios_emus_v = ["all"]
else:
    scenarios_emus_v = scenarios

nr_emus = {}
nr_ts_emus_v = {}
seed = {}
i = 0
for esm in all_esms:
    seed[esm] = {}
    j = 0
    for scen in scenarios_emus_v:
        seed[esm][scen] = {}
        seed[esm][scen]["gv"] = i + j * scen_seed_offset_v
        seed[esm][scen]["lv"] = i + j * scen_seed_offset_v + 1000000
        j += 1
    i += 1


# ---------------------------------------------------------------------------------

# information about loaded data:

# cmip-ng
# Data downloaded from ESGF (https://esgf-node.llnl.gov/projects/esgf-llnl/) and pre-processed according to Brunner et al. 2020 (https://doi.org/10.5281/zenodo.3734128)
# assumes folder structure / file name as in cmip-ng archives at ETHZ -> see mesmer.io.load_cmipng.file_finder_cmipng() for details
# - global mean stratospheric AOD, monthly, 1850-"2020" (0 after 2012), downloaded from KNMI climate explorer in August 2020, no pre-processing
