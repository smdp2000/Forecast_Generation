import numpy as np
from datetime import datetime
############################
offset = 11
zero_date = datetime(2021, 9, 1)
days_back = 0
bin_size = 7
#### model determining
weeks_ahead = 5
smooth_factor = 14
####
num_dh_rates_sample = 5
season_start = datetime(2023, 9, 30)
season_end = zero_date
season_start_day = (season_start - season_end).days

########################### HyperParameters
###%%%%%%%TO ADD %%%%%%%%%%%%%%%%%%%%%
# predictor_models = ["var_ind_beta_un", "var_simulate_pred_un"]
# hyper_params.p1 = [...]
# hyper_params.p2 = [...]
# ...


##############################
rlags = np.array([0, 7])
rlag_list = np.arange(1, len(rlags) + 1)
un_list = np.array([1, 2, 3])
halpha_list = np.arange(0.9, 0.98 + 0.01, 0.04)  # [0.9, 0.92, ..., 0.98]
S = [0, 0.2, 0.4]
hyperparams_lists = [halpha_list, rlag_list, un_list, S]
hk = 2
hjp = 7
########################

npredictors = (len(S) * len(halpha_list) * len(un_list) * len(rlag_list))*(weeks_ahead)
horizon = (weeks_ahead+1)*bin_size 

######################## regress
decay_factor = 0.99
wks_back = 1
default_n_estimators = 100
quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]



