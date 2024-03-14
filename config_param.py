import numpy as np
from datetime import datetime
############################
zero_date = datetime(2021, 9, 1)
days_back = 0
bin_size = 7
weeks_ahead = 5
num_dh_rates_sample = 5
season_start = datetime(2023, 9, 30)
season_end = zero_date
season_start_day = (season_start - season_end).days

########################### HyperParameters
rlags = np.array([0, 7])
rlag_list = np.arange(1, len(rlags) + 1)
un_list = np.array([1, 2, 3])
halpha_list = np.arange(0.9, 0.98 + 0.01, 0.04)  # [0.9, 0.92, ..., 0.98]
S = [0, 0.2, 0.4]
########################

npredictors = (len(S) * len(halpha_list) * len(un_list) * len(rlag_list))*(weeks_ahead)
horizon = (weeks_ahead+1)*bin_size 


