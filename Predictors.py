import pandas as pd
import numpy as np
import config_param
from math import ceil
from util_function import smooth_epidata
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from var_ind_beta_un import *
from var_simulate_pred_un import *



#import preprocessed data
hosp_cumu_s_org= np.loadtxt('hosp_cumu_s.csv', delimiter=',')
#hosp_cumu= pd.read_csv('hosp_cumu.csv')
hosp_dat = pd.read_csv('hosp_dat.csv')
popu = np.loadtxt('us_states_population_data.txt') #population of each state

maxt = hosp_dat.shape[1]
days = maxt
hosp_cumu = np.nancumsum(np.nan_to_num(hosp_dat), axis=1)
num_states = len(popu)
H = pd.DataFrame()
row = 0
for i in range(hosp_dat.shape[0]):  # State
    for j in range(hosp_dat.shape[1]):  # Day
        if j <= config_param.bin_size-1:
            H.at[row, 0] = np.nan
            H.at[row, 1] = hosp_dat.iloc[i, j]
        else:
            H.at[row, 0] = hosp_dat.iloc[i, j-config_param.bin_size]
            H.at[row, 1] = hosp_dat.iloc[i, j]
        row += 1

for wks_ahead in range(1, config_param.weeks_ahead+1):
    temp = H[1][(config_param.bin_size * wks_ahead):].to_list()
    last_data_pnt = maxt + 1 - (config_param.bin_size * wks_ahead)  # Adjust index based on your data
    for i in range(last_data_pnt, maxt):
        idx = [x for x in range(i, len(H), maxt)]
        H.loc[idx, H.shape[1]] = np.nan


wks = int(ceil((days - config_param.offset) / config_param.bin_size - 1)) #offset of 11 days
predT = np.full((config_param.npredictors, num_states, 1), np.nan)
smooth_factor = 14

for x in range(wks, -1, -1):
    print(x)
   
    wks_back = x 
    hosp_cumu_s = hosp_cumu_s_org
    T_full = days - wks_back * config_param.bin_size  # Computing T_full by subtracting 7*wks_back from days
    thisday = T_full  
    ns = hosp_cumu_s.shape[0]  # Getting the number of rows in hosp_cumu_s
    
    if wks_back == 0:
        # if wks_back is zero, use the entire array
        hosp_cumu_s = hosp_cumu_s[:, :]
        hosp_cumu = hosp_cumu[:, :]
    else:
        # if wks_back is non-zero, then proceed with the original slicing
        hosp_cumu_s = hosp_cumu_s[:, :-(wks_back*config_param.bin_size)]
        hosp_cumu = hosp_cumu[:, :-(wks_back*config_param.bin_size)]
    
    un_array =popu[:, None] * 0 + np.array([50, 100, 150])  # define un_list array
    
    scen_list = []
    for scenario in product(*config_param.hyperparams_lists):
        scen_list.append(list(scenario))

    scen_list = np.array(scen_list)
    
    net_hosp_A = np.zeros((len(scen_list)*config_param.num_dh_rates_sample, ns, config_param.horizon))
    
    
    net_h_cell = [None] * len(scen_list)  
    
    base_hosp = hosp_cumu[:, T_full-1]

    def process_scenario(args):
        simnum, scenario_details = args
        halpha, rlag, un, s = scenario_details
        rr = config_param.rlags[rlag.astype(int) - 1]
        print(rr)
        un = un_array[:, un.astype(int) - 1]
        sliced_array = hosp_cumu_s[:, :-rr] if rr != 0 else hosp_cumu_s
        hosp_rate, fC, ci_h = var_ind_beta_un(sliced_array, 0, halpha, config_param.hk, un, popu, config_param.hjp, 0.95,s)
        temp_res = np.zeros((config_param.num_dh_rates_sample, ns, config_param.horizon))
        for rr in range(config_param.num_dh_rates_sample):
            this_rate = hosp_rate.copy()  
            
            if rr != (config_param.num_dh_rates_sample + 1) // 2:  # Using integer division here
                for cid in range(ns):
                    this_rate[cid] = ci_h[cid][:, 0] + (ci_h[cid][:, 1] - ci_h[cid][:, 0]) * (rr) / (config_param.num_dh_rates_sample - 1)
                    
            pred_hosps = var_simulate_pred_un(hosp_cumu_s, 0, this_rate, popu, config_param.hk, config_param.horizon, config_param.hjp, un, base_hosp)
            #print(pred_hosps.shape)
            h_start = 0 
            temp_res[rr, :, :] = pred_hosps[:, h_start:h_start+config_param.horizon] - base_hosp.reshape(-1,1)

        #print(temp_res)
        return temp_res
        
    results = []
    with ProcessPoolExecutor() as executor:
        # Map each scenario (along with its simnum for ordering) to the process_scenario function
        tasks = [(simnum, scen_list[simnum]) for simnum in range(scen_list.shape[0])]
        #futures = {executor.submit(process_scenario, (simnum, scenario)): simnum for simnum, scenario in enumerate(scen_list)}
        futures = {executor.submit(process_scenario, task): task[0] for task in tasks}
        
        # Collect the results as they are completed
        for future in as_completed(futures):
            simnum = futures[future]
            try:
                net_h_cell[simnum] = future.result()
            except Exception as exc:
                print(f"Scenario {simnum} generated an exception: {exc}")
    

     
    
    for simnum in range(scen_list.shape[0]):
        net_hosp_A[simnum, :, :] = np.nanmean(net_h_cell[simnum], axis=0)
    indd = int(config_param.npredictors/config_param.num_dh_rates_sample)
    p = np.squeeze(net_hosp_A[0:indd, :, :])
    predictors = np.diff(p, axis=2)
    lo = predictors[:, :, 0:config_param.weeks_ahead*config_param.bin_size]
# Concatenate slices along the third dimension (axis=2 in Python)
    slices_to_concat = [lo[:, :, i*config_param.bin_size:min((i+1)*config_param.bin_size, lo.shape[2])] for i in range(config_param.weeks_ahead)]
    new_slices = np.concatenate(slices_to_concat, axis=0)
    predT = np.concatenate((predT, new_slices), axis=2)
    #print(new_slices.shape)
    #print(predT.shape)


predT = predT[:, :, 1:]
first = np.full((56, 11, config_param.weeks_ahead), np.nan)

predT = np.concatenate((first, predT), axis=2)

predILI = pd.DataFrame()
for i in range(56):  
    temp_df = pd.DataFrame(predT[:, i, :min(maxt, predT.shape[2])].T)
    predILI = pd.concat([predILI, temp_df], ignore_index=True)

""" To FIX
predictor_size = int(config_param.npredictors/config_param.num_dh_rates_sample)

for i in range(config_param.weeks_ahead):
    start_col = i * 11
    end_col = start_col + 11
    first[:, :, i] = hosp_dat.iloc[:, start_col:end_col]

tempf = np.full((56, 11, predictor_size * config_param.weeks_ahead), np.nan)  # This now matches the target shape.

for i in range(5):
    tempf[:, :, i * 54:(i + 1) * predictor_size] = np.tile(first[:, :, i][:, :, np.newaxis], (1, 1, predictor_size))

tempf = np.transpose(tempf, (2, 0, 1))

"""
