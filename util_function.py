import numpy as np
from numpy import *
import scipy.optimize
import scipy.stats
from functools import partial
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import signal
from config_param import season_start_day
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

def hampel_filter_forloop(input_series, window_size=15, n_sigmas=4):
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution

    indices = []

    for i in range((window_size), (n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)

    return new_series, indices


def smooth_epidata(data_4, smooth_factor=14, week_correction=1, week_smoothing=1):
    if smooth_factor <= 0:
        return data_4

    df = pd.DataFrame(data_4)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.interpolate(method='linear', limit=2, limit_direction='both')

    data_4 =  df

    deldata = np.diff(data_4, axis=1)

    deldata = pd.DataFrame(deldata).T

    data_4_s = data_4

    maxt = data_4.shape[1]

    date_map = np.ceil((np.arange(1, maxt) - np.mod(maxt - 1, 7)) / 7).reshape(1, -1)

    cleandel = deldata

    if week_correction == 1:
        for cid in range(len(data_4)):
            week_dat = pd.DataFrame(data_4).loc[cid][list(range((maxt - 1) % 7, maxt, 7))].diff()[1:]

            

            clean_week_tmp, TF = hampel_filter_forloop(week_dat.to_numpy())

            week_dat_tmp = week_dat.to_list()

            for idx_out in TF:
                week_dat_tmp[idx_out] = np.nan

            clean_week = pd.DataFrame(week_dat_tmp).interpolate(method='linear')

            week_dat_1 = week_dat.to_list()
            week_dat_1.append(0)
            peak_idx = np.array([i for i in range(1, len(week_dat_1) - 1) if
                                 week_dat_1[i] > week_dat_1[i - 1] and week_dat_1[i] > week_dat_1[i + 1]])

            tf_vals = []
            for i in TF:
                if i in peak_idx:

                    tf_vals.append(i)

            for jj in tf_vals:
                get_idx = [w for w in range(len(date_map[0])) if date_map[0][w] == jj]
                for w in get_idx:
                    cleandel.iloc[w][cid] = clean_week.iloc[jj][0] / 7

        deldata = cleandel

    if week_smoothing == 1:
        temp = np.cumsum(deldata.T, axis=1)

        temp['new'] = 0  ### add a constant colum with value 0

        temp = temp[[list(temp.columns)[-1]] + list(temp.columns)[0:-1]]

        temp_array = np.array(temp)

        diff = np.diff(temp_array[:, (maxt - 1) % 7::7].T, axis=0)

        week_dat = pd.DataFrame(diff)

        xx = np.full((deldata.shape), np.nan)

        xx[int((maxt - 1 - 7 * week_dat.shape[0]) + 7) - 1:maxt - 1:7, :] = np.cumsum(week_dat, axis=0)

        xx = pd.DataFrame(xx).interpolate(method='linear', limit_direction='both')

        deldata.iloc[1:, :] = np.diff(xx, axis=0)

    deldata[deldata < 0] = 0

    for state_idx in range(deldata.shape[1]):
        deldata[state_idx] = deldata[state_idx].rolling(smooth_factor, min_periods=0).mean()

    data_4_s = np.concatenate((data_4.iloc[:, :1].to_numpy(), np.cumsum(deldata, axis=0).T), axis=1)

    return data_4_s



def opt_fun(X):
    def F(w):
        return X @ (1 / (1+np.exp(-w)))
    return F

def var_ind_beta_un(data_4, passengerFlow, alpha_l, k_l, un_fact, popu, jp_l, ret_conf, S, compute_region=None, window_size=None, extra_imm=None):
    
    maxt = np.size(data_4, 1)
    
    if compute_region is None:
        compute_region = np.ones((np.size(popu), 0))
    
    if window_size is None:
        window_size = maxt*np.ones((np.size(data_4, 0), 1)); # By default, use all data to fit parameters
    
    if extra_imm is None:
        extra_imm = np.zeros((np.size(data_4, 0), maxt)) + S; # By default, use all data to fit parameters
    #addition of S
    
    F = passengerFlow
    beta_all_cell = [None] * np.size(popu)
    fittedC = [None] * np.size(popu)
    ci = [None] * np.size(popu)
    nn = np.size(popu)
    
    if np.isscalar(un_fact):
        un_fact = np.ones(len(popu)) * un_fact

    if np.isscalar(jp_l):
        jp_l = np.ones(len(popu)) * jp_l

    if np.isscalar(k_l):
        k_l = np.ones(len(popu)) * k_l

    if np.isscalar(alpha_l):
        alpha_l = np.ones(len(popu)) * alpha_l

    if np.isscalar(window_size):
        window_size = np.ones(nn) * window_size
    
    deldata = np.diff(data_4, 1)

  
    
    for j in range(1, np.size(popu) + 1):
        jp = int(jp_l[j - 1])
        k = int(k_l[j - 1])
        alpha = alpha_l[j - 1]
        jk = int(jp*k)
        
        beta_all_cell[j - 1] = np.zeros((k+1, 1))
        ci[j - 1] = np.zeros((k+1, 2))
        fittedC[j - 1] = np.zeros((1, 2))
        # update with start date
        if data_4.size==0 or data_4[j - 1].size == 0 or data_4[j - 1, -1] < 1 or compute_region[j - 1] < 1: # if there is no data or explictly told not to compute
            continue

        
        skip_days = int(maxt - window_size[j - 1])
        
        if season_start_day + 15 < maxt:
            skip_days = season_start_day
            
        if skip_days < 0:
            skip_days = 0
        
        ex = (np.arange(maxt - skip_days - jk - 1, 0, -1)).transpose()
        alphavec = np.power(alpha, ex)
        alphamat = np.tile(alphavec, np.concatenate(([k+1], [1]))).transpose()
      
        tdim = maxt - jk - skip_days - 1
        if tdim<0:
            tdim = 0
        y = np.zeros((tdim, 1))
        X = np.zeros((tdim, k+1))
        Ikt = np.zeros((1,k))
        
        
        for t in range(skip_days+jk+1, maxt):
            Ikt1 = deldata[j - 1, t - jk - 1:t-1]
            #print(j,t)
            S = (1-(extra_imm[j - 1, t - 1] + (un_fact[j - 1] * data_4[j - 1,t - 1])) / popu[j-1])
            for kk in range(1, k+1):
                m0 = Ikt1[(kk-1)*jp : kk*jp]
                Ikt[0, kk - 1] = S*sum(m0, 0)
            
            if F is None or (np.isscalar(F) or len(np.shape(F)) == 0) or np.size(F, 0) != np.size(popu):
                incoming_travel = np.zeros((1, 1))
            else:
                incoming_travel = np.transpose(F[:, j - 1] / popu) @ sum(Ikt1, 1)

            X[(t-1) - jk-skip_days, :] = np.concatenate((Ikt, incoming_travel), 1)
            y[(t-1) - jk-skip_days] = np.transpose(deldata[j - 1, t - 1])
        
        X = alphamat * X
        y = alphavec.transpose() * y.flatten()


        if np.size(X) != 0 and np.size(y) != 0:
            a = np.concatenate((np.ones(k), [np.inf]), 0)
            if ret_conf is None:   # If confidence intervals are not required, we will run this as this seems to be faster
                beta_vec = scipy.optimize.lsq_linear(X, y, bounds=(np.zeros(k+1), a)).x
                
            else:
                def sigmoid(X, *w):
                    w1 = np.array(w).flatten()
                    return X @ (1 / (1 + np.exp(-w1)))
             
                
                k1 = X.shape[1]
                w0 = np.zeros(k1)
                popt, pcov = scipy.optimize.curve_fit(sigmoid, X, y, p0=w0, method="dogbox")

                n = len(y)
                mse = np.sum((y - sigmoid(X, popt)) ** 2) / (n - k1)
                se = np.sqrt(np.diag(pcov) * mse)
                tval = scipy.stats.t.ppf((1 + ret_conf) / 2, n - k1)

                CI_lower = popt - tval * se
                CI_upper = popt + tval * se

                beta_vec = 1 / (1 + np.exp(-popt))
                beta_CI = 1 / (1 + np.exp(-np.vstack((CI_lower, CI_upper))))
                
                ci[j - 1] = beta_CI
                
            beta_all_cell[j - 1] = beta_vec
    return (beta_all_cell, fittedC, ci)


###############################################################################################

"""
Affects each time series (row) separately

beta_all_cell (python list of 1-d numpy arrays): List with each element representing infection parameters.
k_l: List with each element representing number of beta all cell params for the corresponding region
jp_l: List with each element representing a hyperparameter for binning
data_4: (2-D matrix of real numbers n x T) (should just be named data) cumulative reported cases matrix (each row is a region, each column is timestamp)
passengerFlowDarpa: set it to 0. Originally intended to be n x n matrix to capture inter-region mobility
popu (nx1): Population of each region
horizon: Number of time units to simulate starting from T_start
un_fact: under-reporting factor (>= 1)
base_infec: cumulative cases per region at the time of simulation start 
vac: vaccinations given (n x (T + horizon)) 
"""

def var_simulate_pred_un(data_4, passengerFlowDarpa, beta_all_cell, popu, k_l, horizon, jp_l, un_fact, base_infec=None, vac=None, rate_change=None):
    num_countries = size(data_4, 0)
    infec = zeros((num_countries, horizon))
    F = passengerFlowDarpa

    
    if rate_change is None:
        rate_change = ones((num_countries, horizon))

    if base_infec is None:
        base_infec = data_4[:, -1:]

    if vac is None:
        vac = zeros((num_countries, horizon))


    if size(rate_change) == 0:
        rate_change = ones((num_countries, horizon))

    if size(un_fact) == 1:
        un_fact = un_fact*ones((size(popu), 1))

    if size(jp_l) == 1:
        jp_l = ones((size(popu), 1))*jp_l

    if size(k_l) == 1:
        k_l = ones((size(popu), 1))*k_l

    for j in range(0, len(beta_all_cell)):
        this_beta = beta_all_cell[j]
        if size(this_beta)==k_l[j]:
            beta_all_cell[j] = np.append(this_beta, 0)

            #beta_all_cell[j] = concatenate((this_beta, [[0]]), axis=0)
            pass

    data_4_s = data_4
    temp = data_4_s
    deltemp = diff(temp, axis=1)
    
    if isinstance(F, (int, float)) or F is None:    # Optimized code when mobility not considered
        for j in range(0, size(popu)):
            lastinfec = base_infec[j]

            if sum(beta_all_cell[j]) == 0:
                infec[j, :] = lastinfec
               # print(infec.shape, "a")
                continue

            jp = int(jp_l[j])
            k = int(k_l[j])
            jk = int(jp*k)
            Ikt1 = deltemp[j, -(jk):]
            Ikt = zeros((k, 1))
            
            for t in range(0, horizon):
                true_infec = un_fact[j]*lastinfec
                S = 1 - true_infec/popu[j] - ((1-true_infec/popu[j])*vac[j, t]) / popu[j]
                for kk in range(1, k+1):
                    Ikt[kk - 1] = sum(Ikt1[(kk - 1)*jp : kk*jp], 0)
                Xt = concatenate(((S*Ikt), [[0]]), axis=0)
                yt = rate_change[j, t]*sum(transpose(beta_all_cell[j]) @ Xt, axis=0)
                yt = max(yt, 0)
                
                lastinfec = lastinfec + yt
                infec[j, t] = lastinfec
               # print(infec.shape, "b")
                Ikt1 = np.append(Ikt1, yt)  # Append yt to Ikt1
                Ikt1 = np.delete(Ikt1, 0)

    return infec