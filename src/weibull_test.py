# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:26:02 2025

@author: ellar
"""

from os.path import dirname, abspath, join
from os import getcwd
import sys
#run this fro src folder, otherwise it doesn't work
THIS_DIR = dirname(getcwd())
CODE_DIR = join(THIS_DIR, 'src')
RES_DIR =  join(THIS_DIR, 'res')
sys.path.append(CODE_DIR)
sys.path.append(RES_DIR) 
sys.path.append('D:')
import numpy as np
import pandas as pd

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import xarray as xr
import time

from scipy.stats import linregress

def wbl_tail_test_func(samples, oe_times, left_cens_prctile_thr, p_test, niter, censorams):
    
    years = oe_times.dt.year
    df = pd.DataFrame({"year": years, "P": samples})
    df = df.sort_values(by = "P")
    
    is_block_maximum = np.zeros_like(samples, dtype=bool) #make a bool list of the locations of annual maxima
    for i in np.unique(years):
        df_block = df.P[df.year == i]
        max_index = df_block.idxmax()
        is_block_maximum[max_index] = True
    
    thr = df.P.quantile(left_cens_prctile_thr)
    df_use = df[df.P >= thr]
    is_block_maximum_use = is_block_maximum[df.P >= thr]
    
    if censorams:
        df_use = df_use[~is_block_maximum_use]
    
    ECDF = np.arange(0,len(samples))/(len(samples)+1)
    X = np.log(np.log(1./(1-ECDF[df_use.index])))   # Weibull-tranformation for probabilities
    Y = np.log(samples[df_use.index])  # Weibull-tranformation for samples
    slope, intercept, r_value, p_value, std_err = linregress(X, Y) #linear regression
    scale = np.exp(intercept);         #Weibull scale parameter
    shape = 1/slope
    
    
    wblinv = scale * (- np.log(1-np.random.rand(niter,len(samples))))**(1 / shape)
    randy = np.sort(wblinv)    #/this is wrong      #weibull-distributed stochastic samples
    p_lo = nanmean( samples(istest)<quantile(randy(:,istest),p_test/2,1) );    % fraction of block maxima below the (1-p) CI
                   p_hi = nanmean( samples(istest)>quantile(randy(:,istest),1-p_test/2,1)' );  % fraction of block maxima above the (1-p) CI

    p_out = p_hi + p_lo; % fraction of block maxima out of the (1-p) CI
 
    
    
    return is_rejected, p_out, p_hi, p_lo, scale, shape


country = 'Japan'
country_save = 'Japan'
code_str = 'JP' 
n_stations = 10 #number of stations to sample
min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
max_yrs = 1000 #if no max, set to very high
name_col = 'ppt'
temp_name_col = "t2m"
chosen_station = "12261"

T = np.genfromtxt(f"D:/ordinary_events/{country_save}/T_{chosen_station}.csv")
P = np.genfromtxt(f"D:/ordinary_events/{country_save}/P_{chosen_station}.csv")
times = pd.read_csv(f"D:/ordinary_events/{country_save}/time_{chosen_station}.csv",parse_dates = ["oe_time"])
















