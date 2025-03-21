# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:54:07 2025

@author: ellar
"""


from os.path import dirname, join
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
from scipy.stats import gaussian_kde

import datetime as dt
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

import xarray as xr
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from matplotlib import cm



drive = 'D'



# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9

name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 




df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 



S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = 0,
        min_ev_dur = 60,
        niter_smev = 1000, 
        beta = 6
    )



start_time = [0]*len(df_parameters)
g_phat = [0]*len(df_parameters)


for i in np.arange(0,len(df_parameters)):
    start_time[i] = time.time() 
    #read in ppt data
    file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
    
    if 'code_str' in locals():
        G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
    else:
        G = pd.read_csv(f"{file_name}.csv")
        G['prec_time'] = pd.to_datetime(G['prec_time'])
        G.set_index('prec_time', inplace=True)
    
    
    data = G 
    data = S.remove_incomplete_years(data, name_col)
    
    df_arr = np.array(data[name_col])
    df_dates = np.array(data.index)
    
    #extract indexes of ordinary events
    #these are time-wise indexes =>returns list of np arrays with np.timeindex
    idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
        
    
    #get ordinary events by removing too short events
    #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
    arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
    
    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
    dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
    
    AMS = dict_AMS['60']
    
    
    
    T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" 
    
    if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
        print('skip')
        g_phat[i] = [np.nan,np.nan]
    else:
        T_ERA = xr.load_dataarray(T_path)
        t_data = (T_ERA.squeeze()-273.15).to_dataframe()
        df_arr = np.array(data[name_col])
        df_dates = np.array(data.index)
        
        #extract indexes of ordinary events
        #these are time-wise indexes =>returns list of np arrays with np.timeindex
        idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
            
        
        #get ordinary events by removing too short events
        #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
        arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
        
        #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
        dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
        
        AMS = dict_AMS['60']
        
        
        df_arr_t_data = np.array(t_data[temp_name_col])
        df_dates_t_data = np.array(t_data.index)
        
        dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
        
        
        
        # Your data (P, T arrays) and threshold thr=3.8
        P = dict_ordinary["60"]["ordinary"].to_numpy() 
        T = dict_ordinary["60"]["T"].to_numpy()  
        g_phat[i] = S.temperature_model(T)
    time_taken = (time.time()-start_time[i-9])/10
    time_left = (len(df_parameters)-i)*time_taken/60
    print(f"{i}/{len(df_parameters)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops

    
g_phat_df = pd.DataFrame({"station" : df_parameters.station,
                          "mu": np.array(g_phat)[:,0],
                          "sigma": np.array(g_phat)[:,1]})
g_phat_df.to_csv(f"{drive}:/outputs/{country_save}/g_phat6",index= False)


