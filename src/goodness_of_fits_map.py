# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:12:29 2025

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



drive = 'D'
alpha_set = 0.05


country = 'Germany' 
ERA_country = 'Germany'
country_save = 'Germany'
code_str = 'DE_'
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9

name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 

#READ IN META INFO FOR COUNTRY
info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})


# shouldn't need this anymore as changed files
# if name_len!=0:
#     info.station = info['station'].apply(lambda x: f'{int(x):0{name_len}}') #need to edit this according to file
# else:
#     pass

info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

#select stations


val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min



if 'min_startdate' in locals():    
    val_info = val_info[val_info['startdate']>=min_startdate]
else:
    pass

if 'minlat' in locals():
    
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
    
else:
    pass



files = glob.glob(drive+':/'+country+'/*') #list of files in country folder
files_sel = [files[i] for i in val_info.index]


## READ IN FILES
save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 
TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters

df_parameters_0 = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/parameters.csv", dtype={'station': str})

# for some reason in germany there is one less row...
    



if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg, dtype={'station': str})

    #dataframe with all values
    new_df = df_parameters[['station','latitude','longitude','b']].copy()
    mask = new_df['b'] == 0
    
    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
    new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
    new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()

else:
    new_df = df_parameters.copy()

missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    df_parameters = df_parameters.drop(missing_rows.index)
    new_df = new_df.drop(missing_rows.index)
else:
    pass




S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = alpha_set,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )

RL = [0] * len(new_df)
start_time = [0] * len(new_df)
FRMSE = [0] * len(new_df)


for i in np.arange(0, len(new_df)):
    start_time[i] = time.time() 
    #read in ppt data
    file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
    
    if 'code_str' in locals():
        G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
    else:
        G = pd.read_csv(f"{file_name}.csv")
        G['prec_time'] = pd.to_datetime(G['prec_time'])
        G.set_index('prec_time', inplace=True)
        
    print(G[0:3])
    ######################################################################
    #TENAX  AMS

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
    
    # Define the model parameters by reading in those already saved
    g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
    #free
    F_phat = [new_df.kappa.iloc[i],new_df.b.iloc[i],
              new_df["lambda"].iloc[i],new_df.a.iloc[i]]
    #5% sig
    F_phat_5 = [df_parameters.kappa.iloc[i],df_parameters.b.iloc[i],
                df_parameters["lambda"].iloc[i],df_parameters.a.iloc[i]]
    #b always 0
    F_phat_0 = [df_parameters_0.kappa.iloc[i],df_parameters_0.b.iloc[i],
                df_parameters_0["lambda"].iloc[i],new_df.a.iloc[i]]
    
    n = df_parameters.n_events_per_yr.iloc[i]
    
    # Getting predicted return levels
    AMS_sort = AMS.sort_values(by=['AMS'])['AMS']
    plot_pos = np.arange(1,np.size(AMS_sort)+1)/(1+np.size(AMS_sort))
    
    eRP = 1/(1-plot_pos)
    S.return_period = eRP
    
    T_min = g_phat[0] - 2.5 * g_phat[1]
    T_max = g_phat[0] + 2.5 * g_phat[1]
    Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
    
    RL[i], __, __ = S.model_inversion(F_phat, g_phat, n, Ts)
    
    diffs = RL[i] - AMS_sort
    
    FRMSE[i] = np.sqrt(np.sum(diffs**2)/len(diffs))/(np.sum(AMS_sort)/len(diffs))
    #TODO: need to do the 5% and 0 ones still
    
    
    
    time_taken = (time.time()-start_time[i-9])/10
    time_left = (len(new_df)-i)*time_taken/60
    print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops




#TODO: need to save the return levels and the FRMSE and plot
















