# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:21:11 2024

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
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
import xarray as xr
import time
import matplotlib.pyplot as plt

drive = 'D'

country = 'Japan'
#country = 'Germany'
code_str = 'JP_' 
#code_str = 'DE_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
#minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_col = 'ppt'
temp_name_col = "temperatures"
min_yrs = 19 #BUG. need to filter ...
n_stations = 4


Had_stations = pd.read_csv("D:/HadISD/HAD_metadata.txt",names = ['station','latitude','longitude','elevation'],sep=r"\s+")
Had_spec = Had_stations[(Had_stations.latitude>=minlat) & (Had_stations.latitude<=maxlat)]
Had_spec = Had_spec[(Had_spec.longitude>=minlon) & (Had_spec.longitude<=maxlon)]
Had_spec.latitude = round(Had_spec.latitude, 2) #round to 2 dp so lons and lats match
Had_spec.longitude = round(Had_spec.longitude, 2)


info = pd.read_csv('D:/metadata/'+country+'_fulldata.csv')
info.station = info['station'].apply(lambda x: f'{int(x):05}') #need to edit this according to file
info = info.rename(columns={'station': 'station_info'})
info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

info.latitude = round(info.latitude, 2)
info.longitude = round(info.longitude, 2)



#get stations with identical lat and lon to Had
matched_stations_full = info.merge(Had_spec, on=['latitude', 'longitude'])
matched_stations = matched_stations_full[matched_stations_full['cleaned_years']>=min_yrs].sort_values(by=['cleaned_years'],ascending=0)
matched_stations = matched_stations[0:n_stations]

plt.scatter(info.longitude,info.latitude,label = 'GSDR')
plt.scatter(Had_spec.longitude,Had_spec.latitude,label = 'HAD')
plt.scatter(matched_stations.longitude,matched_stations.latitude,label ='matches')
plt.legend()
plt.show()



# GET LIST OF FILES OF MATCHED HAD STATIONS
files = [glob.glob("D:/HadISD/unzipped/*/*"+str(file)+"*.nc") for file in matched_stations.station]
G_files = [glob.glob('D:/'+country+'/*'+str(file)+'*') for file in matched_stations.station_info]


###############################################################################




g_phats = []
F_phats = []
scaling_rate_Ws, scaling_rate_qs  = [],[]

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 1,
        min_ev_dur = 60,
        niter_smev = 1000,
    )

for n in np.arange(0,len(G_files)):
    HAD = xr.open_dataset(files[n][0]).temperatures
    HAD[HAD < -1000] = np.nan
    
    
    
    G = pd.read_csv(G_files[n][0], skiprows=21, names=[name_col])
    data_meta = readIntense(G_files[n][0], only_metadata=True, opened=False)
    
    
       
    #extract start and end dates from metadata
    start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
    end_date_G= dt.datetime.strptime(data_meta.end_datetime, "%Y%m%d%H")
    
    time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
    # replace -999 with nan
    G[G == -999] = np.nan
    
    G['prec_time'] = time_list_G
    G = G.set_index('prec_time')
    
    
    start_time = time.time()
    data = S.remove_incomplete_years(G, name_col)
    
    #get data from pandas to numpy array
    df_arr = np.array(data[name_col])
    df_dates = np.array(data.index)
    
    idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
        
    
    #get ordinary events by removing too short events
    #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
    arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
    
    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
    dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
    
    elapsed_time = time.time() - start_time
    # Print the elapsed time
    print(f"Elapsed time get OE: {elapsed_time:.4f} seconds")
    
    
    t_data = HAD.to_dataframe()
    
    
    start_time = time.time()
    df_arr_t_data = np.array(t_data[temp_name_col])
    df_dates_t_data = np.array( t_data.index)
    
    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
    
    elapsed_time = time.time() - start_time
    # Print the elapsed time
    print(f"Elapsed time : {elapsed_time:.4f} seconds")
    
    
    start_time = time.time()
    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() # Replace with your actual data
    T = dict_ordinary["60"]["T"].to_numpy()  # Replace with your actual data
    
    
    
    # Number of threshold 
    thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
    
    # Sampling intervals for the Montecarlo
    Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
    
    #TENAX MODEL HERE
    #magnitude model
    F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
    #temperature model
    g_phat = S.temperature_model(T)
    # M is mean n of ordinary events
    n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    #estimates return levels using MC samples
    
    RL, _, _ = S.model_inversion(F_phat, g_phat, n, Ts)
    S.n_monte_carlo = np.size(P)*S.niter_smev
    _, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts, gen_P_mc = True,gen_RL=False) 
    S.n_monte_carlo = int(2e4)
    print(RL)
    

    g_phats.append(g_phat)
    F_phats.append(F_phat)
    
    #PLOTTING THE GRAPHS
    
    eT = np.arange(np.min(T)-4,np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    # fig 2a
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [eT[0],eT[-1]])
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(f'fig 2a. {data_meta.latitude,data_meta.longitude}')
    plt.show()
    
    #fig 2b
    TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
    plt.title('fig 2b')
    plt.show()
    
    #fig 4 
    AMS = dict_AMS['60'] # yet the annual maxima
    TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+3])
    plt.title('fig 4')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    
    #fig 5 
    iTs = np.arange(np.min(T)-4,np.max(T)+10,1.5) #idk why we need a different T range here 
    
    scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phat,S.niter_smev,eT,iTs,xlimits = [eT[0],eT[-1]])
    plt.title('fig 5')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    scaling_rate_Ws.append(scaling_rate_W)
    scaling_rate_qs.append(scaling_rate_q)
    
    
###############################################################################
#THE SAME USING ERA5


#TODO: put in the distances too

T_files = sorted(glob.glob(drive+':/ERA5_land/'+country+'*/*')) #make list of era5 files
saved_files = glob.glob(drive+':/'+country+'_temp/*') #files already saved
T_series = []

for i in np.arange(0,len(G_files)): #for each selected station
    target_lat = matched_stations.latitude[matched_stations.index[i]] #define targets
    target_lon = matched_stations.longitude[matched_stations.index[i]]
    start_date = matched_stations.startdate[matched_stations.index[i]]-pd.Timedelta(days=1) #adding an extra day either end to be safe
    end_date = matched_stations.enddate[matched_stations.index[i]]+pd.Timedelta(days=1)
    
    save_path = drive + ':/'+country+'_temp\\'+code_str + str(matched_stations.station_info[matched_stations.index[i]]) + '.nc'
    # Check if file already exists before saving
    
    if save_path not in saved_files:
        T_temp = []
        for file in T_files:
            with xr.open_dataarray(file) as da:
                T_temp.append(da.sel(latitude = target_lat,method = 'nearest').sel(longitude = target_lon,method = 'nearest'))
        
    
        T_ERA = xr.concat(T_temp,dim = 'valid_time').sel(valid_time = slice(start_date,end_date))
        T_series.append(T_ERA)
        
        save_path = drive + ':/'+country+'_temp/'+code_str + str(matched_stations.station_info[matched_stations.index[i]]) + '.nc'
        # Check if file already exists before saving
        
        T_ERA.to_netcdf(save_path) #save file with same name format as GSDR
        saved_files.append(save_path)  # Update saved_files to include the newly saved file
    else:
        print(f"File {save_path} already exists. Skipping save.")
        T_ERA = xr.load_dataarray(save_path)
        T_series.append(T_ERA)

T_temp = []


    
###############################################################################
#TENAX again


temp_name_col = "t2m"

g_phats_ERA = []
F_phats_ERA = []
scaling_rate_Ws_ERA, scaling_rate_qs_ERA  = [],[]

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
        niter_smev = 1000,
    )

for n in np.arange(0,len(G_files)):
    ERA = T_series[n]-273.15
    
    
    G = pd.read_csv(G_files[n][0], skiprows=21, names=[name_col])
    data_meta = readIntense(G_files[n][0], only_metadata=True, opened=False)
    
       
    #extract start and end dates from metadata
    start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
    end_date_G= dt.datetime.strptime(data_meta.end_datetime, "%Y%m%d%H")
    
    time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
    # replace -999 with nan
    G[G == -999] = np.nan
    
    G['prec_time'] = time_list_G
    G = G.set_index('prec_time')
    
    
    start_time = time.time()
    data = S.remove_incomplete_years(G, name_col)
    
    #get data from pandas to numpy array
    df_arr = np.array(data[name_col])
    df_dates = np.array(data.index)
    
    idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
        
    
    #get ordinary events by removing too short events
    #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
    arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
    
    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
    dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
    
    elapsed_time = time.time() - start_time
    # Print the elapsed time
    print(f"Elapsed time get OE: {elapsed_time:.4f} seconds")
    
    
    t_data = ERA.to_dataframe()
    
    
    start_time = time.time()
    df_arr_t_data = np.array(t_data[temp_name_col])
    df_dates_t_data = np.array( t_data.index)
    
    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
    
    elapsed_time = time.time() - start_time
    # Print the elapsed time
    print(f"Elapsed time : {elapsed_time:.4f} seconds")
    
    
    start_time = time.time()
    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() # Replace with your actual data
    T = dict_ordinary["60"]["T"].to_numpy()  # Replace with your actual data
    
    
    
    # Number of threshold 
    thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
    
    # Sampling intervals for the Montecarlo
    Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
    
    #TENAX MODEL HERE
    #magnitude model
    F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
    #temperature model
    g_phat = S.temperature_model(T)
    # M is mean n of ordinary events
    n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    #estimates return levels using MC samples
    
    RL, _, _ = S.model_inversion(F_phat, g_phat, n, Ts)
    S.n_monte_carlo = np.size(P)*S.niter_smev
    _, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts, gen_P_mc = True,gen_RL=False) 
    S.n_monte_carlo = int(2e4)
    print(RL)
    

    g_phats_ERA.append(g_phat)
    F_phats_ERA.append(F_phat)
    
    #PLOTTING THE GRAPHS
    
    eT = np.arange(np.min(T)-4,np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    # fig 2a
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [eT[0],eT[-1]])
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(f'fig 2a. {data_meta.latitude,data_meta.longitude}')
    plt.show()
    
    #fig 2b
    TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
    plt.title('fig 2b')
    plt.show()
    
    #fig 4 
    AMS = dict_AMS['60'] # yet the annual maxima
    TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+3])
    plt.title('fig 4')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    
    #fig 5 
    iTs = np.arange(np.min(T)-4,np.max(T)+10,1.5) #idk why we need a different T range here 
    
    scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phat,S.niter_smev,eT,iTs,xlimits = [eT[0],eT[-1]])
    plt.title('fig 5')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    scaling_rate_Ws_ERA.append(scaling_rate_W)
    scaling_rate_qs_ERA.append(scaling_rate_q)
    
    
    
    
    
    
