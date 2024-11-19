# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:14:46 2024

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
#country = 'Japan'
country = 'Germany'
#code_str = 'JP_' 
code_str = 'DE_'
#minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_col = 'ppt'
temp_name_col = "t2m"
min_yrs = 10 

#READ IN META INFO FOR COUNTRY
info = pd.read_csv('D:/metadata/'+country+'_fulldata.csv')
info.station = info['station'].apply(lambda x: f'{int(x):05}') #need to edit this according to file
info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

#select stations
val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min

files = glob.glob('D:/'+country+'/*') #list of files in country folder
files_sel = [files[i] for i in val_info.index]


S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
    )


#make empty lists to read into


T_files = sorted(glob.glob(drive+':/ERA5_land/'+country+'*/*')) #make list of era5 files
saved_files = glob.glob(drive+':/'+country+'_temp/*') #files already saved

g_phats = [0]*len(files_sel)
F_phats = [0]*len(files_sel)
thr = [0]*len(files_sel)
ns = [0]*len(files_sel)

saved_counter = 0

start_time = time.time()

for i in np.arange(0, len(files_sel)):
    
    #read in ppt data
    G = pd.read_csv(files_sel[i], skiprows=21, names=[name_col])
    data_meta = readIntense(files_sel[i], only_metadata=True, opened=False)

       
    #extract start and end dates from metadata
    start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
    end_date_G= dt.datetime.strptime(data_meta.end_datetime, "%Y%m%d%H")
    
    time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
    # replace -999 with nan
    G[G == -999] = np.nan
    
    G['prec_time'] = time_list_G
    G = G.set_index('prec_time')
    print(G[0:3])


    ######################################################################
    #read in T data
    target_lat = val_info.latitude[val_info.index[i]] #define targets
    target_lon = val_info.longitude[val_info.index[i]]
    start_date = val_info.startdate[val_info.index[i]]-pd.Timedelta(days=1) #adding an extra day either end to be safe
    end_date = val_info.enddate[val_info.index[i]]+pd.Timedelta(days=1)
    
    save_path = drive + ':/'+country+'_temp\\'+code_str + str(val_info.station[val_info.index[i]]) + '.nc'
    # Check if file already exists before saving
    
    if save_path not in saved_files:
        T_temp = []
        for file in T_files:
            with xr.open_dataarray(file) as da:
                T_temp.append(da.sel(latitude = target_lat,method = 'nearest').sel(longitude = target_lon,method = 'nearest'))
        
    
        T_ERA = xr.concat(T_temp,dim = 'valid_time').sel(valid_time = slice(start_date,end_date))
        T_series.append(T_ERA)
        
        T_ERA.to_netcdf(save_path) #save file with same name format as GSDR
        saved_files.append(save_path)  # Update saved_files to include the newly saved file
        saved_counter = saved_counter+1
    else:
        print(f"File {save_path} already exists. Skipping save.")
        T_ERA = xr.load_dataarray(save_path)
        
        #####################################################################
    #TENAX     
    data = G 
    data = S.remove_incomplete_years(data, name_col)
    t_data = (T_ERA.squeeze()-273.15).to_dataframe()
    print(f'{data_meta.latitude},{data_meta.longitude}')
    print(t_data[0:5])
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
    
    
    
    df_arr_t_data = np.array(t_data[temp_name_col])
    df_dates_t_data = np.array(t_data.index)
    
    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
    
    
    
    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() 
    T = dict_ordinary["60"]["T"].to_numpy()  
    
    
    # Number of threshold 
    thr[i] = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
    
    
    #TENAX MODEL HERE
    #magnitude model
    F_phats[i], loglik, _, _ = S.magnitude_model(P, T, thr[i])
    #temperature model
    g_phats[i] = S.temperature_model(T)
    # M is mean n of ordinary events
    ns[i] = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    
    
    
    time_taken = (time.time()-start_time)/(i)
    time_left = (len(files_sel)-i)*time_taken/60
    print(f"{i}/{len(files_sel)}. Approx time left: {time_left:.4f} mins")
    

T_temp = []




