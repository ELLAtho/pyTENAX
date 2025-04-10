# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 21:37:50 2025

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
from scipy.signal import find_peaks

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
from matplotlib import colormaps
from matplotlib.colors import to_rgba



drive = 'D'
alpha_set = 0



country = 'Germany' 
ERA_country = 'Germany'
country_save = 'Germany'
code_str = 'DE_'
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 50
region_lats = [minlat,51,maxlat]
region_lons = [minlon,9,maxlon]


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 30
# region_lats = [minlat,31,35.8,41.3,maxlat]
# region_lons = [minlon,maxlon]



# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 30
# region_lats = [minlat,37.5,maxlat]
# region_lons = [minlon,-116,-105,-90,maxlon]

# country = 'UK' 
# ERA_country = 'UK'
# country_save = 'UK'
# code_str = 'UK_'
# name_len = 0
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 50



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

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
        beta = 4
    )

start_time = [0] * len(df_parameters)

for i in np.arange(0, len(df_parameters)):
    start_time[i] = time.time()
    file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
    
    oe_save = f"{drive}:/ordinary_events/{country_save}\\time_{df_parameters.station.iloc[i]}.csv"
    if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            
        if 'code_str' in locals():
            G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
        else:
            G = pd.read_csv(f"{file_name}.csv")
            G['prec_time'] = pd.to_datetime(G['prec_time'])
            G.set_index('prec_time', inplace=True)
            
        T_savename = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if T_savename in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            data = S.remove_incomplete_years(G, name_col)
        
            df_arr = np.array(data[name_col])
            df_dates = np.array(data.index)
            
            #extract indexes of ordinary events
            #these are time-wise indexes =>returns list of np arrays with np.timeindex
            idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
            _,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
            
            #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
            dict_ordinary, _ = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
            
            oe_times = dict_ordinary["60"]["oe_time"]
            oe_times.to_csv(oe_save,index = False)


        if i%50 == 0:    
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(df_parameters)-i)*time_taken/60
            print(f"{i}/{len(df_parameters)}. Approx time left: {time_left:.0f} mins. T length: {len(T)}, time length: {len(oe_times)}") #this is only correct after 50 loops
        else:
            pass




oe_check = pd.read_csv(oe_save,parse_dates = ["oe_time"])









