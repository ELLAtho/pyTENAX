# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:31:05 2025

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

country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9

name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 

#READ IN META INFO FOR COUNTRY
info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})


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


S = TENAX(
        return_period = [2,5,10,20,50,100, 200],  #for some reason it doesnt like calculating RP =<1
        durations = [60, 180],
        left_censoring = [0, 0.90],
        alpha = 0.05,
    )

max_ppt = [0]*np.size(files_sel)

for i in np.arange(0,np.size(files_sel)):
    data,data_meta = read_GSDR_file(files_sel[i],name_col)    
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
    
    AMS_sizes = dict_AMS['60']
    years = dict_ordinary['60'].year.unique()
    AMS_time = []
    for yr in years:
        dict_ordinary_year = dict_ordinary['60'][dict_ordinary['60'].year == yr]
        max_size = AMS_sizes[AMS_sizes.year == yr].AMS
        timing = dict_ordinary_year[dict_ordinary_year.ordinary == max_size.to_numpy()[0]].oe_time
        print(timing)
        AMS_time.append(timing)
        print(AMS_time)
        
        
    
    
    











