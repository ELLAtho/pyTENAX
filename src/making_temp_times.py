# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:17:34 2024

@author: ellar
Code runs through the ppt files in a country folder and gets the temperature time series at that point from ERA5_land.
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
from pyTENAX.globalTENAX import *

import xarray as xr
import time
from geopy.distance import geodesic


drive = 'D'  #specify name of external drive

country = 'Belgium'
ERA_country = 'Germany'
code_str = 'BE_' #string from beginning of file names
name_len = 8 #how long the numbers are at the end of the files
name_col = 'ppt'
temp_name_col = "t2m"
min_yrs = 0 #get temperature data for records > 0 years
T_nan_limit = 0.1 #limit for amount of nans that can be present in time series 0.3 = 30%. there will basically be 0 or 100% so it isn't really a big deal


#READ IN META INFO FOR COUNTRY
info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv')
info.station = info['station'].apply(lambda x: f'{int(x):0{name_len}}') #need to edit this according to file
info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

info = info[info['cleaned_years']>=min_yrs] #filter out short data

#READ IN ERA5 DATA
T_files = sorted(glob.glob(drive+':/ERA5_land/'+ERA_country+'*/*')) #make list of era5 files
saved_files = glob.glob(drive+':/'+country+'_temp/*') #files already saved
nan_files = []
dist_to_point = []


nans = xr.open_dataarray(T_files[0])[0]
nans = np.invert(np.isnan(nans)).astype(int) #xarray with 0s where there are nans (the ocean)


start_time = time.time()
starti = 0
for i in np.arange(starti,len(info)): #for each selected station
    save_path = drive+':/'+country+'_temp\\'+code_str+str(info.station[info.index[i]])+'.nc'
    if save_path not in saved_files:
        
        target_lat = info.latitude[info.index[i]] #define targets
        target_lon = info.longitude[info.index[i]]
        start_date = info.startdate[info.index[i]]-pd.Timedelta(days=1) #adding an extra day either end to be safe
        end_date = info.enddate[info.index[i]]+pd.Timedelta(days=1) 
        
        T_ERA = make_T_timeseries(target_lat,target_lon,start_date,end_date,T_files,nans,T_nan_limit)
        
    
        T_ERA.to_netcdf(drive+':/'+country+'_temp/'+code_str+str(info.station[info.index[i]])+'.nc') #save file with same name format as GSDR
    
    else:
        print(f"File {save_path} already exists. Skipping save.")

    
    if i % 50 == 0: #print update every 50 files
        print(f'{i}/{len(info)}')
        print(drive+':/'+country+'_temp/'+code_str+str(info.station[info.index[i]])+'.nc')
        time_taken = (time.time()-start_time)/(i-starti)
        time_left = (len(info)-i)*time_taken/60
        print(f"Approx time left: {time_left:.4f} mins")
    else:
        pass

print('Data reading finished.')














