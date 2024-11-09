# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:17:34 2024

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
import xarray as xr
import time
import pickle


country = 'Germany'
code_str = 'DE' 
name_col = 'ppt'
temp_name_col = "t2m"
min_yrs = 5 #get temperature data for records > 5 years


#READ IN META INFO FOR COUNTRY
info = pd.read_csv('D:/metadata/'+country+'_fulldata.csv')
info.station = info['station'].apply(lambda x: f'{int(x):05}') #need to edit this according to file
  
info = info[info['cleaned_years']>=min_yrs] #filter out short data

#READ IN ERA5 DATA
T_files = sorted(glob.glob('D:/ERA5_land/'+country+'*/*'))

#get lats and lons
lats = [info.latitude[i] for i in info.index]
lons = [info.longitude[i] for i in info.index]


#need to add in start and end dates here
start_time = time.time()
T_ERA = [0]*len(info)

for i in np.arange(0,len(info)): #for each selected station
    
    T_temp = [0]*np.size(T_files)
    for n in np.arange(0,np.size(T_files)): #select temperature data at gridpoint closest to station
        T_temp[n] = xr.open_dataarray(T_files[n]).sel(latitude = lats[i],method = 'nearest').sel(longitude = lons[i],method = 'nearest')
        
    T_ERA[i] = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files

print('time to read era5 '+str(time.time()-start_time))
print('Data reading finished.')


for i in np.arange(0,len(T_ERA)):
    T_ERA[i].to_netcdf('D:/'+country+'_temp/'+str(info.station[i]))













