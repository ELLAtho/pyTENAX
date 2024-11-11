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
import xarray as xr
import time


country = 'Germany'
code_str = 'DE_' #string from beginning of file names
name_col = 'ppt'
temp_name_col = "t2m"
min_yrs = 0 #get temperature data for records > 0 years
T_nan_limit = 0.1 #limit for amount of nans that can be present in time series 0.3 = 30%. there will basically be 0 or 100% so it isn't really a big deal


#READ IN META INFO FOR COUNTRY
info = pd.read_csv('D:/metadata/'+country+'_fulldata.csv')
info.station = info['station'].apply(lambda x: f'{int(x):05}') #need to edit this according to file
  
info = info[info['cleaned_years']>=min_yrs] #filter out short data

#READ IN ERA5 DATA
T_files = sorted(glob.glob('D:/ERA5_land/'+country+'*/*'))
edit_files = []

#need to add in start and end dates here
start_time = time.time()

for i in np.arange(184,len(info)): #for each selected station
    
    T_temp = [0]*np.size(T_files)
    for n, file in enumerate(T_files):
        with xr.open_dataarray(file) as da:
            T_temp[n] = da.sel(latitude = info.latitude[info.index[i]],method = 'nearest').sel(longitude = info.longitude[info.index[i]],method = 'nearest')
    #TODO: add selection of start and end times    
    T_ERA = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files
    
    
    #Check T_ERA isnt full of nans
    #TODO: make this better... right now it doesn't cover full possibilities
    nan_perc = np.isnan(T_ERA).sum().item()/np.size(T_ERA)
    if nan_perc > T_nan_limit:
        edit_files.append(info.station[info.index[i]])
        print('percentage nans: '+str(100*nan_perc))
        print(f'({info.latitude[info.index[i]]},{info.longitude[info.index[i]]})')
        print(str(info.station[info.index[i]]))
        j=1
        while nan_perc > T_nan_limit:
            T_temp = [0]*np.size(T_files)
            new_lat = info.latitude[info.index[i]]+0.1*j
            new_lon = info.longitude[info.index[i]]+0.1*(j-1)
            for n, file in enumerate(T_files):
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
                
            T_ERA = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files
            nan_perc = np.isnan(T_ERA).sum().item()/np.size(T_ERA)
            print(f"Percentage of NaNs after offset lat {+0.1 * j}: {100 * nan_perc:.2f}%")
            if nan_perc < T_nan_limit:
                break
            
            new_lat = info.latitude[info.index[i]]+0.1*(j-1)
            new_lon = info.longitude[info.index[i]]+0.1*j
            for n, file in enumerate(T_files):
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
                
            T_ERA = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files
            nan_perc = np.isnan(T_ERA).sum().item()/np.size(T_ERA)
            print(f"Percentage of NaNs after offset lon {+0.1 * j}: {100 * nan_perc:.2f}%")
            if nan_perc < T_nan_limit:
                break
                       
            new_lat = info.latitude[info.index[i]]-0.1*j
            new_lon = info.longitude[info.index[i]]-0.1*(j-1)
            for n, file in enumerate(T_files):
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
                
            T_ERA = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files
            nan_perc = np.isnan(T_ERA).sum().item()/np.size(T_ERA)
            print(f"Percentage of NaNs after offset lat {-0.1 * j}: {100 * nan_perc:.2f}%")
            if nan_perc < T_nan_limit:
                break
            
            new_lat = info.latitude[info.index[i]]-0.1*(j-1)
            new_lon = info.longitude[info.index[i]]-0.1*j
            for n, file in enumerate(T_files):
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
                
            T_ERA = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files
            nan_perc = np.isnan(T_ERA).sum().item()/np.size(T_ERA)
            print(f"Percentage of NaNs after offset lon {-0.1 * j}: {100 * nan_perc:.2f}%")    
            new_lat = info.latitude[info.index[i]]-0.1*j
            new_lon = info.longitude[info.index[i]]-0.1*j
            for n, file in enumerate(T_files):
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
                
            T_ERA = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files
            nan_perc = np.isnan(T_ERA).sum().item()/np.size(T_ERA)
            print(f"Percentage of NaNs after offset double {-0.1 * j}: {100 * nan_perc:.2f}%")
            
            new_lat = info.latitude[info.index[i]]+0.1*j
            new_lon = info.longitude[info.index[i]]+0.1*j
            for n, file in enumerate(T_files):
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
                
            T_ERA = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files
            nan_perc = np.isnan(T_ERA).sum().item()/np.size(T_ERA)
            print(f"Percentage of NaNs after offset double {0.1 * j}: {100 * nan_perc:.2f}%")    
            
            j=j+1
            if j == 10:
                print('too far!')
                print(str(info.station[info.index[i]])+' is all nans')
                break
        print(str(info.latitude[info.index[i]])+' became '+str(new_lat))
        print(str(info.longitude[info.index[i]])+' became '+str(new_lon))
        print(T_ERA)
    
    
    
    T_ERA.to_netcdf('D:/'+country+'_temp/'+code_str+str(info.station[info.index[i]])+'.nc')
    if i % 100 == 0:
        print('{i}/{len(info)}')
        print('D:/'+country+'_temp/'+code_str+str(info.station[info.index[i]])+'.nc')


print('Data reading finished.')














