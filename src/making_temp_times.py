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
from geopy.distance import geodesic


drive = 'D'  #specify name of external drive

country = 'Japan'
code_str = 'JP_' #string from beginning of file names
name_col = 'ppt'
temp_name_col = "t2m"
min_yrs = 0 #get temperature data for records > 0 years
T_nan_limit = 0.1 #limit for amount of nans that can be present in time series 0.3 = 30%. there will basically be 0 or 100% so it isn't really a big deal


#READ IN META INFO FOR COUNTRY
info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv')
info.station = info['station'].apply(lambda x: f'{int(x):05}') #need to edit this according to file
info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

info = info[info['cleaned_years']>=min_yrs] #filter out short data

#READ IN ERA5 DATA
T_files = sorted(glob.glob(drive+':/ERA5_land/'+country+'*/*')) #make list of era5 files
saved_files = glob.glob(drive+':/'+country+'_temp/*') #files already saved
nan_files = []
dist_to_point = []


nans = xr.open_dataarray(T_files[0])[0]
nans = np.invert(np.isnan(nans)).astype(int) #xarray with 0s where there are nans (the ocean)

#make a function to calculate the distance from location of station to closest era5 gridpoint
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

start_time = time.time()
starti = 405
for i in np.arange(starti,len(info)): #for each selected station
    save_path = drive+':/'+country+'_temp\\'+code_str+str(info.station[info.index[i]])+'.nc'
    if save_path not in saved_files:
        
        target_lat = info.latitude[info.index[i]] #define targets
        target_lon = info.longitude[info.index[i]]
        start_date = info.startdate[info.index[i]]-pd.Timedelta(days=1) #adding an extra day either end to be safe
        end_date = info.enddate[info.index[i]]+pd.Timedelta(days=1)
        start_year = str(start_date.year)
        end_year = str(end_date.year)
        first_index = next((i for i, s in enumerate(T_files) if start_year in s), None) #location of first file that has start year so can read fewer files in
        last_index = next((len(T_files) - 1 - i for i, s in enumerate(reversed(T_files)) if end_year in s), None) #same for end year
          
        
        check = xr.load_dataarray(T_files[0]).sel(latitude = target_lat,method = 'nearest').sel(longitude = target_lon,method = 'nearest') #check if series is nans
        nan_perc = np.isnan(check).sum().item()/np.size(check) #get percentage of file that is nans (should basically be 0 or 100)
        
        T_temp = [0]*np.size(T_files[first_index:last_index])
        
        #if there are nans
        if nan_perc > T_nan_limit:
            print(i)
            print(code_str+str(info.station[info.index[i]]))
            nan_files.append(info.station[info.index[i]]) #making a list of the files that had nans
            
            nans_sm = nans.sel(latitude = slice(target_lat+0.5,target_lat-0.5),
                               longitude = slice(target_lon-0.5,target_lon+0.5)) #select smaller map for speed
            
            distances = nans_sm*xr.apply_ufunc(
                calculate_distance,
                xr.full_like(nans_sm.latitude, target_lat),  # Broadcast the target latitude
                xr.full_like(nans_sm.longitude, target_lon),  # Broadcast the target longitude
                nans_sm.latitude,
                nans_sm.longitude,
                vectorize=True,  
                output_dtypes=[float],  
            ) #calculate the distance from target to closest non nan
            
            distances = distances.where(distances !=0, np.nan) #change 0s from mask into nans
            location = distances.where(distances==distances.min(),drop=True) #get location with smallest distance from target
            new_lat = location.latitude.to_numpy() 
            new_lon = location.longitude.to_numpy()
            dist_to_point.append(location) #making list of distances from point
            
            for n, file in enumerate(T_files[first_index:last_index]): #load in the data
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
            
            T_ERA = xr.concat(T_temp,dim = 'valid_time').sel(valid_time = slice(start_date,end_date))
            print(T_ERA)
            print(f'Latitudes: {target_lat} became {new_lat}. Longitudes: {target_lon} became {new_lon}. {location.to_numpy()[0][0]} metres away.')
        
        
        else: #if location has no nans dont do the whole distances thing
            
            for n, file in enumerate(T_files[first_index:last_index]):
                with xr.open_dataarray(file) as da:
                    T_temp[n] = da.sel(latitude = target_lat,method = 'nearest').sel(longitude = target_lon,method = 'nearest')
            
       
            T_ERA = xr.concat(T_temp,dim = 'valid_time').sel(valid_time = slice(start_date,end_date))  #combine multiple time files
            
        
    
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














