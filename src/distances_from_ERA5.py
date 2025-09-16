# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:16:49 2025

@author: ellar
"""

import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import time
import glob


from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *


countries = ["germany","Japan","UK","US"]
country_saves = ["germany","Japan","UK","US_main"]
code_strs = ["DE_","JP_","UK_","US_"]
min_startdates = [dt.datetime(1900,1,1),dt.datetime(1900,1,1),dt.datetime(1950,1,1),dt.datetime(1950,1,1)] #this is for if havent read all ERA5 data yet
min_yrs = 10

lons_lats = [[47, 3, 55, 15],[24, 122.9, 45.6, 145.8],[49, -9.0, 62, 3] ,[24, -125, 56, -66]]



info = [0]*4

for country_i in range(4):
    country_save = country_saves[country_i]
    country = countries[country_i]
    code_str = code_strs[country_i]
    
    minlat, minlon, maxlat, maxlon = lons_lats[country_i]
    min_startdate = min_startdates[country_i]
    
    
    info1 = pd.read_csv('D:/metadata/'+country+'_fulldata.csv', dtype={'station': str})
    
    info1.startdate = pd.to_datetime(info1.startdate)
    info1.enddate = pd.to_datetime(info1.enddate)
    val_info = info1[info1['cleaned_years']>=min_yrs] #filter out stations that are less than min
    val_info = val_info[val_info['startdate']>=min_startdate]
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
    
    info[country_i] = val_info.reset_index()
    

distance_dfs = [0]*len(countries)
for i in range(len(countries)):
    distances = [0]*len(info[i])
    start_time = [0]*len(info[i])
    
    non_count = 0
    
    for j in range(len(info[i])):
        
        start_time[j] = time.time()
        
        station = info[i].station.iloc[j]
        
        
        temp_savename = f"D:/{countries[i]}_temp\\{code_strs[i]}{station}.nc"
        if temp_savename not in glob.glob(f"D:/{countries[i]}_temp/*"):
            distances[j] = np.nan
            non_count = non_count+1
        
        else:
            
            full_temp_xr = xr.load_dataarray(f"D:/{countries[i]}_temp/{code_strs[i]}{station}.nc")
            
            lat_ERA = full_temp_xr.latitude.to_numpy()
            if np.shape(lat_ERA) != ():
                lat_ERA = lat_ERA[0]
                
            
            lon_ERA = full_temp_xr.longitude.to_numpy()
            if np.shape(lon_ERA) != ():
                lon_ERA = lon_ERA[0]
            
            
            
            lat_GSDR = info[i].latitude.iloc[j]
            lon_GSDR = info[i].longitude.iloc[j]
            
            distances[j] = calculate_distance(lat_ERA, lon_ERA, lat_GSDR, lon_GSDR)
    
        
        if j%50 == 0:
            time_taken = (time.time()-start_time[j-9])/10
            time_left = (len(info[i])-j)*time_taken/60
            print(f"{j}/{len(info[i])}. Current average time to complete one {time_taken:.0f}s. Approx time left to complete {countries[i]}: {time_left:.0f} mins") #this is only correct after 50 loops
    print(f"not there: {non_count} stations")
    save_name = f"D:/{countries[i]}_temp/distance_info.csv"
    df = pd.DataFrame({
        "station": info[i].station,
        "distance" : distances
        })
    df.to_csv(save_name)
    distance_dfs[i] = df


# %% print percentages of total within grids


full_df = pd.concat(distance_dfs)
stations_45 = len(full_df[full_df.distance<=(np.sqrt(2*4500**2))])
stations_9 = len(full_df[full_df.distance<9000])
stations_18 =len(full_df[(full_df.distance<18000)]) 


print(f"{stations_45/len(full_df)} of stations within same grid cell")
print(f"{stations_9/len(full_df)} of stations within 9km")
print(f"{stations_18/len(full_df)} of stations within 18km")

print(f"max distance {np.max(full_df.distance)}")






