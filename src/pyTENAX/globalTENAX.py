# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:21:58 2024

@author: ellar
"""

import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pyTENAX.intense import *



def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def make_T_timeseries(target_lat,target_lon,start_date,end_date,T_files,nans,T_nan_limit=0.1):
    #TODO: this is a very slow function
    """
    Creates a time series from teh ERA5 data at the specified lon, lat, dates.

    Parameters
    ----------
    target_lat : float
        Latitude of the location to create a timeseries.
    target_lon : float
        Longitude of the location to create a timeseries.
    start_date : datetime 
        Start date of the time series.
    end_date : datetime
        End date of the time series.
    T_files : list
        List of the names of temperatures netcdf files to be concatenated.
    nans : xarray.dataarray
        A 2D plot of the area (e.g from T_files[0]), with nan locations == 0 and land locations == 1.
    T_nan_limit : float, optional
        maximum fraction of nans allowed (it will usually be either 100% nans or 0% nans so this is a bit unnecessary).

    Returns
    -------
    T_ERA : xr.dataarray or []
        time series of ERA5_land temperature data at closest gridpoint to station. if T_ERA == [], there are no land points within 1 degree and the read has been skipped.

    """
    
    start_year = str(start_date.year)
    end_year = str(end_date.year)
    
    first_index = next((i for i, s in enumerate(T_files) if start_year in s), None) #location of first file that has start year so can read fewer files in
    last_index = next((len(T_files) - 1 - i for i, s in enumerate(reversed(T_files)) if end_year in s), None) #same for end year
    
    
    check = xr.open_dataarray(T_files[first_index]).sel(latitude = target_lat,method = 'nearest').sel(longitude = target_lon,method = 'nearest').sel(valid_time = slice(dt.datetime(start_date.year,1,1),dt.datetime(start_date.year,1,1)+pd.Timedelta(days=1))) #check if series is nans
    nan_perc = np.isnan(check).sum().item()/np.size(check) #get percentage of file that is nans (should basically be 0 or 100)
    
    T_ERA = [0]*np.size(T_files[first_index:last_index])
    
    #if there are nans
    if nan_perc > T_nan_limit:
        
        nans_sm = nans.sel(latitude = slice(target_lat+0.5,target_lat-0.5),
                           longitude = slice(target_lon-0.5,target_lon+0.5)) #select smaller map for speed
        if len(nans_sm)==0:
            print('Warning: location outside data. skip read') #NEED TO ADD THIS TO THE ELSE BIT TOO
            T_ERA = []
            #TODO: add similar to if no nans
        else:
        
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
            
            if len(new_lat) == 0:
                print('warning: no land within 1 degree. Possibly at domain edge. skip read')
                T_ERA = []
        
            else:
                for n, file in enumerate(T_files[first_index:last_index]): #load in the data
                    with xr.open_dataarray(file) as da:
                        T_ERA[n] = da.sel(latitude = new_lat,method = 'nearest').sel(longitude = new_lon,method = 'nearest')
                    
                T_ERA = xr.concat(T_ERA,dim = 'valid_time').sel(valid_time = slice(start_date,end_date))
                
                #if concatting has added an extra dimension, take the mean to remove nans then put latitude value back in 
                if len(np.shape(T_ERA)) == 3:
                    if np.shape(T_ERA)[1] == 2: #have only seen this once and it happened with lat, should check other dims too
                        print('aaaaaah! extra dimension weirdly')
                        T_ERA = T_ERA.mean(dim='latitude').expand_dims({"latitude": T_ERA.latitude[0].to_numpy()})
                    else:
                        pass
                else:
                    pass
                
                
                print(f'Latitudes: {target_lat} became {new_lat}. Longitudes: {target_lon} became {new_lon}. {location.to_numpy()[0][0]} metres away.')
                
                print(T_ERA)
        
    else: #if location has no nans dont do the whole distances thing
        for n, file in enumerate(T_files[first_index:last_index]):
            with xr.open_dataarray(file) as da:
                T_ERA[n] = da.sel(latitude = target_lat,method = 'nearest').sel(longitude = target_lon,method = 'nearest')
        
   
        T_ERA = xr.concat(T_ERA,dim = 'valid_time').sel(valid_time = slice(start_date,end_date))  #combine multiple time files
        if len(np.shape(T_ERA)) == 3:
            if np.shape(T_ERA)[1] == 2: #have only seen this once and it happened with lat, should check other dims too
                print('aaaaaah! extra dimension weirdly')
                T_ERA = T_ERA.mean(dim='latitude').expand_dims({"latitude": T_ERA.latitude[0].to_numpy()})
            else:
                pass
        else:
            pass
        
        
    return T_ERA
    
def read_GSDR_file(file_name,name_col):
    """
    Convert a txt file from the GSDR dataset into a pandas dataframe with datetime as the index.

    Parameters
    ----------
    file_name : string
        path to GSDR file to read.
    name_col : string
        name for the precipitation column.

    Returns
    -------
    G : pandas dataframe
        dataframe of precipitation levels and .
    data_meta : intense object
        meta data of station, including lon, lat, start_datetime.

    """
    G = pd.read_csv(file_name, skiprows=21, names=[name_col])
    data_meta = readIntense(file_name, only_metadata=True, opened=False)

       
    #extract start and end dates from metadata
    start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
    
    time_list_G = [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
    # replace -999 with nan
    G[G == -999] = np.nan
    
    G['prec_time'] = time_list_G
    G = G.set_index('prec_time')
    return G, data_meta

















