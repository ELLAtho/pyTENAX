# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:07:20 2024

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


country = 'Germany'
n_stations = 5
min_yrs = 15
name_col = 'ppt'
temp_name_col = "t2m"



#READ IN META INFO FOR COUNTRY
latslons = pd.DataFrame(np.load('D:/'+country+'_latslons.npy').T,columns = ['latitude','longitude'])
dates = pd.DataFrame(np.load('D:/'+country+'_dates.npy').T,columns = ['startdate','enddate'])
info = pd.DataFrame(np.load('D:/'+country+'_data.npy').T,columns = ['station','missing_data_perc','total_years','cleaned_years'])

comb = pd.concat([info, latslons,dates], axis=1)


#select stations
val_comb = comb[comb['cleaned_years']>=min_yrs]


comb_sort = val_comb.sort_values(by=['cleaned_years'],ascending=0)


selected = comb_sort[0:n_stations] #choose top n_stations stations

#PLOT SELECTED STATIONS LOCATIONS
plt.scatter(selected['longitude'],selected['latitude'])
plt.show()

#READ IN RAIN DATA
files = glob.glob('D:/'+country+'/*') #list of files in country folder
files_sel = [files[i] for i in selected.index]

print(files_sel) #check files and stations the same
print(selected['station'])

#make empty lists to read into
G = [0]*n_stations
data_meta = [0]*n_stations


for i in np.arange(0, n_stations):
    G[i] = pd.read_csv(files_sel[i], skiprows=21, names=[name_col])
    data_meta[i] = readIntense(files_sel[i], only_metadata=True, opened=False)


       
    #extract start and end dates from metadata
    start_date_G= dt.datetime.strptime(data_meta[i].start_datetime, "%Y%m%d%H")
    end_date_G= dt.datetime.strptime(data_meta[i].end_datetime, "%Y%m%d%H")
    
    time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G[i].size)] #make timelist of size of FI
    # replace -999 with nan
    G[i][G[i] == -999] = np.nan
    
    G[i]['prec_time'] = time_list_G
    G[i] = G[i].set_index('prec_time')


#READ IN ERA5 DATA
T_files = sorted(glob.glob('D:/ERA5_land/'+country+'*/*'))

#get selected lats and lons
lats_sel = [selected.latitude[i] for i in selected.index]
lons_sel = [selected.longitude[i] for i in selected.index]


T_ERA = [0]*n_stations

for i in np.arange(0,n_stations): #for each selected station
    
    T_temp = [0]*np.size(T_files)
    for n in np.arange(0,np.size(T_files)): #select temperature data a gridpoint closest to station
        T_temp[n] = xr.open_dataarray(T_files[n]).sel(latitude = lats_sel[i],method = 'nearest').sel(longitude = lons_sel[i],method = 'nearest')
        
    T_ERA[i] = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files


#NOW WE  DO TENAX
S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
    )


data = [0]*n_stations
for i in np.arange(0,n_stations):
    data[i] = S.remove_incomplete_years(G[i], name_col)


# FOR NOW DO WITH ONE STATION - SET LOOP HERE
data = data[0]
t_data = (T_ERA[0]-273.15).to_dataframe()

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
RL, _, __ = S.model_inversion(F_phat, g_phat, n, Ts)
print(RL)







