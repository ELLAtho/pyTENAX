# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:15:13 2025

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



# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 50


country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30



# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 30

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

save_name = f"{drive}:/outputs/{country_save}\\average_temp_shape.csv"
df = pd.read_csv(save_name,dtype = {0:str})

df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'
df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 

#merging the dataframes to ensure station consistency
missing_rows = pd.merge(df_parameters.station, df.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
else:
    pass




eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
eTs = eTs_df.drop(columns = "station").to_numpy()

n_peaks = [0]*len(df)
n_peaks01 = [0]*len(df)
heights = [[np.nan,np.nan,np.nan,np.nan,np.nan]]*len(df)
diffs = [[np.nan,np.nan,np.nan,np.nan]]*len(df)

for i in range(len(df)):
    x, y = eTs[i],df.iloc[i][1:]
    peaks = find_peaks(y,height = 0.000)
    peaks01 = find_peaks(y,height = 0.01)
    n_peaks[i] = len(peaks[0])
    n_peaks01[i] = len(peaks01[0])
    for j in range(n_peaks[i]):
        heights[i][j] = peaks[1]["peak_heights"][j]
    for j in np.arange(0,n_peaks[i]-1):
        diffs[i][j] = x[peaks[0][j+1]] -x[peaks[0][j]] 
        
    # if i % 50 == 0:
    #     print(i)

peaks_df = pd.DataFrame({
    "n_peaks" : n_peaks,
    "n_peaks01" : n_peaks01,
    "height_1" : np.array(heights)[:,0],
    "height_2" : np.array(heights)[:,1],
    "height_3" : np.array(heights)[:,2],
    "height_4" : np.array(heights)[:,3],
    "height_5" : np.array(heights)[:,4],
    "diff_1" : np.array(diffs)[:,0],
    "diff_2" : np.array(diffs)[:,1],
    "diff_3" : np.array(diffs)[:,2],
    "diff_4" : np.array(diffs)[:,3],

    
    })


cmap = 'plasma'
bounds = [0.5,1.5,2.5,3.5,4.5]  # 3 discrete levels
norm = mcolors.BoundaryNorm(bounds, plt.get_cmap(cmap).N)


fig = plt.figure(figsize=(15, 10))

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=peaks_df.n_peaks,
    cmap=cmap,
    norm = norm,
)
ax1.set_title("height = 0")


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 2, 2, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=peaks_df.n_peaks01,
    cmap=cmap,
    norm = norm
)
ax1.set_title("height = 0.01")
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
cbar = plt.colorbar(sc, shrink = 0.2, cax=cbar_ax, ticks=[1, 2, 3, 4])

plt.show()




###############################################################################
S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0,
        min_ev_dur = 60,
        beta = 4
    )
#Skewness
save_name_skew = f"{drive}:/outputs/{country_save}\\temp_skew.csv"
output_files = glob.glob(f"{drive}:/outputs/{country_save}/*")

if save_name_skew not in output_files:
    print("temp skewed not calculated yet. here we gooooooo")
    

    g_phat_skew = [0] * len(df)
    start_time = [0] * len(df)
    
    for i in np.arange(0, len(df)):
        start_time[i] = time.time()
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
                
            
            if 'code_str' in locals():
                G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
            else:
                G = pd.read_csv(f"{file_name}.csv")
                G['prec_time'] = pd.to_datetime(G['prec_time'])
                G.set_index('prec_time', inplace=True)
                
                
            data = S.remove_incomplete_years(G, name_col)
            
            T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
            
            if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                print('skip')
                T = [np.nan]
            else:
                T_ERA = xr.load_dataarray(T_path)
                t_data = (T_ERA.squeeze()-273.15).to_dataframe()
                
        
                df_arr = np.array(data[name_col])
                df_dates = np.array(data.index)
                
                #extract indexes of ordinary events
                #these are time-wise indexes =>returns list of np arrays with np.timeindex
                idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
                    
                
                #get ordinary events by removing too short events
                #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
                _,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
                
                #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
                dict_ordinary, _ = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
                
                
                
                df_arr_t_data = np.array(t_data[temp_name_col])
                df_dates_t_data = np.array(t_data.index)
                
                if type(df_dates_t_data[0]) != np.datetime64:
                        
                    df_dates_t_data = pd.Series([item[0] for item in df_dates_t_data])
                    df_dates_t_data = np.array(df_dates_t_data)
                else:
                    pass
                
                dicts, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                
                #g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]] 
                
                
                # Your data (P, T arrays) and threshold thr=3.8
                P = dicts["60"]["ordinary"].to_numpy() 
                T = dicts["60"]["T"].to_numpy()  
                
                np.savetxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv",T)
                np.savetxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv",P)
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv")
        
        if len(T) <= 2:
            g_phat_skew[i] = [np.nan]*3
        else:
            g_phat_skew[i] = S.temperature_model(T, method = "skewnorm")
        
        
        if i%50 == 0: 
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(df)-i)*time_taken/60
            print(f"{i}/{len(df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
        
    skew_df = pd.DataFrame({
        'station': df_parameters.station,
        "skewness" : np.array(g_phat_skew)[:,0],
        "g_phat1" : np.array(g_phat_skew)[:,1],
        "g_phat2" : np.array(g_phat_skew)[:,2],
        })
    skew_df.to_csv(save_name_skew,index=False)
else:
    skew_df = pd.read_csv(save_name_skew, dtype={"station":str})
        
    


###############################################################################
#plot

lon_lims = [truncate_neg(np.min(df_parameters.longitude),5),np.ceil(np.max(df_parameters.longitude/5))*5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]


cmap = "seismic"
s = 3
norm = mcolors.Normalize(vmin=np.min(skew_df.skewness)*0.6, vmax=np.min(skew_df.skewness)*-0.6)
fig = plt.figure(figsize=(10, 10))

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=skew_df.skewness,
    cmap=cmap,
    norm = norm,
    s = s
)
ax1.set_title("skewness")
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
plt.colorbar(sc,extend = "both")


proj = ccrs.PlateCarree()

cmap = 'plasma'
bounds = [0.5,1.5,2.5,3.5,4.5]  # 3 discrete levels
norm = mcolors.BoundaryNorm(bounds, plt.get_cmap(cmap).N)

ax1 = fig.add_subplot(2, 1, 2, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=peaks_df.n_peaks01,
    cmap=cmap,
    norm = norm,
    s = s
)
ax1.set_title("number of peaks (height = 0.01)")
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
plt.colorbar(sc,ticks=[1, 2, 3, 4])

plt.show()

print(f"max skew: {np.max(skew_df.skewness)}")
print(f"min skew: {np.min(skew_df.skewness)}")


df_parameters_north = df_parameters[df_parameters.latitude > max_lat+10]
peaks_df_north = peaks_df[df_parameters.latitude > max_lat+10]

north_3peak = df_parameters_north[peaks_df_north.n_peaks01 == 3]

eTs_df_north = eTs_df[df_parameters.latitude > max_lat+10]

eTs_3peak = eTs_df_north[peaks_df_north.n_peaks01 == 3]

df_3peak = df[df_parameters.latitude > max_lat+10][peaks_df_north.n_peaks01 == 3]


for i in range(len(north_3peak)):
    oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{north_3peak.station.iloc[i]}.csv"
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{north_3peak.station.iloc[i]}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{north_3peak.station.iloc[i]}.csv")
    
    g_phat_skew = S.temperature_model(T, method = "skewnorm")
    eT = np.arange(np.min(T),np.max(T)+4)
    TNX_FIG_temp_model(T, g_phat_skew, 4, eT, obscol='r',valcol='b',
                           obslabel = 'observations',
                           vallabel = 'skewed normal',
                           xlimits = [-15,30],
                           ylimits = [0,0.06],
                           method = "skewnorm")
    plt.plot(eTs_3peak.iloc[i][1:],df_3peak.iloc[i][1:])
    
    plt.show()







