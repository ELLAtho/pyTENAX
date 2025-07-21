# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:22:37 2025

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
from scipy.stats import skewnorm, skew, kurtosis
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

drive = 'D'



country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30

region_lats = [minlat,37.5,maxlat]
region_lons = [minlon,-116,-105,-90,maxlon]


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

min_yrs = 10

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
val_info = val_info.reset_index()


# read in data for storm types with matching stations
station_ids = val_info.station.to_numpy()

saved_files = glob.glob(f"{drive}:/outputs/{country_save}/*")
savename = f"{drive}:/outputs/{country_save}\\temp_skew_kurt.csv"

if savename not in saved_files:
    print("calculating temperature shape things")
    
    kurts = [0]*len(station_ids)
    skews = [0]*len(station_ids)
    
    kurts_full = [0]*len(station_ids)
    skews_full = [0]*len(station_ids)
    
    
    start_time = [0]*len(station_ids)
    
    for i in range(len(station_ids)):
        start_time[i] = time.time()
        station = station_ids[i]
        full_temp = xr.load_dataarray(f"D:/US_temp/US_{station}.nc").to_numpy().squeeze() - 273.15
        
        T_ = np.genfromtxt(f"D:/ordinary_events/US_main/T_{station}.csv")
        
        
        kurts[i] = kurtosis(T_)
        skews[i] = skew(T_)
        
        kurts_full[i] = kurtosis(full_temp)
        skews_full[i] = skew(full_temp)
        
        
        if i%50 == 0:    
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(station_ids)-i)*time_taken/60
            print(f"skew: {skews[i]}, skew of full: {skews_full[i]}")
            print(f"kurtosis: {kurts[i]}, kurtosis of full: {kurts_full[i]}")
            print(f"{i}/{len(station_ids)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
        
    skew_kurt_df = pd.DataFrame({
        "station": station_ids,
        "kurts" : kurts,
        "skews" : skews,
        "kurt_full_temp" : kurts_full,
        "skew_full_temp" : skews_full,
        })
    skew_kurt_df.to_csv(savename)

else:
    skew_kurt_df = pd.read_csv(savename, dtype={'station': str})



save_name = f"{drive}:/outputs/{country_save}\\average_temp_shape.csv"
df = pd.read_csv(save_name,dtype = {0:str})
eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
eTs = eTs_df.drop(columns = "station").to_numpy()

n_peaks = [0]*len(df)
n_peaks01 = [0]*len(df)
prominences = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan]]*len(df))
diffs = np.array([[np.nan,np.nan,np.nan,np.nan]]*len(df))

for i in range(len(df)):
    x, y = eTs[i],df.iloc[i][1:]
    peaks = find_peaks(y,prominence = 0.000)
    peaks01 = find_peaks(y,prominence = 0.001)
    n_peaks[i] = len(peaks[0])
    n_peaks01[i] = len(peaks01[0])
    for j in range(n_peaks[i]):
        prominences[i][j] = peaks[1]["prominences"][j]
    for j in np.arange(0,n_peaks[i]-1):
        diffs[i][j] = x[peaks[0][j+1]] -x[peaks[0][j]] 
        
    # if i % 50 == 0:
    #     print(i)

peaks_df = pd.DataFrame({
    "n_peaks" : n_peaks,
    "n_peaks01" : n_peaks01,
    "prominence_1" : prominences[:,0],
    "prominence_2" : prominences[:,1],
    "prominence_3" : prominences[:,2],
    "prominence_4" : prominences[:,3],
    "prominence_5" : prominences[:,4],
    "diff_1" : diffs[:,0],
    "diff_2" : diffs[:,1],
    "diff_3" : diffs[:,2],
    "diff_4" : diffs[:,3],

    
    })



bounds = [0.5,1.5,2.5,3.5,4.5]  # 3 discrete levels
norm_peak = mcolors.BoundaryNorm(bounds, plt.get_cmap("plasma").N)

norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
norm_kurt = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

s = 3
fontsize = 15


n_lat = len(region_lats)-1
n_lon = len(region_lons)-1


fig = plt.figure(figsize=(12, 14))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(3,2,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c=skew_kurt_df.skews,
    s = s,
    norm = norm,
    cmap = "RdBu"
    )


gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")


cb = plt.colorbar(sc, orientation='horizontal', extend = "both")
cb.ax.tick_params(labelsize=fontsize)
cb.set_label("skew", fontsize=fontsize)


plt.xlim(-125,-70)
plt.ylim(25,50)

plt.title("Skew storms", fontsize = fontsize)


ax = fig.add_subplot(3,2,2, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c=skew_kurt_df.skew_full_temp,
    s = s,
    norm = norm,
    cmap = "RdBu"
    )


gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")


cb = plt.colorbar(sc, orientation='horizontal', extend = "both")
cb.ax.tick_params(labelsize=fontsize)
cb.set_label("skew", fontsize=fontsize)


plt.xlim(-125,-70)
plt.ylim(25,50)

plt.title("Skew full temp", fontsize = fontsize)


ax = fig.add_subplot(3,2,3, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c=skew_kurt_df.kurts,
    s = s,
    norm = norm_kurt,
    cmap = "PiYG"
    )


gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")


cb = plt.colorbar(sc, orientation='horizontal', extend = "both")
cb.ax.tick_params(labelsize=fontsize)
cb.set_label("kurtosis", fontsize=fontsize)


plt.xlim(-125,-70)
plt.ylim(25,50)

plt.title("Kurtosis storms", fontsize = fontsize)

ax = fig.add_subplot(3,2,4, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c=skew_kurt_df.kurt_full_temp,
    s = s,
    norm = norm_kurt,
    cmap = "PiYG"
    )


gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")


cb = plt.colorbar(sc, orientation='horizontal', extend = "both")
cb.ax.tick_params(labelsize=fontsize)
cb.set_label("kurtosis", fontsize=fontsize)


plt.xlim(-125,-70)
plt.ylim(25,50)

plt.title("Kurtosis full temp", fontsize = fontsize)

ax = fig.add_subplot(3,2,5, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c=peaks_df.n_peaks,
    s = s,
    norm = norm_peak,
    cmap = "plasma"
    )


gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

cb = plt.colorbar(sc, orientation='horizontal', ticks=[1, 2, 3, 4])
cb.ax.tick_params(labelsize=fontsize)
cb.set_label("n peaks", fontsize=fontsize)


plt.xlim(-125,-70)
plt.ylim(25,50)

plt.title("n peaks", fontsize = fontsize)



plt.show()










