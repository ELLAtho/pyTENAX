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


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 30



country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30

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