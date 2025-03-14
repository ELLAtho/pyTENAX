# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:50:36 2025

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
from scipy.stats import anderson
from scipy.stats import gaussian_kde

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import xarray as xr
import time

drive = 'D'

country = 'Japan'
country_save = 'Japan'
code_str = 'JP' 
n_stations = 2 #number of stations to sample
min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
max_yrs = 1000 #if no max, set to very high
name_col = 'ppt'
temp_name_col = "t2m"


temp_output_files = glob.glob(f"{drive}:/outputs/{country_save}/temp_FRMSE*")
df = [0]*len(temp_output_files)
label = [0]*len(temp_output_files)
df_parameters = pd.read_csv(f"{drive}:/outputs/{country_save}\\parameters.csv", dtype={'station': str}) 


for i in range(len(temp_output_files)):
    label[i] = temp_output_files[i][len(country_save)+12:-4]
    df[i] = pd.read_csv(temp_output_files[i], dtype={'station': str})


all_temp_FRMSE = pd.DataFrame({
    "station": df[0].station
    })

for i in range(len(temp_output_files)):
    all_temp_FRMSE[f"{label[i]}_upper_perc"] = df[i].FRMSE_upper_perc
    all_temp_FRMSE[label[i]] = df[i].FRMSE
    
    
number_betas = len(temp_output_files)

box_list = [all_temp_FRMSE[lab].copy().dropna() for lab in label]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks(range(1,number_betas+1),label)
plt.title(f'{country} FRMSE')
plt.show()
    
    
#upper percent
box_list = [all_temp_FRMSE[lab].copy().dropna() 
            for lab in [f"{label[i]}_upper_perc" for i in range(len(temp_output_files))]]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks(range(1,number_betas+1),label)
plt.title(f'{country} FRMSE upper 20%')
plt.show()  
    
    
#differences
box_list = [all_temp_FRMSE.temp_FRMSE6.copy().dropna() - all_temp_FRMSE.temp_FRMSE.copy().dropna()]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1],["beta = 6 - beta = 4"])
plt.title(f'{country} FRMSE')
plt.show()  
    
    
#differences 20%
box_list = [all_temp_FRMSE.temp_FRMSE6_upper_perc.copy().dropna() - all_temp_FRMSE.temp_FRMSE_upper_perc.copy().dropna()]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1],["beta = 6 - beta = 4"])
plt.title(f'{country} FRMSE upper 20%')
plt.show()  
    



#plots
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
s = 3


fig = plt.figure(figsize=(10, 10))
norm = mcolors.Normalize(vmin=-0.3, vmax=0.3)
cmap = 'seismic'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = all_temp_FRMSE.temp_FRMSE6 - all_temp_FRMSE.temp_FRMSE,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("beta = 6 - beta = 4. full")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('FRMSE', fontsize=10)


ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=all_temp_FRMSE.temp_FRMSE6_upper_perc - all_temp_FRMSE.temp_FRMSE_upper_perc,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("beta = 6 - beta = 4. Upper 20%")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('FRMSE', fontsize=10)
plt.show()









































