# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:53:56 2025

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
from scipy.stats import ttest_ind

import datetime as dt
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

import xarray as xr
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from matplotlib import cm
import alphashape
from shapely.geometry import Polygon
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator

drive = 'D'


countries = ["germany","Japan","UK","US"]
country_saves = ["germany","Japan","UK","US_main"]
code_strs = ["DP_","JP_","UK_","US_"]

lons_lats = [[47, 3, 55, 15],[24, 122.9, 45.6, 145.8],[] ,[24, -125, 56, -66]]


country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9

save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 
TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters

df_parameters_0 = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/parameters.csv", dtype={'station': str})
df_parameters_exp = pd.read_csv(f"{drive}:/outputs/{country_save}/parameters_exp.csv", dtype={'station': str})

# for some reason in germany there is one less row...


if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg, dtype={'station': str})

    #dataframe with all values
    new_df = df_parameters[['station','latitude','longitude','b','kappa','lambda','a','mu','sigma','thr','n_events_per_yr']].copy()
    
    mask = new_df['b'] == 0
    
    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
    new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
    new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()

else:
    new_df = df_parameters.copy()

missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    df_parameters = df_parameters.reindex(index = range(len(df_parameters)))
    new_df = new_df.drop(missing_rows.index)
    new_df = new_df.reindex(index = range(len(new_df)))
else:
    pass

# FIG 1

# FIG 2
# maps of spatial distributions

s=3
####################################################
#plot at 5% sig
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

# Choosing cmap
# if df_parameters.b.min() == 0:
#     norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
# else:
#     norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())


norm = norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
# for lon, lat in zip(df_parameters.longitude[df_parameters.thr == 0], 
#                     df_parameters.latitude[df_parameters.thr == 0]):
#     square = patches.Rectangle(
#         (lon - 0.5, lat - 0.5),  # Bottom-left corner of the square
#         1,  # Width (1 degree)
#         1,  # Height (1 degree)
#         color='r',
#         alpha = 0.2,
#         label="no ERA data within 1 deg" if 'no ERA data within 1 deg' not in ax1.get_legend_handles_labels()[1] else ""
#     )
#     ax1.add_patch(square)
sc = ax1.scatter(
    df_parameters.longitude[df_parameters.b==0],
    df_parameters.latitude[df_parameters.b==0],
    s = s,
    color = 'darkgrey',  
)

sc = ax1.scatter(
    df_parameters.longitude[df_parameters.b!=0],
    df_parameters.latitude[df_parameters.b!=0],
    c=df_parameters.b[df_parameters.b!=0],
    s = s,
    cmap='seismic',  
    norm=norm
)

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}


# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05, extend = "both")
cb.set_label('b', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks

# plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
# plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

plt.title(f'GSDR: {ERA_country}. b at {TENAX_use.alpha[0]} sig level', fontsize=16)
plt.legend()
plt.show()

############################################
## PLOT ALL b 2
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

# # Choosing cmap
# if df_parameters.b.min() == 0:
#     norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
# else:
#     norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())

sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = new_df.b,
    s = s,
    cmap = 'seismic',
    norm = norm
)



# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05, extend = "both")
cb.set_label('b', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}


# plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
# plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. b at 0 sig level', fontsize=16)
plt.show()
#THIS SHOWS THE LOCATION OF THE STATION, NOT THE ERA DATA

###############################################################
#Average temps
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=df_parameters.mu,
    s = s,
    cmap='Reds',  
)




# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('μ (°C)', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. μ', fontsize=16)
plt.show()


###############################################################









