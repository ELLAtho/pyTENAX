# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:12:48 2025

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



drive = 'D'



country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9


## READ IN FILES
save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'


df_parameters = pd.read_csv(f"{drive}:/outputs/{country_save}\\parameters.csv", dtype={'station': str}) 
df_parameters_0 = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/parameters.csv", dtype={'station': str})


if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg, dtype={'station': str})

    #dataframe with all values
    new_df = df_parameters[['station','latitude','longitude','b','kappa','lambda','a']].copy()
    
    mask = new_df['b'] == 0
    
    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
    new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
    new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()

else:
    new_df = df_parameters.copy()

#merging the dataframes to ensure station consistency
missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    new_df = new_df.drop(missing_rows.index)
else:
    pass

save_name = f"{drive}:/outputs/{country_save}\\return_levels.csv"

RL_df = pd.read_csv(save_name, dtype={'station': str})
nan_locs = RL_df.return_levels[RL_df.return_levels.isna()].index
replace_range = np.arange(0,len(RL_df))
for k in range(len(nan_locs)):
    replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
for j in replace_range:
    RL_df.loc[j, "return_levels"] = np.fromstring(RL_df.return_levels.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    RL_df.loc[j, "return_levels_5"] = np.fromstring(RL_df["return_levels_5"].iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    RL_df.loc[j, "return_levels_b0"] = np.fromstring(RL_df.return_levels_b0.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    RL_df.loc[j, "obs_AMS"] = np.fromstring(RL_df.obs_AMS.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')

    
    if "return_levels_bset" in RL_df.columns:
        RL_df.loc[j, "return_levels_bset"] = np.fromstring(RL_df.return_levels_bset.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
    if "return_levels_bexp" in RL_df.columns:
        RL_df.loc[j, "return_levels_bexp"] = np.fromstring(RL_df.return_levels_bexp.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
   

FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE.csv", dtype={'station': str})
FRMSE_df = FRMSE_df.drop("FRMSE_5",axis=1) 

save_name6 = f"{drive}:/outputs/{country_save}\\return_levels6.csv"

RL_df6 = pd.read_csv(save_name, dtype={'station': str})
nan_locs = RL_df6.return_levels[RL_df6.return_levels.isna()].index
replace_range = np.arange(0,len(RL_df6))
for k in range(len(nan_locs)):
    replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
for j in replace_range:
    RL_df6.loc[j, "return_levels"] = np.fromstring(RL_df6.return_levels.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    RL_df6.loc[j, "return_levels_5"] = np.fromstring(RL_df6["return_levels_5"].iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    RL_df6.loc[j, "return_levels_b0"] = np.fromstring(RL_df6.return_levels_b0.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    RL_df6.loc[j, "obs_AMS"] = np.fromstring(RL_df6.obs_AMS.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')

    
    if "return_levels_bset" in RL_df.columns:
        RL_df6.loc[j, "return_levels_bset"] = np.fromstring(RL_df6.return_levels_bset.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
    if "return_levels_bexp" in RL_df.columns:
        RL_df6.loc[j, "return_levels_bexp"] = np.fromstring(RL_df6.return_levels_bexp.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
   

FRMSE_df6 = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE6.csv", dtype={'station': str})
FRMSE_df6 = FRMSE_df6.drop("FRMSE_5",axis=1) 


liklihood_df = pd.read_csv(f"{drive}:/outputs/{country_save}/liklihood.csv",dtype={'station': str})

nan_locs = liklihood_df.mult_prob[liklihood_df.mult_prob.isna()].index
replace_range = np.arange(0,len(RL_df))
for k in range(len(nan_locs)):
    replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
for j in replace_range:
    liklihood_df.at[j, "maxes"] = eval(liklihood_df.maxes.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df.at[j, "mins"] = eval(liklihood_df.mins.iloc[j], {"np": np, "nan": np.nan})
    
    liklihood_df.at[j, "maxes_0"] = eval(liklihood_df.maxes_0.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df.at[j, "mins_0"] = eval(liklihood_df.mins_0.iloc[j], {"np": np, "nan": np.nan})
    
    liklihood_df.at[j, "maxes_5"] = eval(liklihood_df.maxes_5.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df.at[j, "mins_5"] = eval(liklihood_df.mins_5.iloc[j], {"np": np, "nan": np.nan})
  
#drop the 5% sig ones  
liklihood_df = liklihood_df.drop([s for s in liklihood_df.columns if "_5" in s],axis = 1)


liklihood_df6 = pd.read_csv(f"{drive}:/outputs/{country_save}/liklihood6.csv",dtype={'station': str})

nan_locs = liklihood_df6.mult_prob[liklihood_df6.mult_prob.isna()].index
replace_range = np.arange(0,len(RL_df))
for k in range(len(nan_locs)):
    replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
for j in replace_range:
    liklihood_df6.at[j, "maxes"] = eval(liklihood_df6.maxes.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df6.at[j, "mins"] = eval(liklihood_df6.mins.iloc[j], {"np": np, "nan": np.nan})
    
    liklihood_df6.at[j, "maxes_0"] = eval(liklihood_df6.maxes_0.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df6.at[j, "mins_0"] = eval(liklihood_df6.mins_0.iloc[j], {"np": np, "nan": np.nan})
    
    liklihood_df6.at[j, "maxes_5"] = eval(liklihood_df6.maxes_5.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df6.at[j, "mins_5"] = eval(liklihood_df6.mins_5.iloc[j], {"np": np, "nan": np.nan})
    
#drop the 5% sig ones
liklihood_df6 = liklihood_df6.drop([s for s in liklihood_df6.columns if "_5" in s],axis = 1)

###############################################################################
#FRMSE
#BETA COMP

box_list_fr = [FRMSE_df6.copy().dropna().FRMSE - FRMSE_df.copy().dropna().FRMSE, FRMSE_df6.copy().dropna().FRMSE_0 - FRMSE_df.copy().dropna().FRMSE_0]
labels_list = ["b free", "b = 0"]

fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(2,2,1)

ax1.boxplot(box_list_fr,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax1.set_title('FRMSE beta = 6 - beta = 4')

ax2 = fig.add_subplot(2,2,2)
ax2.boxplot(box_list_fr,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax2.set_title('FRMSE beta = 6 - beta = 4')
plt.xlim(-0.05,0.05)

box_list_lik = [np.log(liklihood_df6.copy().dropna().mult_prob) - np.log(liklihood_df.copy().dropna().mult_prob),
            np.log(liklihood_df6.copy().dropna().mult_prob_0) - np.log(liklihood_df.copy().dropna().mult_prob_0)]

labels_list = ["b free", "b = 0"]


ax3 = fig.add_subplot(2,2,3)
ax3.boxplot(box_list_lik,vert=False)
plt.xlabel('log(liklihood')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax3.set_title('liklihood beta = 6 - beta = 4')


ax4 = fig.add_subplot(2,2,4)
ax4.boxplot(box_list_lik,vert=False)
plt.xlabel('log(liklihood')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax4.set_title('liklihood beta = 6 - beta = 4')
plt.xlim(-10,10)


plt.show()


#map
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
    c = FRMSE_df6.FRMSE - FRMSE_df.FRMSE,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b=free. beta = 6 - beta = 4")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('FRMSE', fontsize=10)


ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=FRMSE_df6.FRMSE_0 - FRMSE_df.FRMSE_0,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b=0. beta = 6 - beta = 4")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('FRMSE', fontsize=10)


norm = mcolors.Normalize(vmin=np.min(box_list_lik[0]), vmax=-1 * np.min(box_list_lik[0]))

ax1 = fig.add_subplot(2, 2, 3, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = -1 * (np.log(liklihood_df6.mult_prob) - np.log(liklihood_df.mult_prob)),
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b=free. beta = 6 - beta = 4")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('-log(liklihood)', fontsize=10)


ax2 = fig.add_subplot(2, 2, 4, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = -1 * (np.log(liklihood_df6.mult_prob_0) - np.log(liklihood_df.mult_prob_0)),
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b=0. beta = 6 - beta = 4")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('-log(liklihood)', fontsize=10)

plt.suptitle("red means beta = 6 is worse")
fig.tight_layout()
plt.show()











