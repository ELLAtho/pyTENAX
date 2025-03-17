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


name_col = 'ppt' 
temp_name_col = "t2m"

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
#BETA COMP

box_list_fr = [FRMSE_df6.copy().dropna().FRMSE - FRMSE_df.copy().dropna().FRMSE, FRMSE_df6.copy().dropna().FRMSE_0 - FRMSE_df.copy().dropna().FRMSE_0]
labels_list = ["b free", "b = 0"]

fig = plt.figure(figsize = (10,15))
ax1 = fig.add_subplot(3,2,1)

ax1.boxplot(box_list_fr,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax1.set_title('FRMSE beta = 6 - beta = 4')

ax2 = fig.add_subplot(3,2,2)
ax2.boxplot(box_list_fr,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax2.set_title('FRMSE beta = 6 - beta = 4')
plt.xlim(-0.05,0.05)

box_list_lik = [np.log(liklihood_df6.copy().dropna().mult_prob) - np.log(liklihood_df.copy().dropna().mult_prob),
            np.log(liklihood_df6.copy().dropna().mult_prob_0) - np.log(liklihood_df.copy().dropna().mult_prob_0)]

labels_list = ["b free", "b = 0"]


ax3 = fig.add_subplot(3,2,3)
ax3.boxplot(box_list_lik,vert=False)
plt.xlabel('log(liklihood)')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax3.set_title('liklihood beta = 6 - beta = 4')


ax4 = fig.add_subplot(3,2,4)
ax4.boxplot(box_list_lik,vert=False)
plt.xlabel('log(liklihood)')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax4.set_title('liklihood beta = 6 - beta = 4')
plt.xlim(-10,10)


box_list_ave = [np.log(liklihood_df6.ave_prob) - np.log(liklihood_df.ave_prob),
            np.log(liklihood_df6.ave_prob_0) - np.log(liklihood_df.ave_prob_0)]

ax3 = fig.add_subplot(3,2,5)
ax3.boxplot([box.copy().dropna() for box in box_list_ave],vert=False)
plt.xlabel('average probability')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax3.set_title('liklihood beta = 6 - beta = 4')


ax4 = fig.add_subplot(3,2,6)
ax4.boxplot([box.copy().dropna() for box in box_list_ave],vert=False)
plt.xlabel('average probability')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
ax4.set_title('liklihood beta = 6 - beta = 4')
plt.xlim(-0.2,0.2)


plt.show()


#map
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
s = 3


fig = plt.figure(figsize=(10, 15))
norm = mcolors.Normalize(vmin=-0.3, vmax=0.3)
cmap = 'seismic'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 2, 1, projection=proj)

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


ax2 = fig.add_subplot(3, 2, 2, projection=proj)
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
cmap = 'seismic_r'

ax1 = fig.add_subplot(3, 2, 3, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = (np.log(liklihood_df6.mult_prob) - np.log(liklihood_df.mult_prob)),
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
cb.set_label('log(liklihood)', fontsize=10)


ax2 = fig.add_subplot(3, 2, 4, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = (np.log(liklihood_df6.mult_prob_0) - np.log(liklihood_df.mult_prob_0)),
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
cb.set_label('log(liklihood)', fontsize=10)


norm = mcolors.Normalize(vmin=-1, vmax=1)
cmap = 'seismic_r'

ax1 = fig.add_subplot(3, 2, 5, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = box_list_ave[0],
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
cb.set_label('average probability', fontsize=10)


ax2 = fig.add_subplot(3, 2, 6, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = box_list_ave[1],
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
cb.set_label('average probability', fontsize=10)


plt.suptitle("red means beta = 6 is worse")
fig.tight_layout()
plt.show()


###############################################################################
# b comp
box_list_fr = [FRMSE_df.FRMSE - FRMSE_df.FRMSE_0, FRMSE_df.FRMSE_bexp - FRMSE_df.FRMSE_0]
labels_list = ["b = free - b = 0", "b = exp - b = 0"]

fig = plt.figure(figsize = (10,15))
ax1 = fig.add_subplot(3,2,1)

ax1.boxplot([box.copy().dropna() for box in box_list_fr],vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)

ax2 = fig.add_subplot(3,2,2)
ax2.boxplot([box.copy().dropna() for box in box_list_fr],vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
plt.xlim(-0.05,0.05)

box_list_lik = [np.log(liklihood_df.mult_prob) - np.log(liklihood_df.mult_prob_0),
            np.log(liklihood_df.mult_prob_bexp) - np.log(liklihood_df.mult_prob_0)]



ax3 = fig.add_subplot(3,2,3)
ax3.boxplot([box.copy().dropna() for box in box_list_lik],vert=False)
plt.xlabel('log(liklihood)')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)


ax4 = fig.add_subplot(3,2,4)
ax4.boxplot([box.copy().dropna() for box in box_list_lik],vert=False)
plt.xlabel('log(liklihood)')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)
plt.xlim(-10,10)


box_list_ave = [liklihood_df.ave_prob - liklihood_df.ave_prob_0,
            liklihood_df.ave_prob_bexp - liklihood_df.ave_prob_0]



ax3 = fig.add_subplot(3,2,5)
ax3.boxplot([box.copy().dropna() for box in box_list_ave],vert=False)
plt.xlabel('average probability')
#plt.xlim(-0.1,0.2)
plt.yticks([1,2],labels_list)


ax4 = fig.add_subplot(3,2,6)
ax4.boxplot([box.copy().dropna() for box in box_list_ave],vert=False)
plt.xlabel('average probability')
plt.yticks([1,2],labels_list)
plt.xlim(-0.05,0.05)


plt.show()


#map
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
s = 3


fig = plt.figure(figsize=(10, 15))
norm = mcolors.Normalize(vmin=-0.3, vmax=0.3)
cmap = 'seismic'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = box_list_fr[0],
    cmap=cmap,
    norm = norm,
    s = s,
)

ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b=free - b = 0")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('FRMSE', fontsize=10)


ax2 = fig.add_subplot(3, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=box_list_fr[1],
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b = exp - b = 0")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('FRMSE', fontsize=10)


norm = mcolors.Normalize(vmin=np.min(box_list_lik[0]), vmax=-1 * np.min(box_list_lik[0]))
cmap = 'seismic_r'

ax1 = fig.add_subplot(3, 2, 3, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = box_list_lik[0],
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b = free - b = 0")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('log(liklihood)', fontsize=10)


ax2 = fig.add_subplot(3, 2, 4, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = box_list_lik[1],
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b = exp - b = 0")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('log(liklihood)', fontsize=10)



norm = mcolors.Normalize(vmin=-np.max(box_list_ave[0]), vmax=np.max(box_list_ave[0]))
ax1 = fig.add_subplot(3, 2, 5, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = box_list_ave[0],
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b = free - b = 0")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('average probability', fontsize=10)


ax2 = fig.add_subplot(3, 2, 6, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = box_list_ave[1],
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b = exp - b = 0")
cb = plt.colorbar(sc,extend = "both")
cb.set_label('average probability', fontsize=10)

plt.suptitle("blue means b = 0 is worse")
fig.tight_layout()
plt.show()








###############################################################################
# TEMP

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

###############################################################################

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = 0,
        min_ev_dur = 60,
        niter_smev = 1000, 
        beta = 4
    )


g_phats6 = pd.read_csv(f"{drive}:\outputs\{country_save}\g_phat6", dtype={'station': str})

# LOOK AT SOME STATIONS
for j in np.arange(0,5):
    plot_pos = np.arange(1,np.size(RL_df.obs_AMS.iloc[j])+1)/(1+np.size(RL_df.obs_AMS.iloc[j]))
    
    eRP = 1/(1-plot_pos)
    
    TNX_FIG_valid(RL_df.obs_AMS.iloc[j],eRP,RL_df.return_levels_b0.iloc[j],TENAXlabel = f"b = 0. FRMSE: {FRMSE_df.FRMSE_0.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob_0.iloc[j]):.1f}",obslabel='AMS')
    plt.plot(eRP,RL_df.return_levels.iloc[j],"r",label = f"b = free. FRMSE: {FRMSE_df.FRMSE.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob.iloc[j]):.1f}")
    #plt.plot(eRP,RL_df.return_levels_5.iloc[j],"g",alpha = 0.5, label = "b = 5% sig")
    if "return_levels_bset" in RL_df.columns:
        plt.plot(eRP,RL_df.return_levels_bset.iloc[j],"y", label = f"b = mean. FRMSE: {FRMSE_df.FRMSE_bset.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob_bset.iloc[j]):.1f}")
    if "return_levels_bexp" in RL_df.columns:
        plt.plot(eRP,RL_df.return_levels_bexp.iloc[j],"m", label = f"b exp. FRMSE: {FRMSE_df.FRMSE_bexp.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob_bexp.iloc[j]):.1f}")
    
    plt.plot(eRP,RL_df6.return_levels.iloc[j],"g", label = f"b free, beta = 6. FRMSE: {FRMSE_df6.FRMSE.iloc[j]:.3f}. log: {np.log(liklihood_df6.mult_prob.iloc[j]):.1f}")

    plt.fill_between(eRP,liklihood_df.mins_0.iloc[j],liklihood_df.maxes_0.iloc[j],color = "b", alpha = 0.1)
    plt.fill_between(eRP,liklihood_df.mins.iloc[j],liklihood_df.maxes.iloc[j],color = "r", alpha = 0.1)
    
    plt.ylim(0,np.max(RL_df.return_levels.iloc[j])+5)
    plt.xlim(1,np.max(eRP)+2)
    
    plt.legend()
    if "df_parameters_bset" in locals():
        plt.title(f"station {j}: {FRMSE_df.station.iloc[j]}.b mean = {df_parameters_bset.b.iloc[0]:.3f}")
    else:
        plt.title(f"station {j}: {FRMSE_df.station.iloc[j]}.")
    plt.show()
    
    #temperature model plot
    file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[j]}"
    
    if 'code_str' in locals():
        G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
    else:
        G = pd.read_csv(f"{file_name}.csv")
        G['prec_time'] = pd.to_datetime(G['prec_time'])
        G.set_index('prec_time', inplace=True)
        
    ######################################################################
    #TENAX  AMS

    data = G 
    data = S.remove_incomplete_years(data, name_col)
    T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[j]}.nc"
    T_ERA = xr.load_dataarray(T_path)
    t_data = (T_ERA.squeeze()-273.15).to_dataframe()
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
    
    AMS = dict_AMS['60']
    
    
    df_arr_t_data = np.array(t_data[temp_name_col])
    df_dates_t_data = np.array(t_data.index)
    
    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
    

    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() 
    T = dict_ordinary["60"]["T"].to_numpy() 
    
    g_phat = [df_parameters.mu.iloc[j], df_parameters.sigma.iloc[j]]
    g_phat6 = [g_phats6.mu.iloc[j],g_phats6.sigma.iloc[j]]
    
    eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    _,_ =TNX_FIG_temp_model(T, g_phat, 4, eT, obscol='r',valcol='b',
                           obslabel = 'observations',
                           vallabel = 'beta = 4')
    
    pdf_values = gen_norm_pdf(eT, g_phat6[0], g_phat6[1], 6)
    plt.plot(eT, pdf_values, '-', color="g", label="beta = 6")
    plt.legend()
    
    plt.title(f"({df_parameters.latitude.iloc[j]:.2f},{df_parameters.longitude.iloc[j]:.2f}) \n Beta = 4: FRMSE = {all_temp_FRMSE.temp_FRMSE.iloc[j]:.2f}. FRMSE_20 = {all_temp_FRMSE.temp_FRMSE_upper_perc.iloc[j]:.2f} \n Beta = 6: FRMSE = {all_temp_FRMSE.temp_FRMSE6.iloc[j]:.2f}. FRMSE_20 = {all_temp_FRMSE.temp_FRMSE6_upper_perc.iloc[j]:.2f}")
    plt.show()




