# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:12:16 2025

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



drive = 'D'
alpha_set = 0.05


country = 'Germany' 
ERA_country = 'Germany'
country_save = 'Germany'
code_str = 'DE_'
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 

#READ IN META INFO FOR COUNTRY
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


files = glob.glob(drive+':/'+country+'/*') #list of files in country folder
files_sel = [files[i] for i in val_info.index]


## READ IN FILES
save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 
TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters

df_parameters_0 = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/parameters.csv", dtype={'station': str})

# for some reason in germany there is one less row...
    



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

missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    new_df = new_df.drop(missing_rows.index)
else:
    pass


df_generated_parameters = pd.read_csv(drive + ':/outputs/'+country_save+'\\synth_generated_parameters.csv')

#without L-moments...


print(f"{len(new_df)} stations")

variables = df_generated_parameters.columns[-5:-1]
for variable in variables:
    v = np.std(new_df[variable])
    mu_v = np.mean(df_generated_parameters[variable])
    sigma_v = np.std(df_generated_parameters[variable])

    H = np.abs((v - mu_v)/sigma_v)
    if H < 1:
        print(f"{country}. {variable} definitely homogeneous. H = {H:.2f}")
    elif H > 1 and H < 2:
        print(f"{country}. {variable} maybe heterogeneous. H = {H:.2f}")
    else:
        print(f"{country}. {variable} definitely heterogeneous. H = {H:.2f}")


#cutout


minlat_cut , minlon_cut , maxlat_cut , maxlon_cut = 48, 8, 50, 10



s = 3 


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


ax1.plot([minlon, minlon],[minlat, maxlat],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([maxlon, maxlon],[maxlat, minlat],  'r', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([minlon, maxlon],[minlat, minlat],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([maxlon, minlon],[maxlat, maxlat],  'r', linewidth=2, transform=ccrs.PlateCarree(),label = 'Germany')


ax1.plot([minlon_cut , minlon_cut ],[minlat_cut , maxlat_cut ],  'b', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([maxlon_cut , maxlon_cut ],[maxlat_cut , minlat_cut ],  'b', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([minlon_cut , maxlon_cut ],[minlat_cut , minlat_cut ],  'b', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([maxlon_cut , minlon_cut ],[maxlat_cut , maxlat_cut ],  'b', linewidth=2, transform=ccrs.PlateCarree(),label = 'Germany cut')

if df_parameters.b.min() == 0:
    norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
else:
    norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())



plt.scatter(new_df.longitude,new_df.latitude,
            c = new_df.b,
            s = s,
            cmap = 'seismic',
            norm = norm)


plt.legend()
plt.show()



new_df_cut = new_df[new_df['latitude']>=minlat_cut] #filter station locations to within ERA bounds
new_df_cut = new_df_cut[new_df_cut['latitude']<=maxlat_cut]
new_df_cut = new_df_cut[new_df_cut['longitude']>=minlon_cut]
new_df_cut = new_df_cut[new_df_cut['longitude']<=maxlon_cut]

print("###############################################")
print(f"{len(new_df_cut)} stations")
for variable in variables:
    v = np.std(new_df_cut[variable])
    mu_v = np.mean(df_generated_parameters[variable])
    sigma_v = np.std(df_generated_parameters[variable])

    H = np.abs((v - mu_v)/sigma_v)
    if H < 1:
        print(f"{country}. {variable} definitely homogeneous. H = {H:.2f}")
    elif H > 1 and H < 2:
        print(f"{country}. {variable} maybe heterogeneous. H = {H:.2f}")
    else:
        print(f"{country}. {variable} definitely heterogeneous. H = {H:.2f}")













