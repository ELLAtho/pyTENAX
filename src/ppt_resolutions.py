# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 12:13:16 2025

@author: ellar
"""

from os.path import dirname, join
from os import getcwd
import sys
import pickle

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
from scipy.stats import chi2
from scipy import odr

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
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib.patches import Patch

drive = "D"

# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
# station_chose = "18256"
# station_chose = "12261"
station_chose = "19376"

# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


# country = 'UK' 
# ERA_country = 'UK'
# country_save = 'UK'
# code_str = 'UK_'
# minlat,minlon,maxlat,maxlon = 49, -9.0, 62, 3
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9

name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 


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

#getting info of correct size
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
df_parameters_exp = pd.read_csv(f"{drive}:/outputs/{country_save}/parameters_exp.csv", dtype={'station': str})

# for some reason in germany there is one less row...
    



if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg, dtype={'station': str})

    #dataframe with all values
    new_df = df_parameters[['station','latitude','longitude','b','kappa','lambda','a','thr','mu','sigma','n_events_per_yr']].copy()
    
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



res_savename = f"{drive}:/outputs/resolutions/{country_save}\\resolution_info.csv"
if res_savename not in glob.glob(f"{drive}:/outputs/resolutions/{country_save}/*"):
    print("not calculated the resolutions")
    starttime = [0]*len(val_info)
    yearly_non0_mins = [0]*len(val_info)
    unique_mins = [0]*len(val_info)
    GSDR_res = [0]*len(val_info)
    change_yr = [0]*len(val_info)
    n_res = [0]*len(val_info)
    
    for i in range(len(val_info)):
        
        starttime[i] = time.time()
        station = val_info.station.iloc[i]
        
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        
        if 'code_str' in locals():
            G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
        else:
            G = pd.read_csv(f"{file_name}.csv")
            G['prec_time'] = pd.to_datetime(G['prec_time'])
            G.set_index('prec_time', inplace=True)
        
        G[G.ppt == 0] = np.nan #replaces 0s with nan so the min value is the min nonzero value
        yearly_non0_mins[i] = G.groupby(G.index.year).min()
        yearly_non0_mins[i] = yearly_non0_mins[i][~np.isnan(yearly_non0_mins[i].ppt)]
        unique_mins[i] = np.unique(yearly_non0_mins[i])
        unique_mins[i] = unique_mins[i][~np.isnan(unique_mins[i])]
        GSDR_res[i] = data_meta.resolution
        change_yr[i] = [
            yearly_non0_mins[i].index[j]
            for j in range(1, len(yearly_non0_mins[i]))
            if yearly_non0_mins[i]["ppt"].iloc[j] != yearly_non0_mins[i]["ppt"].iloc[j-1]
        ]
        n_res[i] = len(unique_mins[i])
        
        if i%50 == 0:
            time_taken = (time.time()-starttime[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
    
    resolution_df = pd.DataFrame({
        "station" : val_info.station,
        "GSDR_res" : GSDR_res,
        "n_mins" : n_res,
        
        })  
    
    resolution_df.to_csv(f"{drive}:/outputs/resolutions/{country_save}/resolution_info.csv", index = False)
      
    
    yearly_non0_mins_labelled = dict(zip(val_info.station, [yearly_non0_mins[j].ppt.to_numpy() for j in range(len(yearly_non0_mins))]))
    change_yr_labelled = dict(zip(val_info.station,change_yr))
    unique_mins_labelled = dict(zip(val_info.station,unique_mins))
    
    with open(f"{drive}:/outputs/resolutions/{country_save}/non0_mins.pkl", 'wb') as f:
        pickle.dump(yearly_non0_mins_labelled, f)
        
    with open(f"{drive}:/outputs/resolutions/{country_save}/years_when_change_res.pkl", 'wb') as f:
        pickle.dump(change_yr_labelled, f)
    
    with open(f"{drive}:/outputs/resolutions/{country_save}/unique_mins.pkl", 'wb') as f:
        pickle.dump(unique_mins_labelled, f)
        
else:
    print("reading resolution data")
    resolution_df = pd.read_csv(f"{drive}:/outputs/resolutions/{country_save}/resolution_info.csv")
    
    with open(f"{drive}:/outputs/resolutions/{country_save}/non0_mins.pkl", 'rb') as f:
        yearly_non0_mins_labelled = pickle.load(f)
        
    with open(f"{drive}:/outputs/resolutions/{country_save}/years_when_change_res.pkl", 'rb') as f:
        change_yr_labelled = pickle.load(f)
    
    with open(f"{drive}:/outputs/resolutions/{country_save}/unique_mins.pkl", 'rb') as f:
        unique_mins_labelled = pickle.load(f)
      
        
        
        