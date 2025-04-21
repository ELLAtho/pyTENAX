# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 18:00:15 2025

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
station_chose = "18256"


# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
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



save_name = f"{drive}:/outputs/{country_save}\\return_levels.csv"

RL_df = pd.read_csv(save_name, dtype={'station': str})
nan_locs = RL_df.return_levels[RL_df.return_levels.isna()].index
replace_range = np.arange(0,len(RL_df))

RL_column_names = [col for col in RL_df.columns if "return_levels" in col]


for col in RL_column_names:
    nan_locs = RL_df[col][RL_df[col].isna()].index
    replace_range = np.arange(0,len(RL_df))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
    
        RL_df.loc[j, col] = np.fromstring(RL_df[col].iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        
nan_locs = RL_df.obs_AMS[RL_df.obs_AMS.isna()].index
replace_range = np.arange(0,len(RL_df))
for k in range(len(nan_locs)):
    replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
for j in replace_range:
    RL_df.loc[j, "obs_AMS"] = np.fromstring(RL_df.obs_AMS.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')



new_df.index = range(len(new_df))


###############################################################################

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = 0,
        min_ev_dur = 60,
        niter_smev = 1000, 
        beta = 6
    )


###############################################################################
# Just one station

T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station_chose}.csv")
P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station_chose}.csv")
times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{station_chose}.csv",parse_dates = ["oe_time"])
oe_df = pd.DataFrame({"year":times.oe_time.dt.year, "P": P, "T": T,})
AMS = oe_df.groupby(oe_df.year).P.max().rename({"P" : "AMS"})

start_time = times.iloc[0]
end_time = times.iloc[-1]

midyear = (start_time.dt.year + (end_time.dt.year - start_time.dt.year)/2).to_numpy()[0]

T1 = T[times.oe_time.dt.year <= midyear]
P1 = P[times.oe_time.dt.year <= midyear]
times1 = times[times.oe_time.dt.year <= midyear]
thr1 = np.quantile(P1,S.left_censoring[1])
n1 = len(T1)/(midyear - start_time.dt.year + 1)
AMS1 = pd.DataFrame(AMS[AMS.index <= midyear]).rename(columns = {"P" : "AMS"})


T2 = T[times.oe_time.dt.year > midyear]
P2 = P[times.oe_time.dt.year > midyear]
times2 = times[times.oe_time.dt.year > midyear]
thr2 = np.quantile(P2,S.left_censoring[1])
n2 = len(T2)/(end_time.dt.year - midyear)
AMS2 = pd.DataFrame(AMS[AMS.index > midyear]).rename(columns = {"P" : "AMS"})


g_phat1 = S.temperature_model(T1)
g_phat2 = S.temperature_model(T2)


F_phat1,_,_,_ = S.magnitude_model(P1, T1, thr1)
F_phat2,_,_,_ = S.magnitude_model(P2, T2, thr2)

S.alpha = 1

F_phat1_b0,_,_,_ = S.magnitude_model(P1, T1, thr1)
F_phat2_b0,_,_,_ = S.magnitude_model(P2, T2, thr2)


eT = np.arange(np.min(T),np.max(T)+4)
Ts = np.arange(np.min(T)- S.temp_delta, np.max(T)+ S.temp_delta, S.temp_res_monte_carlo)



TNX_FIG_temp_model(T1, g_phat1, 6, eT,obscol='b',valcol='b',
                       obslabel = f'observations {start_time.dt.year.to_numpy()[0]} - {int(midyear)}',
                       vallabel = 'temperature model g(T) first period')

TNX_FIG_temp_model(T2, g_phat2, 6, eT,obscol='r',valcol='r',
                       obslabel = f'observations {int(midyear+1)} - {end_time.dt.year.to_numpy()[0]}',
                       vallabel = 'temperature model g(T) second period')
plt.show()


RL1, _, _ = S.model_inversion(F_phat1_b0, g_phat1, n1, Ts)

RL2, _, _ = S.model_inversion(F_phat1_b0, g_phat2, n1, Ts) #calculated with the same F_phat and n

TNX_FIG_valid(AMS1, S.return_period, RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'first period',obslabel='Observed annual maxima')
TNX_FIG_valid(AMS2, S.return_period, RL2,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'predicted second period',obslabel='Observed annual maxima')

plt.show()

###############################################################################




























