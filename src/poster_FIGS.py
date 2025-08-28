# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 09:58:40 2025

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


###############################################################################
# 3a and 3b
country = "germany"
country_save = "germany"
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15

info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
station = "03660"
RL_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\return_levels.csv", dtype={'station': str})
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


T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{station}.csv",parse_dates = ["oe_time"])


S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.0,
        min_ev_dur = 60,
        beta = 4
    )


g_phat = S.temperature_model(T)
thr = np.quantile(P,S.left_censoring[1])
n = len(T)/info[info.station == station].cleaned_years


F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)

S.alpha = 1
F_phat0, loglik, _, _ = S.magnitude_model(P, T, thr)


eT = np.arange(np.min(T),np.max(T)+4,1)


fontsize = 14

qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [eT[0],eT[-1]])

# plot b=0 percentiles
percentile_lines = inverse_magnitude_model(F_phat0,eT,qs)

#first one outside loop so can be in legend
n=0
plt.plot(eT,percentile_lines[n],label = 'b = 0',color = "b", alpha = 0.3)
n=1
while n<np.size(qs):
    plt.plot(eT,percentile_lines[n],color = "b", alpha = 0.3) #,label = str(qs[n]),
    
    n=n+1



plt.ylabel("Hourly precipitation (mm)",fontsize = fontsize)
plt.xlabel("T (°C)",fontsize = fontsize)
plt.xticks(fontsize = fontsize-2)
plt.yticks(fontsize = fontsize-2)
plt.legend(fontsize = fontsize,frameon=False,loc = "upper left")
plt.xlim(-12,29)
# plt.title("The magnitude model",fontsize = fontsize)
plt.show()

TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
plt.xlabel("T (°C)",fontsize = fontsize)
plt.ylabel("pdf",fontsize = fontsize)
plt.xticks(fontsize = fontsize-2)
plt.yticks(fontsize = fontsize-2)
plt.legend(fontsize = fontsize,frameon=False,loc = "upper left")
plt.xlim(-12,29)
plt.ylim(0,0.07)
# plt.title("The temperature model",fontsize = fontsize)
plt.show()



AMS = RL_df[new_df.station == station].obs_AMS.to_numpy()[0]
RL = RL_df[new_df.station == station].return_levels.to_numpy()[0]
RL0 = RL_df[new_df.station == station].return_levels_b0.to_numpy()[0]

plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))

RP = 1/(1-plot_pos)

TNX_FIG_valid(AMS,RP,RL,smev_RL=[],RL_unc=0,smev_RL_unc=0,TENAXcol='b',obscol_shape = 'g+',smev_colshape = '--r',TENAXlabel = 'The TENAX model',obslabel='Observed annual maxima',smevlabel = 'The SMEV model',alpha = 0.2,xlimits = [1,200],ylimits = [0,50])
plt.plot(RP,RL0,color = "b", alpha = 0.3)
plt.xlim(1,23)
plt.ylim(0,40)
plt.xticks(fontsize = fontsize-2)
plt.yticks(fontsize = fontsize-2)
plt.legend(fontsize = fontsize,frameon=False,loc = "upper left")
plt.show()




