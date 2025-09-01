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
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

drive = 'D'


###############################################################################
# %% reading in germany data

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

# %% calculating parameters


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


# %% basic TENAX graphs

qs = [.85,.95,.99,.999]

fig = plt.figure(figsize = (6,6))
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


fig = plt.figure(figsize = (6,6))
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

fig = plt.figure(figsize = (6,6))
TNX_FIG_valid(AMS,RP,RL,smev_RL=[],RL_unc=0,smev_RL_unc=0,TENAXcol='b',obscol_shape = 'g+',smev_colshape = '--r',TENAXlabel = 'The TENAX model',obslabel='Observed annual maxima',smevlabel = 'The SMEV model',alpha = 0.2,xlimits = [1,200],ylimits = [0,50])
plt.plot(RP,RL0,color = "b", alpha = 0.3)
plt.xlim(1,23)
plt.ylim(0,40)
plt.xticks(fontsize = fontsize-2)
plt.yticks(fontsize = fontsize-2)
plt.xlabel('return period (years)', fontsize = fontsize)
plt.ylabel("hourly precipitation (mm)", fontsize = fontsize)
plt.legend(fontsize = fontsize,frameon=False,loc = "upper left")
plt.show()

############################################################################
# %% Reading Japan data


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



station_0 = "12261"
# station_0 = "19376"


T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station_0}.csv")
P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station_0}.csv")
eT = np.arange(np.min(T),np.max(T)+4,1)

kde  = gaussian_kde(T) #use kernel density to get probability




T_min = np.min(T)
T_max = np.max(T)
Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)




g_phat = [new_df[new_df.station == station_0].mu.to_numpy(),new_df[new_df.station == station_0].sigma.to_numpy()]

F_phat = [new_df[new_df.station == station_0].kappa.to_numpy(),
          new_df[new_df.station == station_0].b.to_numpy(),
          new_df[new_df.station == station_0]["lambda"].to_numpy(),
          new_df[new_df.station == station_0].a.to_numpy()]

F_phat_0 = [df_parameters_0[df_parameters_0.station == station_0].kappa.to_numpy(),
          df_parameters_0[df_parameters_0.station == station_0].b.to_numpy(),
          df_parameters_0[df_parameters_0.station == station_0]["lambda"].to_numpy(),
          df_parameters_0[df_parameters_0.station == station_0].a.to_numpy()]

thr = new_df[new_df.station == station_0].thr
n = new_df[new_df.station == station_0].n_events_per_yr

AMS = RL_df[new_df.station == station_0].obs_AMS.to_numpy()[0]

plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))

eRP = 1/(1-plot_pos)

S.return_period = eRP

pdf_values = kde(Ts)
df = np.vstack([pdf_values, Ts + 2]) # shifted by 2 degrees

T_mc = randdf(S.n_monte_carlo, df, 'pdf').T              


wbl_phat_0 = np.column_stack((
                            F_phat_0[2] * np.exp(F_phat_0[3] * T_mc),
                            F_phat_0[0] + F_phat_0[1] * T_mc
                            ))



vguess = 10 ** np.arange(np.log10(0.05), np.log10(5e2), 0.05)
RL_2 = SMEV_Mc_inversion(wbl_phat_0, n, S.return_period, vguess, method_root_scalar="brentq")



RL = RL_df[new_df.station == station_0].return_levels_kernal.to_numpy()[0]
RL_0 = RL_df[new_df.station == station_0].return_levels_kernal_0.to_numpy()[0]
RL_exp = RL_df[new_df.station == station_0].return_levels_kernal_exp.to_numpy()[0]

fontsize = 15



# %% plotting return levels

fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(1,1,1)

plt.plot(eRP,AMS,"k+",label = "annual maxima") #plot observed return levels
plt.plot(eRP,RL_0,color = "b", label = f"b = 0")
plt.plot(eRP,RL,"r--",label = "b = fitted")  #plot TENAX return levels

plt.ylim(0,45)
plt.xscale('log')
plt.xlabel('return period (years)', fontsize = fontsize)
plt.ylabel("hourly precipitation (mm)", fontsize = fontsize)
plt.yticks(fontsize = fontsize-2)
plt.xticks([1,3,10,30],fontsize = fontsize-2)
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.legend(fontsize = fontsize, frameon = False)
plt.show()

###############################################################################
# %% Map of b

countries = ["germany","Japan","UK","US"]
country_saves = ["germany","Japan","UK","US_main"]
code_strs = ["DP_","JP_","UK_","US_"]
min_startdates = [dt.datetime(1900,1,1),dt.datetime(1900,1,1),dt.datetime(1950,1,1),dt.datetime(1950,1,1)] #this is for if havent read all ERA5 data yet

lons_lats = [[47, 3, 55, 15],[24, 122.9, 45.6, 145.8],[49, -9.0, 62, 3] ,[24, -125, 56, -66]]


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
min_yrs = 10
# censor_thr = 0.9



df_parameters = [0]*4
TENAX_use = [0]*4
df_parameters_0 = [0]*4
df_parameters_exp = [0]*4
df_parameters_neg = [0]*4
new_df = [0]*4
df_generated_parameters = [0]*4
df_generated_parameters_0 = [0]*4
df_generated_parameters_exp = [0]*4


df_generated_parameters_one2one = [0]*4
df_generated_parameters_0_one2one = [0]*4
df_generated_parameters_exp_one2one = [0]*4


comb_df_gen_params = [0]*4
comb_df_gen_params_0 = [0]*4
comb_df_gen_params_exp = [0]*4


info = [0]*4

for country_i in range(4):
    country_save = country_saves[country_i]
    country = countries[country_i]
    code_str = code_strs[country_i]
    
    minlat, minlon, maxlat, maxlon = lons_lats[country_i]
    min_startdate = min_startdates[country_i]
    
    
    save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
    df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'
    df_gen_savename = drive + ':/outputs/'+country_save+'\\synth_generated_parameters.csv'
    
    df_parameters[country_i] = pd.read_csv(df_savename, dtype={'station': str}) 
    TENAX_use[country_i] = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters
    
    df_parameters_0[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/parameters.csv", dtype={'station': str})
    df_parameters_exp[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/parameters_exp.csv", dtype={'station': str})
    df_generated_parameters[country_i] = pd.read_csv(df_gen_savename)
    df_generated_parameters_0[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/synth_generated_parameters.csv")
    df_generated_parameters_exp[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/synth_generated_parameters_exp.csv")
    
    df_generated_parameters_one2one[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/synth_generated_parameters_one2one.csv")
    df_generated_parameters_0_one2one[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/synth_generated_parameters_one2one.csv")
    df_generated_parameters_exp_one2one[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/synth_generated_parameters_one2one_exp.csv")
    
    
    # puts the redos all together into one dataframe
    comb_df_gen_params[country_i] = pd.DataFrame()
    comb_df_gen_params_0[country_i] = pd.DataFrame()
    comb_df_gen_params_exp[country_i] = pd.DataFrame()

    params = ["b", "kappa", "lambda", "a"]
    for param in params:
        comb_df_gen_params[country_i][param] = pd.concat([df_generated_parameters_one2one[country_i][f"{param}{j}"].copy().dropna() for j in range(5)])
        comb_df_gen_params_0[country_i][param] = pd.concat([df_generated_parameters_0_one2one[country_i][f"{param}{j}"].copy().dropna() for j in range(5)])
        comb_df_gen_params_exp[country_i][param] = pd.concat([df_generated_parameters_exp_one2one[country_i][f"{param}{j}"].copy().dropna() for j in range(5)])
        
    comb_df_gen_params[country_i].reset_index(inplace = True)
    comb_df_gen_params_0[country_i].reset_index(inplace = True)
    comb_df_gen_params_exp[country_i].reset_index(inplace = True)
    
    
    info1 = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
    
    info1.startdate = pd.to_datetime(info1.startdate)
    info1.enddate = pd.to_datetime(info1.enddate)
    val_info = info1[info1['cleaned_years']>=min_yrs] #filter out stations that are less than min
    val_info = val_info[val_info['startdate']>=min_startdate]
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
    
    info[country_i] = val_info.reset_index()
    
    if np.size(glob.glob(save_path_neg)) != 0:
        df_parameters_neg[country_i] = pd.read_csv(save_path_neg, dtype={'station': str})
    
        #dataframe with all values
        new_df[country_i] = df_parameters[country_i][['station','latitude','longitude','b','kappa','lambda','a','mu','sigma','thr','n_events_per_yr']].copy()
        
        mask = new_df[country_i]['b'] == 0
        
        new_df[country_i].loc[mask, 'b'] = df_parameters_neg[country_i]['b2'].to_numpy()
        new_df[country_i].loc[mask, 'kappa'] = df_parameters_neg[country_i]['kappa2'].to_numpy()
        new_df[country_i].loc[mask, 'lambda'] = df_parameters_neg[country_i]['lambda2'].to_numpy()
        new_df[country_i].loc[mask, 'a'] = df_parameters_neg[country_i]['a2'].to_numpy()
    
    else:
        new_df[country_i] = df_parameters[country_i].copy()
    
    missing_rows = pd.merge(df_parameters[country_i].station, df_parameters_0[country_i].station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    if len(missing_rows) != 0:
        print("miss-match, dropping")
        df_parameters[country_i] = df_parameters[country_i].drop(missing_rows.index)
        df_parameters[country_i] = df_parameters[country_i].reindex(index = range(len(df_parameters[country_i])))
        new_df[country_i] = new_df[country_i].drop(missing_rows.index)
        new_df[country_i] = new_df[country_i].reindex(index = range(len(new_df[country_i])))
    else:
        pass


s = 1
sig_mod = 8
norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)

legend_elements = [
    plt.Line2D([0], [0], marker = "o",markersize = np.sqrt(s*sig_mod), linestyle = " ", color='k', label=r'$b$ significantly'+' \ndifferent from 0'),
    plt.Line2D([0], [0], marker = "o",markersize  = np.sqrt(s), linestyle = " ", color='k', label=r'not significant'),
]


proj = ccrs.PlateCarree()
  

#loop to go through the countries
for country_i in range(4): 
    
    if country_i == 1: 
        fig = plt.figure(figsize = (8,8))
    else:
        fig = plt.figure(figsize = (4,4))
        
    ax = fig.add_subplot(1,1,1, projection=proj)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    sc = ax.scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b==0],
        df_parameters[country_i].latitude[df_parameters[country_i].b==0],
        c=new_df[country_i].b[df_parameters[country_i].b==0],
        s = s,
        cmap='seismic',  
        norm=norm,
    )

    sc = ax.scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b!=0],
        df_parameters[country_i].latitude[df_parameters[country_i].b!=0],
        c=new_df[country_i].b[df_parameters[country_i].b!=0],
        s = s*sig_mod,
        # edgecolors = "grey",
        cmap='seismic',  
        norm=norm,
    )
    
    if country_i == 1:        
        ax.legend(handles = legend_elements,fontsize = fontsize)
    else: 
        pass
    
    
    # Set x and y ticks
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    if country_i == 0:
        gl.xlocator = FixedLocator([6,9,12,15])
        gl.ylocator = FixedLocator([48,50,52,54])
    elif country_i == 2:
        gl.xlocator = FixedLocator([-7,-4,-1,2])
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    gl.xformatter = LongitudeFormatter(degree_symbol="° ")
    gl.yformatter = LatitudeFormatter(degree_symbol="° ")
    
    if country_i == 1:
        cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    
    
        cb.set_label(r'$b$ [°C$^{-1}$]', fontsize=fontsize)  
        cb.ax.tick_params(labelsize=fontsize)
    else:
        pass

    plt.show()


# %% Synthetic return levels


gap1 = 0.3 #gaps between the bar plots
gap2 = 1.2
fontsize = 20

colors = ['#377eb8', '#ff7f00', '#4daf4a']*4
    
labels = ["(a)","(b)","(c)"]

ret_lvls = ["10","100"]
# plot the fractionals all together in a different layout
fig = plt.figure(figsize=(12,12))
for i in range(3):
    ax = fig.add_subplot(2,3,i+1)
    
    #plot boxes
    boxplot_list = [synth_RL[years][f"{bstyle}_{ret_lvls[ret_n]}"].dropna()/RL_true[years][ret_n*3] for ret_n in range(2) for bstyle in ["free","set","b0"] for years in [i*3,i*3+2,i*3+1]]
    # positions = np.concat([np.arange(0,4.5,0.5),np.arange(5,9.5,0.5)]) 
    
    base_positions = np.concat([np.arange(0+pos*(1.5+gap1),1.5+pos*(1.5+gap1),0.5) for pos in range(3)])
    positions = np.concat([base_positions,base_positions+(1.5+2*(1.5+gap1))+gap2])
                          
    
    box_plot = ax.boxplot(boxplot_list,
                          positions = positions,
                          showmeans = True, meanline = True, patch_artist=True,
                          meanprops=dict(marker=None, linestyle=':', linewidth=1,color = 'k'),
                          sym = "",
                          whis = [5,95])
    
    strip_list = [box[(box<np.quantile(box,0.05)) | (box>np.quantile(box,0.95))] for box in boxplot_list]
    
    for l in range(len(strip_list)):
        strip_list[l][strip_list[l]>2] = 2
        strip_list[l][strip_list[l]<1/2] = 1/2
    
    strip_data = pd.DataFrame({
        
        "x": np.concatenate([[pos]*len(vals) for pos, vals in zip(positions, strip_list)]),
        "y": np.concatenate(strip_list)
        })
    
    
    seaborn.stripplot(x="x", y="y", data=strip_data, color='black',alpha = 0.5,size = 3,native_scale=True)
    
    
    ax.grid(axis = "y")
    
    
    alpha = [1,0.6,0.3]
    for n_patch in range(len(box_plot['boxes'])):
        patch = box_plot['boxes'][n_patch]
        patch.set_facecolor(colors[int(np.trunc(n_patch/3))])
        patch.set_alpha(alpha[int(n_patch%3)])
                
    for median_line in box_plot["medians"]:
        median_line.set_color('k')
    
    ax.set_xticks([positions[4],positions[13]],ret_lvls,fontsize = fontsize)
    
    ax.set_yscale("log")
    ax.set_ylim(1/2.1,2.1)
    custom_ticks = [1/2,1/1.5,1/1.1, 1, 1.1,1.5,2]
    custom_ticklabels = ["≤ 1/2","1/1.5","1/1.1", "1", "1.1","1.5","≥ 2"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels,fontsize = fontsize)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    ax.text(-0.03, 1.06, labels[i], transform=ax.transAxes,
      fontsize=fontsize+2, va='top', ha='right')
    
    ax.set_title(f"$b$ = {uses[i*3].b[0]}"+r" °C$^{-1}$",fontsize = fontsize+2)
    ax.set_xlabel("Return period (years)",fontsize = fontsize)
    ax.set_ylabel("Bias in return levels",fontsize = fontsize)
    
    if i == 0:
        legend_elements = [
        plt.Line2D([0], [0], color='k', label='median'),
        plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
        ]
        plt.legend(handles=legend_elements,fontsize = fontsize)

    
    if i == 1:
        legend_elements = [
            Patch(facecolor="k", label='30 years'),
            Patch(facecolor="k", alpha = 0.6, label='20 years'),
            Patch(facecolor="k", alpha = 0.3, label='10 years'),
            ]    
        
        plt.legend(handles=legend_elements,fontsize = fontsize)
    else:
        pass

legend_elements = [
    Patch(facecolor=colors[0], label='Free'),  # default matplotlib colors
    Patch(facecolor=colors[1], label='Set'),
    Patch(facecolor=colors[2], label=r'$b$ = 0'),
]

plt.legend(handles=legend_elements,fontsize = fontsize)

plt.tight_layout()
plt.show()

# %% Temperature

df_JP = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape.csv",dtype = {0:str})
df_parameters_JP = pd.read_csv(drive + ':/outputs/'+country_save+'\\parameters.csv', dtype={'station': str}) 


eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv",dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()

interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df_JP)
for i in np.arange(0,len(df_JP)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df_JP.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        

# %% define the shapes

alpha = 0.4

peaks_df_JP = pd.read_csv(f"{drive}:/outputs/Japan\\peaks.csv")
skew_df_JP = pd.read_csv(f"{drive}:/outputs/Japan\\temp_skew.csv", dtype={"station":str})
df_parameters_JP = df_parameters[1]

mask = ((skew_df_JP.skewness < 0) &
    (peaks_df_JP.n_peaks01 == 1) &
    (df_parameters_JP.latitude > 36) &
    (df_parameters_JP.latitude > 31)) #1 peak, neg skew, north half


interp_y_region1 = np.array(interp_y)[mask]
aves_region1 = np.array(aves)[mask]
loc_region1 = df_parameters_JP[mask]

points = np.column_stack((loc_region1.longitude, loc_region1.latitude))

shape1 = alphashape.alphashape(points, alpha)


mask = ((skew_df_JP.skewness < 0) &
    (peaks_df_JP.n_peaks01 == 1) &
    (df_parameters_JP.latitude <= 36) &
    (df_parameters_JP.latitude > 31)) #1 peak, neg skew, south half


interp_y_region2 = np.array(interp_y)[mask]
aves_region2 = np.array(aves)[mask]
loc_region2 = df_parameters_JP[mask]

points = np.column_stack((loc_region2.longitude, loc_region2.latitude))

shape2 = alphashape.alphashape(points, alpha)


mask = ((skew_df_JP.skewness > 0) &
    (peaks_df_JP.n_peaks01 != 1) &
    (df_parameters_JP.latitude > 36) &
    (df_parameters_JP.latitude > 31)) #this is positive skew and 2 peaks, north half


interp_y_region3 = np.array(interp_y)[mask]
aves_region3 = np.array(aves)[mask]
loc_region3 = df_parameters_JP[mask]

points = np.column_stack((loc_region3.longitude, loc_region3.latitude))

shape3 = alphashape.alphashape(points, alpha)


mask = ((skew_df_JP.skewness > 0) &
    (peaks_df_JP.n_peaks01 != 1) &
    (df_parameters_JP.latitude <= 36) &
    (df_parameters_JP.latitude > 31)) #this is positive skew and 2 peaks, south half


interp_y_region4 = np.array(interp_y)[mask]
aves_region4 = np.array(aves)[mask]
loc_region4 = df_parameters_JP[mask]

points = np.column_stack((loc_region4.longitude, loc_region4.latitude))

shape4 = alphashape.alphashape(points, alpha)


# %% plot map with shapes

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1, projection=proj)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax.scatter(
    df_parameters.longitude[df_parameters.b==0],
    df_parameters.latitude[df_parameters.b==0],
    c=new_df.b[df_parameters.b==0],
    s = s,
    cmap='seismic',  
    norm=norm,
)

sc = ax.scatter(
    df_parameters.longitude[df_parameters.b!=0],
    df_parameters.latitude[df_parameters.b!=0],
    c=new_df.b[df_parameters.b!=0],
    s = s*sig_mod,
    # edgecolors = "grey",
    cmap='seismic',  
    norm=norm,
)

shape_list = [shape1, shape2, shape3, shape4]
colors = ["r","b","w","k--"]


for i, shape in enumerate(shape_list):
    if isinstance(shape, Polygon):
        x, y = shape.exterior.xy
        plt.plot(x, y, colors[i], linewidth=2, label='Alpha Shape', transform=ccrs.PlateCarree())
    else:
        for polygon in shape.geoms:
            x, y = polygon.exterior.xy
            plt.plot(x, y, colors[i], linewidth=2, label='Alpha Shape', transform=ccrs.PlateCarree())





ax.legend(handles = legend_elements,fontsize = fontsize)


# Set x and y ticks
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')


gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))


cb.set_label(r'$b$ [°C$^{-1}$]', fontsize=fontsize)  
cb.ax.tick_params(labelsize=fontsize)

plt.show()

















