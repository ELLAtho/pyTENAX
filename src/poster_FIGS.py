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

from scipy.stats import norm
from scipy.optimize import minimize
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
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

import seaborn


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




# %% basic TENAX graphs

fontsize = 17

qs = [.85,.95,.99,.999]

fig = plt.figure(figsize = (5,5))


xlimits = [eT[0],eT[-1]]

percentile_lines = inverse_magnitude_model(F_phat,eT,qs)
plt.scatter(T,P,s=1,color="r",label = 'observations')

#first one outside loop so can be in legend
n=0
plt.plot(eT,percentile_lines[n],label = 'Magnitude model',color = "b")
plt.text(eT[-1], percentile_lines[n][-1], str(qs[n]*100)+'th', ha='left', va='center',fontsize = fontsize-2)
n=1
while n<np.size(qs):
    plt.plot(eT,percentile_lines[n],color = "b") #,label = str(qs[n]),
    plt.text(eT[-1], percentile_lines[n][-1], str(qs[n]*100)+'th', ha='left', va='center', fontsize = fontsize-2)
    n=n+1

plt.plot(eT,[thr]*np.size(eT),'--',alpha = 0.5,color = 'k',label = 'Left censoring') #plot threshold

plt.legend()
plt.yscale('log')
plt.ylim(0.1,1000)
plt.xlim(xlimits[0],xlimits[1])
plt.xlabel('T [°C]')


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
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.legend(fontsize = fontsize,frameon=False,loc = "upper left")
plt.xlim(-12,29)
# plt.title("The magnitude model",fontsize = fontsize)
plt.show()


fig = plt.figure(figsize = (5,5))
TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
plt.xlabel("T (°C)",fontsize = fontsize)
plt.ylabel("pdf",fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
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

fig = plt.figure(figsize = (5,5))
TNX_FIG_valid(AMS,RP,RL,smev_RL=[],RL_unc=0,smev_RL_unc=0,TENAXcol='b',obscol_shape = 'g+',smev_colshape = '--r',TENAXlabel = 'The TENAX model',obslabel='Annual maxima',smevlabel = 'The SMEV model',alpha = 0.2,xlimits = [1,200],ylimits = [0,50])
plt.plot(RP,RL0,color = "b", alpha = 0.3)
plt.xlim(1,23)
plt.ylim(0,40)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
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
plt.plot(eRP,RL_0,color = "b", label = "b = 0")
plt.plot(eRP,RL,"r--",label = f"b = {F_phat[1][0]:.3f}")  #plot TENAX return levels

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
colnorm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)


country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30
region_lats = [minlat,37.5,maxlat]
region_lons = [minlon,-116,-105,-90,maxlon]

lon_lims = [truncate_neg(np.min(df_parameters[3].longitude),2.5),np.ceil(np.max(df_parameters[3].longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters[3].latitude),2.5),np.ceil(np.max(df_parameters[3].latitude/2.5))*2.5]



n_lat = len(region_lats)-1
n_lon = len(region_lons)-1

# show regions





proj = ccrs.PlateCarree()
  
sizes = [(3,3),(4,4),(4,4),(8,5)]
#loop to go through the countries
for country_i in range(4): 
    
    fig = plt.figure(figsize = sizes[country_i])
    
    ax = fig.add_subplot(1,1,1, projection=proj)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    sc = ax.scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b==0],
        df_parameters[country_i].latitude[df_parameters[country_i].b==0],
        c=new_df[country_i].b[df_parameters[country_i].b==0],
        s = s,
        cmap='seismic',  
        norm=colnorm,
    )

    sc = ax.scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b!=0],
        df_parameters[country_i].latitude[df_parameters[country_i].b!=0],
        c=new_df[country_i].b[df_parameters[country_i].b!=0],
        s = s*sig_mod,
        # edgecolors = "grey",
        cmap='seismic',  
        norm=colnorm,
    )
    
    # plot the grid for temps
    if country_i == 3:
        for lat_i in range(n_lat-1):
            ax.plot([minlon-3,maxlon+3],[region_lats[lat_i+1],region_lats[lat_i+1]],  'r--', linewidth=1.5, transform=ccrs.PlateCarree())

        for lon_i in range(n_lon-1):
            ax.plot([region_lons[lon_i+1],region_lons[lon_i+1]],[minlat-3,maxlat+3],  'r--', linewidth=1.5, transform=ccrs.PlateCarree())
            
        plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
        plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

    else:
        pass
    
    # Set x and y ticks
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    if country_i == 0:
        gl.xlocator = FixedLocator([6,9,12,15])
        gl.ylocator = FixedLocator([48,50,52,54])
    elif country_i == 2:
        gl.xlocator = FixedLocator([-7,-4,-1,2])
    elif country_i == 3:
        gl.xlocator = FixedLocator([-120,-100,-80])
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    gl.xformatter = LongitudeFormatter(degree_symbol="° ")
    gl.yformatter = LatitudeFormatter(degree_symbol="° ")
    
    if country_i == 3:
        cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    
    
        cb.set_label(r'$b$ [°C$^{-1}$]', fontsize=fontsize)  
        cb.ax.tick_params(labelsize=fontsize)
    else:
        pass

    plt.show()


# %% Synthetic return levels

synth_files = [f"D:/outputs/synthetic\\RL_specific{num}.csv" for num in np.arange(1,10)]
#glob.glob("D:/outputs/synthetic\\RL_specific*")
use_files = [f"D:/outputs/synthetic\\parameters_set{num}.csv" for num in np.arange(1,10)]
#glob.glob("D:/outputs/synthetic\\parameters_set*")


synth_RL = [pd.read_csv(file) for file in synth_files]
uses = [pd.read_csv(file) for file in use_files]

S = TENAX(
        return_period = [10,20,50,100],  
        durations = [60, 180],
        left_censoring = [0, 0.90],
        alpha = 0,
        n_monte_carlo = 20000, # total number of events (on average)
        
    )

RL_true = []
RL_true_exp = []
for i in range(len(uses)):
    F_phat_typical = [uses[i].kappa[0],uses[i].b[0],uses[i]["lambda"][0],uses[i].a[0]]
    g_phat_typical = [uses[i].mu[0],uses[i].sigma[0]]
    
    Ts = np.arange(g_phat_typical[0]-2*g_phat_typical[1] - S.temp_delta, g_phat_typical[0]+2*g_phat_typical[1] + S.temp_delta, S.temp_res_monte_carlo)
    
    RL_typical_exp, _, _ = S.model_inversion(F_phat_typical, g_phat_typical, uses[i].n, Ts,b_exp = True)
    RL_typical, _, _ = S.model_inversion(F_phat_typical, g_phat_typical, uses[i].n, Ts)
    RL_true.append(RL_typical)
    RL_true_exp.append(RL_typical_exp)


# %% plot Synthetic return levels

gap1 = 0.3 #gaps between the bar plots
gap2 = 1.2
fontsize = 20

colors = ['#377eb8', '#ff7f00', '#4daf4a']*4
    
labels = ["(a)","(b)","(c)"]

ret_lvls = ["10","100"]
# plot the fractionals all together in a different layout
fig = plt.figure(figsize=(13,8))
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
    ax.set_ylim(1/1.6,1.6)
    custom_ticks = [1/1.5,1/1.1, 1, 1.1,1.5]
    custom_ticklabels = ["1/1.5","1/1.1", "1", "1.1","1.5"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels,fontsize = fontsize)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    
    ax.set_title(f"$b$ = {uses[i*3].b[0]}"+r" °C$^{-1}$",fontsize = fontsize+6)
    if i == 0:    
        ax.set_ylabel("Bias in return levels",fontsize = fontsize+4)
        ax.set_xlabel("Return period (years)",fontsize = fontsize+4)
    

plt.tight_layout()
plt.show()

# just plotting the legends
plt.plot([7,8],[7,8])

legend_elements = [
    Patch(facecolor=colors[0], label='Free'),  # default matplotlib colors
    Patch(facecolor=colors[1], label='Set'),
    Patch(facecolor=colors[2], label=r'$b$ = 0'),
]


plt.legend(handles=legend_elements,fontsize = fontsize)
plt.show()

plt.plot([7,8],[7,8])
legend_elements = [
    Patch(facecolor="k", label='30 years'),
    Patch(facecolor="k", alpha = 0.6, label='20 years'),
    Patch(facecolor="k", alpha = 0.3, label='10 years'),
    ]    

plt.legend(handles=legend_elements,fontsize = fontsize)
plt.show()

plt.plot([7,8],[7,8])
legend_elements = [
plt.Line2D([0], [0], color='k', label='median'),
plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
]
plt.legend(handles=legend_elements,fontsize = fontsize)
plt.show()



# %% US temperature split, reading and interping


country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30
region_lats = [minlat,37.5,maxlat]
region_lons = [minlon,-116,-105,-90,maxlon]




save_name = f"{drive}:/outputs/{country_save}\\average_temp_shape.csv"
average_filename = f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv"


df = pd.read_csv(save_name,dtype = {0:str})
eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(average_filename,dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()


#interping for shifts
temp_aves = df.drop(columns = "station").mean(axis = 0)

x_vals = np.arange(-0.5,0.5,1/1000)
ymax = np.nanmax(df.drop(columns = "station"))

xmin = np.nanmin(eTs)
xmax = np.nanmax(eTs)



interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df_parameters[3])
for i in np.arange(0,len(df_parameters[3])):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        



ymax2 = np.nanmax(interp_y)+0.05

temp_aves_proper = np.nanmean(interp_y,axis =0)

df_south = df[df_parameters[3].reset_index().latitude <= max_lat]
temp_aves_south = df_south.drop(columns = "station").mean(axis = 0)
eTs = np.array(eTs)
eTs_south = eTs[df_parameters[3].reset_index().latitude <= max_lat]

sds_south = np.array(sds)[df_parameters[3].reset_index().latitude <= max_lat]
aves_south = np.array(aves)[df_parameters[3].reset_index().latitude <= max_lat]
interp_y_south  = [np.nan] * len(df_south)

for i in np.arange(0,len(df_south)):  
    if np.isnan(aves_south[i]):
        interp_y_south[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs_south[i]-aves_south[i])/sds_south[i],df_south.iloc[i][1:])
        interp_y_south[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs_south[i]-aves_south[i])/sds_south[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs_south[i]-aves_south[i])/sds_south[i])]
        
        interp_y_south[i][(interp_x>=np.min((eTs_south[i]-aves_south[i])/sds_south[i])) & (interp_x<=np.max((eTs_south[i]-aves_south[i])/sds_south[i]))] = interp_func(interp_x_here)*sds_south[i]
        

temp_aves_proper_south = np.nanmean(interp_y_south,axis =0)




df_north = df[df_parameters[3].reset_index().latitude > max_lat]
temp_aves_north = df_north.drop(columns = "station").mean(axis = 0)
eTs = np.array(eTs)
eTs_north = eTs[df_parameters[3].reset_index().latitude > max_lat]


sds_north= np.array(sds)[df_parameters[3].reset_index().latitude > max_lat]
aves_north= np.array(aves)[df_parameters[3].reset_index().latitude > max_lat]
interp_y_north = [np.nan] * len(df_north)

for i in np.arange(0,len(df_north)):  
    if np.isnan(aves_north[i]):
        interp_y_north[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs_north[i]-aves_north[i])/sds_north[i],df_north.iloc[i][1:])
        interp_y_north[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs_north[i]-aves_north[i])/sds_north[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs_north[i]-aves_north[i])/sds_north[i])]
        
        interp_y_north[i][(interp_x>=np.min((eTs_north[i]-aves_north[i])/sds_north[i])) & (interp_x<=np.max((eTs_north[i]-aves_north[i])/sds_north[i]))] = interp_func(interp_x_here)*sds_north[i]
        

temp_aves_proper_north= np.nanmean(interp_y_north,axis =0)





# %% plotting the grid

lon_lims = [truncate_neg(np.min(df_parameters[3].longitude),2.5),np.ceil(np.max(df_parameters[3].longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters[3].latitude),2.5),np.ceil(np.max(df_parameters[3].latitude/2.5))*2.5]



n_lat = len(region_lats)-1
n_lon = len(region_lons)-1

# # show regions
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# ax1.coastlines()
# ax1.add_feature(cfeature.BORDERS, linestyle=':')

# for lat_i in range(n_lat-1):
#     ax1.plot([minlon-3,maxlon+3],[region_lats[lat_i+1],region_lats[lat_i+1]],  'r', linewidth=2, transform=ccrs.PlateCarree())

# for lon_i in range(n_lon-1):
#     ax1.plot([region_lons[lon_i+1],region_lons[lon_i+1]],[minlat-3,maxlat+3],  'r', linewidth=2, transform=ccrs.PlateCarree())

# plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
# plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
# plt.show()



fig,axs = plt.subplots(n_lat,n_lon,figsize = (n_lon*3,n_lat*3))
for lat_i in range(n_lat):
    for lon_i in range(n_lon):
        interp_y_region = np.array(interp_y)[(df_parameters[3].longitude<=region_lons[lon_i+1])
                                              & (df_parameters[3].longitude>region_lons[lon_i])
                                              & (df_parameters[3].latitude<=region_lats[lat_i+1])
                                              & (df_parameters[3].latitude>region_lats[lat_i])
                                             ]
        aves_region = aves[(df_parameters[3].longitude<=region_lons[lon_i+1])
                                              & (df_parameters[3].longitude>region_lons[lon_i])
                                              & (df_parameters[3].latitude<=region_lats[lat_i+1])
                                              & (df_parameters[3].latitude>region_lats[lat_i])
                                             ]
        
        axs[n_lat-1-lat_i,lon_i].tick_params(labelsize=fontsize)
        
        
        if n_lon>1:
            for i in np.arange(0,len(interp_y_region)):
                if np.isnan(aves_region[i]):
                    pass
                else:    
                    axs[n_lat-1-lat_i,lon_i].plot(interp_x ,interp_y_region[i],alpha = 0.01,color =  "b")
            axs[n_lat-1-lat_i,lon_i].plot(interp_x,np.nanmean(interp_y_region,axis=0),color = "r")
            
            axs[n_lat-1-lat_i,lon_i].set_ylim(0,0.5)
            
            axs[n_lat-1-lat_i,lon_i].set_xlabel(u"$(T - \mu)/\sigma$",fontsize = fontsize)
            
            
        else:
            for i in np.arange(0,len(interp_y_region)):  
                if np.isnan(aves_region[i]):
                    pass
                else:    
                    axs[n_lat-1-lat_i].plot(interp_x ,interp_y_region[i],alpha = 0.01,color =  "b")
            axs[n_lat-1-lat_i].plot(interp_x,np.nanmean(interp_y_region,axis=0),color = "r")
            axs[n_lat-1-lat_i].set_ylim(0,0.5)


plt.suptitle("Observed temperatures",fontsize = fontsize)
fig.tight_layout(pad=1.0, w_pad=1.9, h_pad=1.0)
plt.show()


# %% simulations, deninitions

#### a simulation of all the temperatures during the year

### define the easy functions
A = 13 #average yearly temperature
B = 5 # range of temperature going up and down
p = 365.25/(2*np.pi) #period (aka a year)
shift = 180 # shift of starting point


var = 2 #variance of the normal distribution around
delta = 0.5 #dependence on the day
ave_loc = 0 #equivalent to shift, where the changes start in the year
daysize = 3

x = np.arange(0,365)

xhour = np.arange(0,365,1/24)

# definition of the function for mu in the temperature distribution over the year
def yearly_mu(x, A, B, shift = 0, p = 365.25/(2*np.pi), daysize = 0, daylength = 1/(2*np.pi)):
    return A + B*np.sin((x + shift)/p) + daysize*np.sin(x/daylength)

def yearly_sigma(x, var, delta, ave_loc, p = 365.25/(2*np.pi)):
    return (1 + delta * np.sin(x/p + ave_loc/p))*var

def storm_filter(mu, sigma, x = np.arange(0,365)): #basically removes a chunk of days, following a normal distribution centered at day mu and spread by day sigma
    pdf = norm.pdf(x, loc = mu, scale = sigma)
    weights = (1-pdf)/len(x)
    return weights


def gen_sine_temperature_pdf(eT, A, B, var, delta, ave_loc, x= np.arange(0,365), p_mu = 365.25/(2*np.pi),  p_sigma = 365.25/(2*np.pi), daysize = 0, daylength = 1/(2*np.pi)):
    mu = yearly_mu(x, A, B, p = p_mu, daysize = daysize, daylength = daylength)
    sigma = yearly_sigma(x, var, delta, ave_loc, p = p_sigma)#*np.sqrt(2)
    
    norms = [norm.pdf(eT, loc = mu[i], scale = sigma[i])/len(x) for i in x] #TODO: here you can put weighting as a storm filter
    pdf = sum(norms) 
    return pdf



def sine_temperature_loglik_day_incl(theta, T, p = 365.25/(2*np.pi), daylength = 1/(2*np.pi)):
   
    A, B, var, delta = theta[0], theta[1], theta[2], theta[3] 
    
    ave_loc, daysize =theta[4], theta[5]
    
    pdf = gen_sine_temperature_pdf(T, A, B, var, delta, ave_loc, x= np.arange(0,365), p = p, daysize = daysize, daylength = daylength)
    
    return sum(np.log(pdf + 1e-10))

def sine_temperature_loglik(theta, T, p_mu = 365.25/(2*np.pi), p_sigma = 365.25/(2*np.pi)):
   
    A, B, var, delta, ave_loc = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    
    pdf = gen_sine_temperature_pdf(T, A, B, var, delta, ave_loc, x= np.arange(0,365), p_mu = p_mu, p_sigma = p_sigma)
    
    return sum(np.log(pdf + 1e-10))

def sine_temperature_model(T_obs, init_params = [13, 4, 3, 0.5, 90, 0], day_incl = False):
    if day_incl:
        phat = minimize(lambda theta: -sine_temperature_loglik_day_incl(theta, T_obs),
                        init_params,
                        method = 'Nelder-Mead')
        param_names = ['A', 'B', 'var', 'delta', 'ave_loc', 'daysize']
    else:
        init_params = init_params[0:-1]
        phat = minimize(lambda theta: -sine_temperature_loglik(theta, T_obs),
                        init_params,
                        method = 'Nelder-Mead')
        param_names = ['A', 'B', 'var', 'delta', 'ave_loc']
        
        
    return dict(zip(param_names, phat.x))

# functions to fit the two seperately
def datetime_series_to_array(series): #series = oe.oe_time
    dates = pd.to_datetime(series).dt.date
    start_date = dates[0]
    diffs = np.array([(day - start_date).days for day in dates])
    return diffs
    

# fits a sine wave to the observed T by minimizing the residuals
def yearly_mu_fit(x,T_obs,init_params = [13, 5, 0, 365.25/(2*np.pi)]): # x in days as integer array
    
    def residuals(theta, x, T_obs):
        mu_sim = yearly_mu(x, A = theta[0], B = theta[1], shift = theta[2], p = theta[3])
        
        diffs = mu_sim - T_obs
        
        return np.nansum(diffs**2)
    
    phat = minimize(lambda theta: residuals(theta, x, T_obs),
                    init_params,
                    method = 'L-BFGS-B')
    
    
    param_names = ['A', 'B', 'shift', 'p_mu']
    return dict(zip(param_names, phat.x))


def yearly_sigma_fit(std_obs_cycle, init_params = [5, 0.5, 100, 365.25/(2*np.pi)]): # std_obs_cycle is a pd.Series
    
    x = std_obs_cycle.index.astype("int")
    
    def residuals(theta, x, obs):
        sigma_sim = yearly_sigma(x,theta[0],theta[1],theta[2], p = theta[3])
        diffs = sigma_sim - obs
        
        return np.nansum(diffs**2)
    
    
    phat = minimize(lambda theta: residuals(theta, x, std_obs_cycle.to_numpy()),
                    init_params,
                    method = 'L-BFGS-B')
    
    param_names = ['var', 'delta', 'ave_loc', 'p_sigma']
    return dict(zip(param_names, phat.x))


# %% simulation plots

Bs = [2,6]
ave_locs = [60,200]

eT = np.arange(-12,40,0.2)

fig = plt.figure(figsize = (12,7))
for i in range(2):
    B = Bs[i]
    for j in range(2):
        ave_loc = ave_locs[j]
        
        ax = fig.add_subplot(2,4,1+4*i+2*j)
        ax.plot(x,yearly_mu(x, A, B, shift), color = "b")
        plt.fill_between(x, 
                         yearly_mu(x, A, B, shift) - yearly_sigma(x, var, delta, ave_loc),
                         yearly_mu(x, A, B, shift) + yearly_sigma(x, var, delta, ave_loc),
                         alpha = 0.5, color = "b")
        plt.title(f"B = {B}, φ = {ave_loc}",fontsize = fontsize)
        ax.set_xticks([0,180,365])
        ax.tick_params(labelsize=fontsize)
        
        plt.ylim(6,22)
        plt.xlim(0,365)
        plt.xlabel("day of year", fontsize = fontsize)
        plt.ylabel("T [°C]", fontsize = fontsize)
        
        norms = [gen_norm_pdf(eT, yearly_mu(x[i], A, B, shift),
                              yearly_sigma(x[i], var, delta, ave_loc), 2) for i in range(365)]
        
        
        
        ax = fig.add_subplot(2,4,2+4*i+2*j)
        ax.plot(eT,sum(norms)/365, color = "r")
        plt.xlim(0,30)
        plt.title(f"B = {B}, φ = {ave_loc}",fontsize = fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_xticks([0,10,20,30])
        plt.xlabel("T [°C]", fontsize = fontsize)
        

plt.suptitle(f"mu = {A} + B*sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + ave_loc)*2pi/365.25))*{var}")
fig.tight_layout(pad=1.0, w_pad=1., h_pad=1.2)
plt.show()





# %% fits to observations

stations = ["044181", "406750", "243558"]
xlims = [(-10,35),(-10,35),(-20,35)]
ylims = [(0,0.09),(0,0.07),(0,0.04)]

eT = np.arange(-20,40)

# next you plot the fits of the stations above


for i in range(len(stations)):
    station = stations[i]
    station_info = info[3][info[3].station == station]
    
    lat = station_info.latitude.to_numpy()[0]
    lon = station_info.longitude.to_numpy()[0]
    
    T_ = np.genfromtxt(f"D:/ordinary_events/US_main/T_{station}.csv")
    P_ = np.genfromtxt(f"D:/ordinary_events/US_main/P_{station}.csv")
    times = pd.read_csv(f"D:/ordinary_events/US_main/time_{station}.csv",parse_dates = ["oe_time"])
    
    
    oe = pd.DataFrame({
        "oe_time": times.oe_time,
        "T" : T_,
        "P" : P_
        })
    
    oe["date"] = pd.to_datetime(oe.oe_time).dt.date
    oe['days_of_year'] = pd.to_datetime(oe['date']).dt.dayofyear
    
    
    cycle_mean = oe.groupby("days_of_year")["T"].mean()
    cycle_std = oe.groupby("days_of_year")["T"].std()

    
    phat = sine_temperature_model(T_)
    pdf = gen_sine_temperature_pdf(eT,phat["A"],phat["B"],phat["var"],phat["delta"],phat["ave_loc"])
    
    eT_hist = np.arange(-20,40)
    eT_edges = np.concatenate([np.array([eT_hist[0]-(eT_hist[1]-eT_hist[0])/2]),(eT_hist + (eT_hist[1]-eT_hist[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(T_, bins=eT_edges, density=True)
    
    # the opposite way round
    days = datetime_series_to_array(oe.oe_time)
    
    phat_sigma = yearly_sigma_fit(cycle_std)
    phat_mu = yearly_mu_fit(days, T_)
    
    
    deviations_from_yearly_mu = oe["T"] - yearly_mu(days, phat_mu["A"], phat_mu["B"], shift = phat_mu["shift"], p = phat_mu["p_mu"])
    
    eT_hist_deviation = np.arange(-20,20)
    eT_edges_deviation = np.concatenate([np.array([eT_hist_deviation[0]-(eT_hist_deviation[1]-eT_hist_deviation[0])/2]),(eT_hist_deviation + (eT_hist_deviation[1]-eT_hist_deviation[0])/2)]) #convert bin centres into bin edges
    hist_deviation, bin_edges = np.histogram(deviations_from_yearly_mu, bins=eT_edges_deviation, density=True)
    
    
    
    pdf_back = gen_sine_temperature_pdf(eT, phat_mu["A"], phat_mu["B"],
                                        phat_sigma["var"], phat_sigma["delta"],
                                        phat_sigma["ave_loc"] - phat_mu["shift"],
                                        p_mu = phat_mu["p_mu"], p_sigma = phat_sigma["p_sigma"])
    
    
    
    
    day_difference = oe.days_of_year.iloc[0]
    
    
    fig = plt.figure(figsize = (13,5))
    ax = fig.add_subplot(1,3,1)
    plt.plot(eT_hist, hist, 'r--', linewidth = 3, label = "observations")
    
    plt.plot(eT,pdf, "b", linewidth = 3,  label = "fitted on pdf")
    
    plt.plot(eT, pdf_back, '#4daf4a', linewidth = 3,
             label = "backwards fit")
    
    plt.xlim(xlims[i])
    plt.ylim(ylims[i])
    
    
    
    plt.title(f"({lat:.2f}, {lon:.2f})",fontsize = fontsize)
    plt.xlabel("Temperature [°C]", fontsize = fontsize)
    plt.ylabel("pdf", fontsize = fontsize)
    
    plt.xticks(fontsize = fontsize)
    plt.yticks(np.arange(0,ylims[i][1]+0.01, 0.01),fontsize = fontsize)
    
    
    xs = np.arange(1,len(cycle_std)+1)
    
    
    shift = minimize(lambda theta: np.sum((yearly_mu(xs, phat["A"], phat["B"], shift = theta) - cycle_mean)**2),
                    0,
                    method='Nelder-Mead').x[0]
    
    day_difference = oe.days_of_year.astype("int").iloc[0]/(3600*24*10e8)
    
    
    
    
    
    ax = fig.add_subplot(1,3,2)
    
    #plot observed averaged cycle
    ax.plot(xs,cycle_mean,"r--", alpha = 0.5)
    
    #plot simulated cycle
    plt.plot(xs,yearly_mu(xs, phat["A"], phat["B"], shift = shift), label = "fitted on pdf",color = "b", linewidth = 3)
    plt.plot(xs,yearly_mu(xs, phat_mu["A"], phat_mu["B"], shift = phat_mu["shift"] - day_difference, p = phat_mu["p_mu"]), label = "backwards fit", color = '#4daf4a', linewidth = 3)
    
    
    
    plt.xticks([0,100,200,300], fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xlim(0,365)
    
    ax.set_xlabel("day of year", fontsize = fontsize)
    ax.set_ylabel("Temperature [°C]", fontsize = fontsize)
    plt.title("Mean yearly cycle", fontsize = fontsize)
    
    ax = fig.add_subplot(1,3,3)
    ax.plot(xs,cycle_std, "r--", alpha = 0.5)
    plt.plot(xs,yearly_sigma(xs,phat["var"],phat["delta"],shift+phat["ave_loc"]), color = "b", linewidth = 3) 
    plt.plot(xs,yearly_sigma(xs,phat_sigma["var"],phat_sigma["delta"],phat_sigma["ave_loc"],p=phat_sigma["p_sigma"]), color = '#4daf4a', linewidth = 3) 
    
    
    ax.set_xlabel("day of year", fontsize = fontsize)
    ax.set_ylabel("standard deviation [°C]", fontsize = fontsize)  
    
    plt.xticks([0,100,200,300],fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xlim(0,365)
    
    plt.title("Standard deviation yearly cycle", fontsize = fontsize)
    
    fig.tight_layout(pad=1.0, w_pad=1.9, h_pad=1.0)
    
    plt.show()
    



plt.plot([1,2,6],[1,3,6], label = "backwards fit", color = '#4daf4a', linewidth = 3)

plt.plot([1,2,6],[1,3,6], label = "fitted on pdf",color = "b", linewidth = 3)

plt.plot([1,2,6],[1,3,6],'r--', linewidth = 3, label = "observations")
plt.legend(fontsize = fontsize)
plt.show()
