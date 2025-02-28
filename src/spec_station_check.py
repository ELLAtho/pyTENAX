# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:52:23 2025

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

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import xarray as xr
import time

drive = 'D'


chosen_station = ['12041']

country = 'Japan'
code_str = 'JP' 
n_stations = 1 #number of stations to sample
min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
max_yrs = 1000 #if no max, set to very high
name_col = 'ppt'
temp_name_col = "t2m"

comb = pd.read_csv('D:/metadata/'+country+'_fulldata.csv', dtype={'station': str})

comb.startdate = pd.to_datetime(comb.startdate)
comb.enddate = pd.to_datetime(comb.enddate)

selected = comb[comb.station.isin(chosen_station)]


#PLOT SELECTED STATION LOCATION

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS)
plt.scatter(selected['longitude'],selected['latitude'])
plt.xlim(np.min(comb.longitude)-5,np.max(comb.longitude)+5)
plt.ylim(np.min(comb.latitude)-5,np.max(comb.latitude)+5)
plt.show()

###############################################################################
# Look at plots from df arrays
output_files = glob.glob(f"{drive}:/outputs/{country}/*")

save_path_neg = drive + ':/outputs/'+country+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 
df_parameters_0 = pd.read_csv(f"{drive}:/outputs/{country}_b0/parameters.csv", dtype={'station': str})



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

save_bset = f"{drive}:/outputs/{country}\\parameters_bset.csv"

if save_bset in output_files:
    df_parameters_bset = pd.read_csv(save_bset,dtype={'station': str})
else:
    pass


liklihood_df = pd.read_csv(f"{drive}:/outputs/{country}/liklihood.csv",dtype={'station': str})

nan_locs = liklihood_df.mult_prob[liklihood_df.mult_prob.isna()].index
replace_range = np.arange(0,len(liklihood_df))
for k in range(len(nan_locs)):
    replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
for j in replace_range:
    liklihood_df.at[j, "maxes"] = eval(liklihood_df.maxes.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df.at[j, "mins"] = eval(liklihood_df.mins.iloc[j], {"np": np, "nan": np.nan})
    
    liklihood_df.at[j, "maxes_0"] = eval(liklihood_df.maxes_0.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df.at[j, "mins_0"] = eval(liklihood_df.mins_0.iloc[j], {"np": np, "nan": np.nan})
    
    liklihood_df.at[j, "maxes_5"] = eval(liklihood_df.maxes_5.iloc[j], {"np": np, "nan": np.nan})
    liklihood_df.at[j, "mins_5"] = eval(liklihood_df.mins_5.iloc[j], {"np": np, "nan": np.nan})
    
    

RL_df = pd.read_csv(f"{drive}:/outputs/{country}/return_levels.csv", dtype={'station': str})
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


FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country}/FRMSE.csv", dtype={'station': str})




AMS = RL_df.obs_AMS[RL_df.station == chosen_station[0]].iloc[0]

plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))

eRP = 1/(1-plot_pos)


TNX_FIG_valid(AMS,
              eRP,
              RL_df.return_levels_b0[RL_df.station == chosen_station[0]].iloc[0],
              TENAXlabel = "b = 0.",obslabel='AMS')


plt.plot(eRP,RL_df.return_levels[RL_df.station == chosen_station[0]].iloc[0],
         "r",label = "b = free.")

#plt.plot(eRP,RL_df.return_levels_5[RL_df.station == chosen_station[0]],"g",alpha = 0.5, label = "b = 5% sig")
if "return_levels_bset" in RL_df.columns:
    plt.plot(eRP,
             RL_df.return_levels_bset[RL_df.station == chosen_station[0]].iloc[0],
             "y", label = "b = mean.")
    
if "return_levels_bexp" in RL_df.columns:
    plt.plot(eRP,
             RL_df.return_levels_bexp[RL_df.station == chosen_station[0]].iloc[0],
             "m", label = "b exp.")

plt.fill_between(eRP,
                 liklihood_df.mins_0[RL_df.station == chosen_station[0]].iloc[0],
                 liklihood_df.maxes_0[RL_df.station == chosen_station[0]].iloc[0],
                 color = "b", alpha = 0.1)

plt.fill_between(eRP,
                 liklihood_df.mins[RL_df.station == chosen_station[0]].iloc[0],
                 liklihood_df.maxes[RL_df.station == chosen_station[0]].iloc[0],
                 color = "r", alpha = 0.1)

plt.ylim(0,np.max(RL_df.return_levels[RL_df.station == chosen_station[0]].iloc[0])+5)
plt.xlim(1,np.max(eRP)+2)

plt.legend()
plt.title(f"station {chosen_station[0]}.")
plt.show()

###############################################################################

ppt_filename = f"{drive}:/{country}/{code_str}_{chosen_station[0]}.txt"
G, metadata = read_GSDR_file(ppt_filename,name_col)


T_path = f"{drive}:/{country}_temp\\{code_str}_{chosen_station[0]}.nc"
T_ERA = xr.load_dataarray(T_path)


S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0,
        min_ev_dur = 60,
    )


data = S.remove_incomplete_years(G, name_col)
t_data = (T_ERA-273.15).to_dataframe()

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
 
 
 
df_arr_t_data = np.array(t_data[temp_name_col])
df_dates_t_data = np.array(t_data.index)
 
dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
 
 
 
start_time = time.time()
 # Your data (P, T arrays) and threshold thr=3.8
P = dict_ordinary["60"]["ordinary"].to_numpy() 
T = dict_ordinary["60"]["T"].to_numpy()  
 
 
 # Number of threshold 
thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
 
 # Sampling intervals for the Montecarlo
Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
 
AMS = dict_AMS['60']
AMS_sort = AMS.sort_values(by=['AMS'])['AMS']
plot_pos = np.arange(1,np.size(AMS_sort)+1)/(1+np.size(AMS_sort))
eRP = 1/(1-plot_pos)
S.return_period = eRP
 
 #TENAX MODEL HERE
 #magnitude model
F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
F_phat_exp, loglik_exp, _, _ = S.magnitude_model(P, T, thr,b_exp = True)
#temperature model
g_phat = S.temperature_model(T)
# M is mean n of ordinary events
n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
#estimates return levels using MC samples
RL, __, __ = S.model_inversion(F_phat, g_phat, n, Ts)
RL_exp, __, __ = S.model_inversion(F_phat_exp, g_phat, n, Ts, b_exp = True)
 
S.n_monte_carlo = np.size(P)*S.niter_smev
_, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,gen_P_mc = True,gen_RL=False) 
S.n_monte_carlo = 20000
 
print(RL)
 
 
#PLOTTING THE GRAPHS
 
eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
 
 # fig 2a
qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [eT[0],eT[-1]])
TNX_FIG_magn_model(P,T,F_phat_exp,thr,eT,qs,xlimits = [eT[0],eT[-1]],valcol='g',b_exp = True)
plt.show()
 
 # fig 2a
qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [eT[0],eT[-1]])
plt.show()
 
 #fig 2b
TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])

plt.show()
 
 #fig 4 (without SMEV and uncertainty) 
AMS = dict_AMS['60'] # yet the annual maxima
TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+3])
TNX_FIG_valid(AMS,S.return_period,RL_exp,ylimits = [0,np.max(AMS.AMS)+3],TENAXcol = "g")

plt.show()


################################################################################
#Visual single filter
P_thrplus= P[P>=thr]
T_thrplus= T[P>=thr]


P_bad = P_thrplus[T_thrplus<-10]
T_bad = T_thrplus[T_thrplus<-10]

P_bad_loc = np.where((P == P_bad) & (T == T_bad))

P_filter = np.delete(P, P_bad_loc)
T_filter = np.delete(T, P_bad_loc)

n_filter = len(P_filter)/len(n_ordinary_per_year)
thr_filter = np.quantile(P_filter,S.left_censoring[1])

# Sampling intervals for the Montecarlo
Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
 
#TENAX MODEL HERE
#magnitude model
F_phat_filter , loglik, _, _ = S.magnitude_model(P_filter , T_filter , thr_filter)
F_phat_exp_filter , loglik_exp, _, _ = S.magnitude_model(P_filter , T_filter , thr_filter,b_exp = True)
#temperature model
g_phat_filter  = S.temperature_model(T_filter)

#estimates return levels using MC samples
RL_filter , __, __ = S.model_inversion(F_phat_filter, g_phat_filter, n_filter, Ts)
RL_exp_filter , __, __ = S.model_inversion(F_phat_exp_filter, g_phat_filter, n_filter, Ts, b_exp = True)
 
S.n_monte_carlo = np.size(P)*S.niter_smev
_, T_mc_filter, P_mc_filter  = S.model_inversion(F_phat_filter, g_phat_filter, n, Ts,gen_P_mc = True,gen_RL=False) 
S.n_monte_carlo = 20000
 
print(RL_filter )
 
 
#PLOTTING THE GRAPHS
 
eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
 


# fig 2a
qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P_filter,T_filter,F_phat_filter ,thr,eT,qs,xlimits = [eT[0],eT[-1]])

plt.scatter(T_bad,P_bad,alpha = 0.3)
plt.title("Visual outlier removed")
plt.show()


 # fig 2a
qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P_filter,T_filter,F_phat_filter ,thr,eT,qs,xlimits = [eT[0],eT[-1]])
TNX_FIG_magn_model(P_filter ,T_filter ,F_phat_exp_filter ,thr,eT,qs,xlimits = [eT[0],eT[-1]],valcol='g',b_exp = True)
plt.title("Visual outlier removed")
plt.show()
 
 #fig 2b
TNX_FIG_temp_model(T=T_filter, g_phat=g_phat_filter,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])

plt.show()
 
 #fig 4 (without SMEV and uncertainty) 
AMS = dict_AMS['60'] # yet the annual maxima
TNX_FIG_valid(AMS,S.return_period,RL_filter ,ylimits = [0,np.max(AMS.AMS)+3])
TNX_FIG_valid(AMS,S.return_period,RL_exp_filter ,ylimits = [0,np.max(AMS.AMS)+3],TENAXcol = "g")

plt.show()



###############################################################################

T_mc = T_mc.reshape(-1)
W_log_transform = np.log((F_phat[0]+F_phat[1]*T)*(np.log(P)-np.log(F_phat[2]*np.exp(F_phat[3]*T)))) #log(log(1/(1-W)))


plt.scatter(W_log_transform,P)
plt.yscale('log')
plt.ylim(1,100)
plt.show()


###############################################################################
#non power n plotsp = 0
S.return_period = S.return_period *29
RL_all, __, __ = S.model_inversion(F_phat, g_phat, 1.0, Ts)

plot_pos_full = np.arange(1,np.size(P)+1)/(1+np.size(P))
eRP_full = 1/(1-plot_pos_full)/n.to_numpy()


plt.scatter(eRP_full,np.sort(P))
plt.xscale("log")
plt.plot(S.return_period/29,RL_all)
plt.show()





















