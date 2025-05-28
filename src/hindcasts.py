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
from scipy.stats import chi2

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


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# station_chose = "18256"
# station_chose = "12261"
station_chose = "19376"

country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9



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
        beta = 4
    )


###############################################################################
# Just one station

# T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station_chose}.csv")
# P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station_chose}.csv")
# times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{station_chose}.csv",parse_dates = ["oe_time"])
# oe_df = pd.DataFrame({"year":times.oe_time.dt.year, "P": P, "T": T,})
# AMS = oe_df.groupby(oe_df.year).P.max().rename({"P" : "AMS"})

# start_time = times.iloc[0]
# end_time = times.iloc[-1]

# midyear = (start_time.dt.year + (end_time.dt.year - start_time.dt.year)/2).to_numpy()[0]

# T1 = T[times.oe_time.dt.year <= midyear]
# P1 = P[times.oe_time.dt.year <= midyear]
# times1 = times[times.oe_time.dt.year <= midyear]
# thr1 = np.quantile(P1,S.left_censoring[1])
# n1 = len(T1)/(midyear - start_time.dt.year + 1)
# AMS1 = pd.DataFrame(AMS[AMS.index <= midyear]).rename(columns = {"P" : "AMS"})


# T2 = T[times.oe_time.dt.year > midyear]
# P2 = P[times.oe_time.dt.year > midyear]
# times2 = times[times.oe_time.dt.year > midyear]
# thr2 = np.quantile(P2,S.left_censoring[1])
# n2 = len(T2)/(end_time.dt.year - midyear)
# AMS2 = pd.DataFrame(AMS[AMS.index > midyear]).rename(columns = {"P" : "AMS"})


# g_phat1 = S.temperature_model(T1)
# g_phat2 = S.temperature_model(T2)
# #g_phat2 = [g_phat1[0]+1,g_phat1[1]]


# F_phat1,_,_,_ = S.magnitude_model(P1, T1, thr1)
# F_phat2,_,_,_ = S.magnitude_model(P2, T2, thr2)

# S.alpha = 1

# F_phat1_b0,_,_,_ = S.magnitude_model(P1, T1, thr1)
# F_phat2_b0,_,_,_ = S.magnitude_model(P2, T2, thr2)


# eT = np.arange(np.min(T),np.max(T)+4)
# Ts = np.arange(np.min(T)- S.temp_delta, np.max(T)+ S.temp_delta, S.temp_res_monte_carlo)



# TNX_FIG_temp_model(T1, g_phat1, 6, eT,obscol='b',valcol='b',
#                        obslabel = f'observations {start_time.dt.year.to_numpy()[0]} - {int(midyear)}',
#                        vallabel = 'temperature model g(T) first period')

# TNX_FIG_temp_model(T2, g_phat2, 6, eT,obscol='r',valcol='r',
#                        obslabel = f'observations {int(midyear+1)} - {end_time.dt.year.to_numpy()[0]}',
#                        vallabel = 'temperature model g(T) second period')
# plt.show()


# RL1, _, _ = S.model_inversion(F_phat1_b0, g_phat1, n1, Ts)

# RL2, _, _ = S.model_inversion(F_phat1_b0, g_phat2, n1, Ts) #calculated with the same F_phat and n

# TNX_FIG_valid(AMS1, S.return_period, RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'first period',obslabel='Observed annual maxima')
# TNX_FIG_valid(AMS2, S.return_period, RL2,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'predicted second period',obslabel='Observed annual maxima')
# plt.title("b = 0")
# plt.show()


# RL1, _, _ = S.model_inversion(F_phat1, g_phat1, n1, Ts)

# RL2, _, _ = S.model_inversion(F_phat1, g_phat2, n1, Ts) #calculated with the same F_phat and n


# TNX_FIG_valid(AMS1, S.return_period, RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'first period',obslabel='Observed annual maxima')
# TNX_FIG_valid(AMS2, S.return_period, RL2,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'predicted second period',obslabel='Observed annual maxima')
# plt.title("free b")
# plt.show()


###############################################################################
# calculate g_phat 1 and 2 for all stations

hindcast_savename = f"{drive}:/outputs/{country_save}/hindcasts\\g_phat{S.beta}.csv"

hindcast_files = glob.glob(f"{drive}:/outputs/{country_save}/hindcasts\\*.csv")
if hindcast_savename not in hindcast_files:
    
    print(f"g_phat not calculated for two periods with beta = {S.beta}")
    
    g_phats1 = [0]*len(val_info)
    g_phats2 = [0]*len(val_info)
    
    starttime = [0]*len(val_info)
    
    for i in range(len(val_info)):
        
        starttime[i] = time.time()
        
        
        station = val_info.station.iloc[i]
        
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{station}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            g_phats1[i] = [np.nan,np.nan]
            g_phats2[i] = [np.nan,np.nan]
    
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
            
            times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{station}.csv",parse_dates = ["oe_time"])
            
            
            
            start_time = times.iloc[0]
            end_time = times.iloc[-1]
    
            midyear = np.trunc((start_time.dt.year + (end_time.dt.year - start_time.dt.year)/2).to_numpy()[0])
            T1 = T[times.oe_time.dt.year <= midyear]
            T2 = T[times.oe_time.dt.year > midyear]
            
    
            g_phats1[i] = S.temperature_model(T1)
            g_phats2[i] = S.temperature_model(T2)
            
        
        if i%50 == 0:
            time_taken = (time.time()-starttime[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
    
    hindcast_gphat = pd.DataFrame({"station" : val_info.station,
                                   "mu1": np.array(g_phats1)[:,0],
                                   "sigma1" : np.array(g_phats1)[:,1],
                                   "mu2": np.array(g_phats2)[:,0],
                                   "sigma2" : np.array(g_phats2)[:,1],
        })
    hindcast_gphat.to_csv(hindcast_savename, index = False)
else:
    print(f"gphats already saved for beta = {S.beta}, loading")
    hindcast_gphat = pd.read_csv(hindcast_savename, dtype = {"station" : str})

##############################################################################

#F_phat hindcast loop
hindcast_savename = f"{drive}:/outputs/{country_save}/hindcasts\\F_phat.csv"

hindcast_files = glob.glob(f"{drive}:/outputs/{country_save}/hindcasts\\*.csv")
if hindcast_savename not in hindcast_files:
    
    print(f"F_phat not calculated for two periods ")
    
    F_phats1 = [0]*len(val_info)
    F_phats2 = [0]*len(val_info)
    
    F_phats1_0 = [0]*len(val_info)
    F_phats2_0 = [0]*len(val_info)
    
    pvals = [0]*len(val_info)
    pvals_0 = [0]*len(val_info)
    
    
    starttime = [0]*len(val_info)
    
    for i in range(len(val_info)):
        
        starttime[i] = time.time()
        
        
        station = val_info.station.iloc[i]
        
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{station}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            
            F_phats1[i] = [np.nan,np.nan,np.nan,np.nan]
            F_phats2[i] = [np.nan,np.nan,np.nan,np.nan]
            
            F_phats1_0[i] = [np.nan,np.nan,np.nan,np.nan]
            F_phats2_0[i] = [np.nan,np.nan,np.nan,np.nan]
            
            pvals[i] = np.nan
            pvals_0[i] = np.nan
    
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
            P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
            times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{station}.csv",parse_dates = ["oe_time"])
            oe_df = pd.DataFrame({"year":times.oe_time.dt.year, "P": P, "T": T,})
            AMS = oe_df.groupby(oe_df.year).P.max()
            thr = np.quantile(P,S.left_censoring[1])
            
            
            start_time = times.iloc[0]
            end_time = times.iloc[-1]
            
            midyear = np.trunc((start_time.dt.year + (end_time.dt.year - start_time.dt.year)/2).to_numpy()[0])
            
            
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
            
            S.alpha = 0
            F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
            
            
            F_phats1[i],loglik1,_,_ = S.magnitude_model(P1, T1, thr1)
            F_phats2[i],loglik2,_,_ = S.magnitude_model(P2, T2, thr2)
            
            
            S.alpha = 1
            F_phat_b0, loglik_b0, _, _ = S.magnitude_model(P, T, thr)

            F_phats1_0[i],loglik1_b0,_,_ = S.magnitude_model(P1, T1, thr1)
            F_phats2_0[i],loglik2_b0,_,_ = S.magnitude_model(P2, T2, thr2)
            
            lambda_LR = -2*( loglik - (loglik1+loglik2) )
            pvals[i] = chi2.sf(lambda_LR, 4)
        
            
            lambda_LR = -2*( loglik_b0 - (loglik1_b0+loglik2_b0) )
            pvals_0[i] = chi2.sf(lambda_LR, 3)
            
        
        if i%50 == 0:
            time_taken = (time.time()-starttime[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
    
    hindcast_Fphat = pd.DataFrame({"station" : val_info.station,
                                   'kappa1':np.array(F_phats1)[:,0],
                                   'b1':np.array(F_phats1)[:,1],
                                   'lambda1':np.array(F_phats1)[:,2],
                                   'a1':np.array(F_phats1)[:,3],
                                   'kappa2':np.array(F_phats2)[:,0],
                                   'b2':np.array(F_phats2)[:,1],
                                   'lambda2':np.array(F_phats2)[:,2],
                                   'a2':np.array(F_phats2)[:,3],
                                   'kappa1_0':np.array(F_phats1_0)[:,0],
                                   'b1_0':np.array(F_phats1_0)[:,1],
                                   'lambda1_0':np.array(F_phats1_0)[:,2],
                                   'a1_0':np.array(F_phats1_0)[:,3],
                                   'kappa2_0':np.array(F_phats2_0)[:,0],
                                   'b2_0':np.array(F_phats2_0)[:,1],
                                   'lambda2_0':np.array(F_phats2_0)[:,2],
                                   'a2_0':np.array(F_phats2_0)[:,3],
                                   'pvals' : pvals,
                                   'pvals_0' : pvals_0
        })
    hindcast_Fphat.to_csv(hindcast_savename, index = False)
else:
    print("Fphats already saved, loading")
    hindcast_Fphat = pd.read_csv(hindcast_savename, dtype = {"station" : str})


val_info.index = range(len(val_info))

norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = 'Blues'
# plot comparisons of the two period F_phat values
variables = ["kappa","b","lambda","a"]
for vari in variables:  
    [slope,intc] = np.polyfit(hindcast_Fphat[f"{vari}1"],hindcast_Fphat[f"{vari}2"],1)
    if vari != "b":    
        [slope_0,intc_0] = np.polyfit(hindcast_Fphat[f"{vari}1_0"],hindcast_Fphat[f"{vari}2_0"],1)
    
    x = np.arange(np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1,(np.max(hindcast_Fphat[f"{vari}1"])*1.1 - np.min(hindcast_Fphat[f"{vari}1"]))/10)
    y = intc + slope * x
    
    if vari != "b":    
        y_0 = intc_0 + slope_0 * x
    
    
    
    fig = plt.figure(figsize = (10,5))
    ax1 = fig.add_subplot(1,2,1)
    sc = ax1.scatter(hindcast_Fphat[f"{vari}1"],hindcast_Fphat[f"{vari}2"],
                s=3,c = hindcast_Fphat.pvals,
                norm = norm, cmap = cmap, marker = "*" if val_info.cleaned_years>=30 else ".")
    
    ax1.plot([np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],[np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],label = "line of equality")
    ax1.plot(x,y,label = "best fit")
    ax1.set_xlabel(f"{vari}1")
    ax1.set_ylabel(f"{vari}2")
    ax1.set_title("free b")
    plt.legend()
    
    ax2 = fig.add_subplot(1,2,2)
    sc = ax2.scatter(hindcast_Fphat[f"{vari}1_0"],hindcast_Fphat[f"{vari}2_0"],
                s=3,c = hindcast_Fphat.pvals,
                norm = norm, cmap = cmap)
    ax2.plot([np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],[np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],label = "line of equality")
    
    if vari != "b":    
        ax2.plot(x,y_0,label = "best fit")
    ax2.set_xlabel(f"{vari}1_0")
    ax2.set_ylabel(f"{vari}2_0")
    ax2.set_title("b = 0")
    
    
    cbar_ax = fig.add_subplot([0.15, -0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('p-value', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()


fig = plt.figure(figsize = (12,7))
ax1 = fig.add_subplot(1,2,1)
plt.hist(hindcast_Fphat.pvals.dropna(),density = True)
plt.ylim(0,6)
plt.xlabel("p value")
plt.title("b=free")


ax2 = fig.add_subplot(1,2,2)
plt.hist(hindcast_Fphat.pvals_0.dropna(),density = True)
plt.ylim(0,6)
plt.xlabel("p value")
plt.title("b=0")
plt.show()


# cutting out shorter years

hindcast_Fphat_short = hindcast_Fphat[val_info.cleaned_years>=30]

norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = 'Blues'
# plot comparisons of the two period F_phat values
variables = ["kappa","b","lambda","a"]
for vari in variables:  
    [slope,intc] = np.polyfit(hindcast_Fphat_short[f"{vari}1"],hindcast_Fphat_short[f"{vari}2"],1)
    if vari != "b":    
        [slope_0,intc_0] = np.polyfit(hindcast_Fphat_short[f"{vari}1_0"],hindcast_Fphat_short[f"{vari}2_0"],1)
    
    x = np.arange(np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1,(np.max(hindcast_Fphat_short[f"{vari}1"])*1.1 - np.min(hindcast_Fphat_short[f"{vari}1"]))/10)
    y = intc + slope * x
    
    if vari != "b":    
        y_0 = intc_0 + slope_0 * x
    
    
    
    fig = plt.figure(figsize = (10,5))
    ax1 = fig.add_subplot(1,2,1)
    sc = ax1.scatter(hindcast_Fphat_short[f"{vari}1"],hindcast_Fphat_short[f"{vari}2"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    
    ax1.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality")
    ax1.plot(x,y,label = "best fit")
    ax1.set_xlabel(f"{vari}1")
    ax1.set_ylabel(f"{vari}2")
    ax1.set_title("free b")
    plt.legend()
    
    ax2 = fig.add_subplot(1,2,2)
    sc = ax2.scatter(hindcast_Fphat_short[f"{vari}1_0"],hindcast_Fphat_short[f"{vari}2_0"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    ax2.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality")
    
    if vari != "b":    
        ax2.plot(x,y_0,label = "best fit")
    ax2.set_xlabel(f"{vari}1_0")
    ax2.set_ylabel(f"{vari}2_0")
    ax2.set_title("b = 0")
    
    
    cbar_ax = fig.add_subplot([0.15, -0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('p-value', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()



fig = plt.figure(figsize = (12,7))
ax1 = fig.add_subplot(1,2,1)
plt.hist(hindcast_Fphat_short.pvals.dropna(),density = True,bins = 20)
plt.ylim(0,12)
plt.xlabel("p value")
plt.title("b=free 30 yrs plus")


ax2 = fig.add_subplot(1,2,2)
plt.hist(hindcast_Fphat_short.pvals_0.dropna(),density = True,bins = 20)
plt.ylim(0,12)
plt.xlabel("p value")
plt.title("b=0 30 yrs plus")
plt.show()














perc_different5_0 = len(hindcast_Fphat[hindcast_Fphat.pvals_0<0.05])/len(hindcast_Fphat)
perc_different10_0 = len(hindcast_Fphat[hindcast_Fphat.pvals_0<0.1])/len(hindcast_Fphat)


perc_different5 = len(hindcast_Fphat[hindcast_Fphat.pvals<0.05])/len(hindcast_Fphat)
perc_different10 = len(hindcast_Fphat[hindcast_Fphat.pvals<0.1])/len(hindcast_Fphat)

print(f"b = 0: percentage of stations where F_phat different in {country} at 5% level: {perc_different5_0 *100:.1f}%")
print(f"b = free: percentage of stations where F_phat different in {country} at 5% level: {perc_different5 *100:.1f}%")


print(f"b = 0: percentage of stations where F_phat different in {country} at 10% level: {perc_different10_0 *100:.1f}%")
print(f"b = free: percentage of stations where F_phat different in {country} at 10% level: {perc_different10 *100:.1f}%")



delta_mu = hindcast_gphat.mu2 - hindcast_gphat.mu1







###############################################################################
#hindcasts loop

# val_info.index = range(len(val_info))
# df_parameters.index = range(len(df_parameters))
# delta_mu.index = range(len(delta_mu))




# mask = ((val_info.cleaned_years >= 20) &
#         (delta_mu >= 1) 
#         & (val_info.latitude > 40)
#         )


# info_mask = val_info[mask]
# parameters_mask = df_parameters[mask]

# n_hindcasts = len(info_mask)


# for i in range(n_hindcasts):
#     station = info_mask.station.iloc[i]
#     T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
#     P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
#     times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{station}.csv",parse_dates = ["oe_time"])
#     oe_df = pd.DataFrame({"year":times.oe_time.dt.year, "P": P, "T": T,})
#     AMS = oe_df.groupby(oe_df.year).P.max()
#     thr = parameters_mask.thr.iloc[i]
    
#     unique_years = oe_df.year.unique()
    
    
#     start_time = times.iloc[0]
#     end_time = times.iloc[-1]
    
#     n = len(T)/(len(unique_years))

#     midyear = unique_years[int(np.trunc(len(unique_years)/2) - 1)]
    
#     S.alpha = 0
#     F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
#     g_phat = S.temperature_model(T)
    
#     T1 = T[times.oe_time.dt.year <= midyear]
#     P1 = P[times.oe_time.dt.year <= midyear]
#     times1 = times[times.oe_time.dt.year <= midyear]
#     thr1 = np.quantile(P1,S.left_censoring[1])
#     n1 = len(T1)/(midyear - start_time.dt.year + 1)
#     AMS1 = pd.DataFrame(AMS[AMS.index <= midyear]).rename(columns = {"P" : "AMS"})


#     T2 = T[times.oe_time.dt.year > midyear]
#     P2 = P[times.oe_time.dt.year > midyear]
#     times2 = times[times.oe_time.dt.year > midyear]
#     thr2 = np.quantile(P2,S.left_censoring[1])
#     n2 = len(T2)/(end_time.dt.year - midyear)
#     AMS2 = pd.DataFrame(AMS[AMS.index > midyear]).rename(columns = {"P" : "AMS"})


#     g_phat1 = S.temperature_model(T1)
#     g_phat2 = S.temperature_model(T2)
    
#     delta_mu_here = g_phat2[0] - g_phat1[0]
#     g_phat2 = [g_phat1[0]+delta_mu_here,g_phat1[1]]


#     F_phat1,loglik1,_,_ = S.magnitude_model(P1, T1, thr1)
#     F_phat2,loglik2,_,_ = S.magnitude_model(P2, T2, thr2)

#     S.alpha = 1
#     F_phat_b0, loglik_b0, _, _ = S.magnitude_model(P, T, thr)

#     F_phat1_b0,loglik1_b0,_,_ = S.magnitude_model(P1, T1, thr1)
#     F_phat2_b0,loglik2_b0,_,_ = S.magnitude_model(P2, T2, thr2)


#     eT = np.arange(np.min(T),np.max(T)+4)
#     Ts = np.arange(np.min(T)- S.temp_delta, np.max(T)+ S.temp_delta, S.temp_res_monte_carlo)



#     TNX_FIG_temp_model(T1, g_phat1, S.beta, eT,obscol='b',valcol='b',
#                            obslabel = f'observations {start_time.dt.year.to_numpy()[0]} - {int(midyear)}',
#                            vallabel = 'temperature model g(T) first period')

#     TNX_FIG_temp_model(T2, g_phat2, S.beta, eT,obscol='r',valcol='r',
#                            obslabel = f'observations {int(midyear+1)} - {end_time.dt.year.to_numpy()[0]}',
#                            vallabel = 'temperature model g(T) second period')
#     plt.xlim(np.min(T)-4,np.max(T)+4)
#     plt.title(f"{station}.")
#     plt.show()

#     RL, _, _ = S.model_inversion(F_phat, g_phat, n, Ts)
    
#     RL1, _, _ = S.model_inversion(F_phat1_b0, g_phat1, n1, Ts)

#     RL2, _, _ = S.model_inversion(F_phat1_b0, g_phat2, n1, Ts) #calculated with the same F_phat and n
    
#     lambda_LR = -2*( loglik - (loglik1+loglik2) )
#     pval = chi2.sf(lambda_LR, 4)
#     if pval > 0.05:
#         mag_str = f"p={pval}. Magnitude models not  different at 5% significance."
#     else:
#         mag_str = f"p={pval}. Magnitude models are different at 5% significance."
    
    
#     lambda_LR = -2*( loglik_b0 - (loglik1_b0+loglik2_b0) )
#     pval = chi2.sf(lambda_LR, 3)
#     if pval > 0.05:
#         mag_str_b0 = f"p={pval}. Magnitude models not  different at 5% significance."
#     else:
#         mag_str_b0 = f"p={pval}. Magnitude models are different at 5% significance."
    
    
#     TNX_FIG_valid(pd.DataFrame(AMS).rename(columns = {"P" : "AMS"}), S.return_period, RL,TENAXcol='b',obscol_shape = 'g+',TENAXlabel = 'TENAX all',obslabel='Observed annual maxima')
#     plt.show()
    
#     qs = [0.75,0.9,0.99]
#     TNX_FIG_magn_model(P, T, F_phat, thr, eT, qs)
#     plt.show()
    
#     TNX_FIG_valid(AMS1, S.return_period, RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'first period',obslabel='Observed annual maxima')
#     TNX_FIG_valid(AMS2, S.return_period, RL2,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'predicted second period',obslabel='Observed annual maxima')
#     plt.ylim(0,np.nanmax(RL2))
#     plt.title(f"{station}.b = 0.\n {mag_str_b0}")
#     plt.show()


#     RL1, _, _ = S.model_inversion(F_phat1, g_phat1, n1, Ts)

#     RL2, _, _ = S.model_inversion(F_phat1, g_phat2, n1, Ts) #calculated with the same F_phat and n


#     TNX_FIG_valid(AMS1, S.return_period, RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'first period',obslabel='Observed annual maxima')
#     TNX_FIG_valid(AMS2, S.return_period, RL2,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'predicted second period',obslabel='Observed annual maxima')
#     plt.ylim(0,np.nanmax(RL2))
#     plt.title(f"{station}.  free b. \n {mag_str}")
#     plt.show()










