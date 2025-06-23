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



# save_name = f"{drive}:/outputs/{country_save}\\return_levels.csv"

# RL_df = pd.read_csv(save_name, dtype={'station': str})
# nan_locs = RL_df.return_levels[RL_df.return_levels.isna()].index
# replace_range = np.arange(0,len(RL_df))

# RL_column_names = [col for col in RL_df.columns if "return_levels" in col]


# for col in RL_column_names:
#     nan_locs = RL_df[col][RL_df[col].isna()].index
#     replace_range = np.arange(0,len(RL_df))
#     for k in range(len(nan_locs)):
#         replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
#     for j in replace_range:
    
#         RL_df.loc[j, col] = np.fromstring(RL_df[col].iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        
# nan_locs = RL_df.obs_AMS[RL_df.obs_AMS.isna()].index
# replace_range = np.arange(0,len(RL_df))
# for k in range(len(nan_locs)):
#     replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
# for j in replace_range:
#     RL_df.loc[j, "obs_AMS"] = np.fromstring(RL_df.obs_AMS.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')



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
            n1 = len(T1)/(midyear - start_time.dt.year + 1)
            AMS1 = pd.DataFrame(AMS[AMS.index <= midyear]).rename(columns = {"P" : "AMS"})
            
            
            T2 = T[times.oe_time.dt.year > midyear]
            P2 = P[times.oe_time.dt.year > midyear]
            times2 = times[times.oe_time.dt.year > midyear]
            n2 = len(T2)/(end_time.dt.year - midyear)
            AMS2 = pd.DataFrame(AMS[AMS.index > midyear]).rename(columns = {"P" : "AMS"})
            
            S.alpha = 0
            F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
            
            
            F_phats1[i],loglik1,_,_ = S.magnitude_model(P1, T1, thr)
            F_phats2[i],loglik2,_,_ = S.magnitude_model(P2, T2, thr)
            
            
            S.alpha = 1
            F_phat_b0, loglik_b0, _, _ = S.magnitude_model(P, T, thr)

            F_phats1_0[i],loglik1_b0,_,_ = S.magnitude_model(P1, T1, thr)
            F_phats2_0[i],loglik2_b0,_,_ = S.magnitude_model(P2, T2, thr)
            
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

##############################################################################

#F_phat hindcast loop
hindcast_savename_exp = f"{drive}:/outputs/{country_save}/hindcasts\\F_phat_exp.csv"

hindcast_files = glob.glob(f"{drive}:/outputs/{country_save}/hindcasts\\*.csv")
if hindcast_savename_exp not in hindcast_files:
    
    print(f"F_phat not calculated for two periods exponential")
    
    F_phats1 = [0]*len(val_info)
    F_phats2 = [0]*len(val_info)
    
    
    pvals = [0]*len(val_info)
    
    
    starttime = [0]*len(val_info)
    
    for i in range(len(val_info)):
        
        starttime[i] = time.time()
        
        
        station = val_info.station.iloc[i]
        
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{station}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            
            F_phats1[i] = [np.nan,np.nan,np.nan,np.nan]
            F_phats2[i] = [np.nan,np.nan,np.nan,np.nan]
            
            pvals[i] = np.nan
    
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
            n1 = len(T1)/(midyear - start_time.dt.year + 1)
            AMS1 = pd.DataFrame(AMS[AMS.index <= midyear]).rename(columns = {"P" : "AMS"})
            
            
            T2 = T[times.oe_time.dt.year > midyear]
            P2 = P[times.oe_time.dt.year > midyear]
            times2 = times[times.oe_time.dt.year > midyear]
            n2 = len(T2)/(end_time.dt.year - midyear)
            AMS2 = pd.DataFrame(AMS[AMS.index > midyear]).rename(columns = {"P" : "AMS"})
            
            S.alpha = 0
            F_phat, loglik, _, _ = S.magnitude_model(P, T, thr, b_exp = True)
            
            
            F_phats1[i],loglik1,_,_ = S.magnitude_model(P1, T1, thr, b_exp = True)
            F_phats2[i],loglik2,_,_ = S.magnitude_model(P2, T2, thr, b_exp = True)
            
            
            lambda_LR = -2*( loglik - (loglik1+loglik2) )
            pvals[i] = chi2.sf(lambda_LR, 4)
        
            
            
        
        if i%50 == 0:
            time_taken = (time.time()-starttime[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
    
    hindcast_Fphat_exp = pd.DataFrame({"station" : val_info.station,
                                   'kappa1':np.array(F_phats1)[:,0],
                                   'b1':np.array(F_phats1)[:,1],
                                   'lambda1':np.array(F_phats1)[:,2],
                                   'a1':np.array(F_phats1)[:,3],
                                   'kappa2':np.array(F_phats2)[:,0],
                                   'b2':np.array(F_phats2)[:,1],
                                   'lambda2':np.array(F_phats2)[:,2],
                                   'a2':np.array(F_phats2)[:,3],
                                   'pvals' : pvals,
        })
    hindcast_Fphat_exp.to_csv(hindcast_savename_exp, index = False)
else:
    print("Fphats already saved for exponential, loading")
    hindcast_Fphat_exp = pd.read_csv(hindcast_savename_exp, dtype = {"station" : str})

##############################################################################

val_info.index = range(len(val_info))

norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = 'plasma'
# plot comparisons of the two period F_phat values
variables = ["kappa","b","lambda","a"]
for vari in variables:  
    df_small = hindcast_Fphat[[f"{vari}1",f"{vari}2",f"{vari}1_0",f"{vari}2_0"]]
    corr_table = df_small.corr()
    
    
    poly_model = odr.polynomial(1)  # using first order polynomial model
    data = odr.Data(hindcast_Fphat[f"{vari}1"].dropna(),hindcast_Fphat[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y = poly(hindcast_Fphat[f"{vari}1"].dropna())
    
    
    if vari != "b":    
        data = odr.Data(hindcast_Fphat[f"{vari}1_0"].dropna(),hindcast_Fphat[f"{vari}2_0"].dropna())
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()  # running ODR fitting
        poly = np.poly1d(output.beta[::-1])
        poly_y_0 = poly(hindcast_Fphat[f"{vari}1_0"].dropna())
    
    #exponential stuff
    df_small_exp = hindcast_Fphat_exp[[f"{vari}1",f"{vari}2"]]
    corr_table_exp = df_small_exp.corr()
    
    
    data = odr.Data(hindcast_Fphat_exp[f"{vari}1"].dropna(),hindcast_Fphat_exp[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y_exp = poly(hindcast_Fphat_exp[f"{vari}1"].dropna())
    
    
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(1,3,1)
    sc = ax1.scatter(hindcast_Fphat[f"{vari}1"],hindcast_Fphat[f"{vari}2"],
                s=3,c = hindcast_Fphat.pvals,
                norm = norm, cmap = cmap)#, marker = "*" if val_info.cleaned_years>=30 else ".")
    
    ax1.plot([np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],[np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax1.plot(hindcast_Fphat[f"{vari}1"].dropna(),poly_y,label = "best fit",color = "r")
    ax1.set_xlabel(f"{vari}1")
    ax1.set_ylabel(f"{vari}2")
    ax1.set_title(f"free b. corr = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}")
    plt.legend()
    
    ax2 = fig.add_subplot(1,3,2)
    sc = ax2.scatter(hindcast_Fphat[f"{vari}1_0"],hindcast_Fphat[f"{vari}2_0"],
                s=3,c = hindcast_Fphat.pvals,
                norm = norm, cmap = cmap)
    ax2.plot([np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],[np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    
    if vari != "b":    
        ax2.plot(hindcast_Fphat[f"{vari}1_0"].dropna(),poly_y_0,label = "best fit",color = "r")
    ax2.set_xlabel(f"{vari}1_0")
    ax2.set_ylabel(f"{vari}2_0")
    ax2.set_title(f"b = 0. corr = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}")
    
    
    ax3 = fig.add_subplot(1,3,3)
    sc = ax3.scatter(hindcast_Fphat_exp[f"{vari}1"],hindcast_Fphat_exp[f"{vari}2"],
                s=3,c = hindcast_Fphat_exp.pvals,
                norm = norm, cmap = cmap)#, marker = "*" if val_info.cleaned_years>=30 else ".")
    
    ax3.plot([np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],[np.min(hindcast_Fphat[f"{vari}1"]),np.max(hindcast_Fphat[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax3.plot(hindcast_Fphat_exp[f"{vari}1"].dropna(),poly_y_exp,label = "best fit",color = "r")
    ax3.set_xlabel(f"{vari}1")
    ax3.set_ylabel(f"{vari}2")
    ax3.set_title(f"free b exponential. corr = {corr_table_exp[f"{vari}1"][f"{vari}2"]:.2f}")
    
    
    
    
    cbar_ax = fig.add_subplot([0.15, -0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('p-value', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    
    
    
    
    plt.tight_layout()
    plt.suptitle(f"{country_save}")
    plt.show()


fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot(1,3,1)
plt.hist(hindcast_Fphat.pvals.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title("b=free")


ax2 = fig.add_subplot(1,3,2)
plt.hist(hindcast_Fphat.pvals_0.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title("b=0")


ax3 = fig.add_subplot(1,3,3)
plt.hist(hindcast_Fphat_exp.pvals.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title("b=free exponential")







plt.suptitle(f"{country_save}")
plt.show()



#plot maps

base_cmap = plt.cm.get_cmap("viridis")
color_list = base_cmap(np.linspace(0,1,10))

discrete_viridis = ListedColormap(color_list, name = "viridis")

s = 3
fontsize = 12

fig = plt.figure(figsize=(8, 22))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 2)
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = hindcast_Fphat.pvals,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)

# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
plt.title(f'{country_save} p value, b = free', fontsize=16)


ax2 = fig.add_subplot(3, 1, 2, projection=proj)
ax2.coastlines(zorder = 2)
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = hindcast_Fphat.pvals_0,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)

plt.title(f'{country_save} p value, b = 0', fontsize=16)

ax3 = fig.add_subplot(3, 1, 3, projection=proj)
ax3.coastlines(zorder = 2)
ax3.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.Normalize(vmin=-1, vmax=1)
sc = ax3.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = hindcast_Fphat.pvals_0 - hindcast_Fphat.pvals,
    s = s,
    cmap = 'seismic',
    norm = norm,zorder = 1
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}

# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('delta p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)

plt.title('p_0 - p_free', fontsize=16)

plt.show()

# make list of significance
significance = 0.05
sig_list = hindcast_Fphat.pvals > significance #True/1 = insignificant
sig_list_0 = hindcast_Fphat.pvals_0 > significance

hindcast_Fphat["sig"] = sig_list.replace({True: 1, False: 0})
hindcast_Fphat["sig_0"] = sig_list_0.replace({True: 1, False: 0})

changes_pvals = hindcast_Fphat.sig + hindcast_Fphat.sig_0*2 # 0 means both sig, 1 means free insig but 0 sig, 2 means free sig then 0 insig, 3 means both insig


# plot map showing sig vs not sig

norm = mcolors.Normalize(vmin=0, vmax=1)
s = 3
fontsize = 12

fig = plt.figure(figsize=(10, 18))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 2)
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = sig_list,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


legend_elements = [
    Patch(facecolor=base_cmap(1.0), label=f'insignificant at {significance*100}%'),  # default matplotlib colors
    Patch(facecolor=base_cmap(0), label=f'significant at {significance*100}%'),
]

plt.legend(handles=legend_elements)



plt.title(f'{country_save} p value, b = free', fontsize=16)


ax2 = fig.add_subplot(3, 1, 2, projection=proj)
ax2.coastlines(zorder = 2)
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = sig_list_0,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


plt.legend(handles=legend_elements)

plt.title(f'{country_save} p value, b = 0', fontsize=16)



base_cmap = plt.cm.get_cmap("rainbow") #new cmap for ax3

ax3 = fig.add_subplot(3, 1, 3, projection=proj)
ax3.coastlines(zorder = 2)
ax3.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.Normalize(vmin=0, vmax=3)
sc = ax3.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = changes_pvals,
    s = s,
    cmap = 'rainbow',
    norm = norm,zorder = 1
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


legend_elements = [
    Patch(facecolor=base_cmap(0), label=f'both significant at {significance*100}% ({np.sum(changes_pvals==0)*100/len(changes_pvals):.0f}% stations)'),  # default matplotlib colors
    Patch(facecolor=base_cmap(1/3), label=f'free insignificant, 0 significant ({np.sum(changes_pvals==1)*100/len(changes_pvals):.0f}% stations)'),
    Patch(facecolor=base_cmap(2/3), label=f'free significant, 0 insignificant ({np.sum(changes_pvals==2)*100/len(changes_pvals):.0f}% stations)'),
    Patch(facecolor=base_cmap(1.0), label=f'both insignificant ({np.sum(changes_pvals==3)*100/len(changes_pvals):.0f}% stations)'),
]

plt.legend(handles=legend_elements)

plt.title('changes in significance', fontsize=16)

plt.show()



# cutting out shorter years
min_years_strong = 20

if min_years_strong < np.max(val_info.cleaned_years):
    hindcast_Fphat_short = hindcast_Fphat[val_info.cleaned_years>=min_years_strong]
    hindcast_Fphat_exp_short = hindcast_Fphat_exp[val_info.cleaned_years>=min_years_strong]
    new_df_short = new_df[val_info.cleaned_years>=min_years_strong]
    
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = 'plasma'
    # plot comparisons of the two period F_phat values
    variables = ["kappa","b","lambda","a"]
    for vari in variables: 
        df_small = hindcast_Fphat_short[[f"{vari}1",f"{vari}2",f"{vari}1_0",f"{vari}2_0"]]
        df_small_exp = hindcast_Fphat_exp_short[[f"{vari}1",f"{vari}2"]]
        corr_table = df_small.corr()
        corr_table_exp = df_small_exp.corr()
        
        poly_model = odr.polynomial(1)  # using first order polynomial model
        data = odr.Data(hindcast_Fphat_short[f"{vari}1"].dropna(),hindcast_Fphat_short[f"{vari}2"].dropna())
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()  # running ODR fitting
        poly = np.poly1d(output.beta[::-1])
        poly_y = poly(hindcast_Fphat_short[f"{vari}1"].dropna())
        
        
        data = odr.Data(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),hindcast_Fphat_exp_short[f"{vari}2"].dropna())
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()  # running ODR fitting
        poly = np.poly1d(output.beta[::-1])
        poly_y_exp = poly(hindcast_Fphat_exp_short[f"{vari}1"].dropna())
        
        
        if vari != "b":    
            data = odr.Data(hindcast_Fphat_short[f"{vari}1_0"].dropna(),hindcast_Fphat_short[f"{vari}2_0"].dropna())
            odr_obj = odr.ODR(data, poly_model)
            output = odr_obj.run()  # running ODR fitting
            poly = np.poly1d(output.beta[::-1])
            poly_y_0 = poly(hindcast_Fphat_short[f"{vari}1_0"].dropna())
        
        
        
        fig = plt.figure(figsize = (12,5))
        ax1 = fig.add_subplot(1,3,1)
        sc = ax1.scatter(hindcast_Fphat_short[f"{vari}1"],hindcast_Fphat_short[f"{vari}2"],
                    s=3,c = hindcast_Fphat_short.pvals,
                    norm = norm, cmap = cmap)
        
        ax1.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
        ax1.plot(hindcast_Fphat_short[f"{vari}1"].dropna(),poly_y,label = "best fit",color = "r")
        ax1.set_xlabel(f"{vari}1")
        ax1.set_ylabel(f"{vari}2")
        ax1.set_title(f"free b. corr = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}")
        plt.legend()
        
        ax2 = fig.add_subplot(1,3,2)
        sc = ax2.scatter(hindcast_Fphat_short[f"{vari}1_0"],hindcast_Fphat_short[f"{vari}2_0"],
                    s=3,c = hindcast_Fphat_short.pvals,
                    norm = norm, cmap = cmap)
        ax2.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
        
        if vari != "b":    
            ax2.plot(hindcast_Fphat_short[f"{vari}1_0"].dropna(),poly_y_0,label = "best fit",color = "r")
            
        ax2.set_xlabel(f"{vari}1_0")
        ax2.set_ylabel(f"{vari}2_0")
        ax2.set_title(f"b = 0. corr = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}")
        
        ax3 = fig.add_subplot(1,3,3)
        sc = ax3.scatter(hindcast_Fphat_exp_short[f"{vari}1"],hindcast_Fphat_exp_short[f"{vari}2"],
                    s=3,c = hindcast_Fphat_exp_short.pvals,
                    norm = norm, cmap = cmap)#, marker = "*" if val_info.cleaned_years>=30 else ".")
        
        ax3.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
        ax3.plot(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),poly_y_exp,label = "best fit",color = "r")
        ax3.set_xlabel(f"{vari}1")
        ax3.set_ylabel(f"{vari}2")
        ax3.set_title(f"free b exponential. corr = {corr_table_exp[f"{vari}1"][f"{vari}2"]:.2f}")
        
        
        
        
        cbar_ax = fig.add_subplot([0.15, -0.02, 0.7, 0.03])  # Position for the colorbar
        cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        cb.set_label('p-value', fontsize=14)
        cb.ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.suptitle(f"{country_save}. longer than {min_years_strong} years")
        plt.show()
    
    
    
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(1,3,1)
    plt.hist(hindcast_Fphat_short.pvals.dropna(),density = True,bins = 20)
    plt.ylim(0,7)
    plt.xlabel("p value")
    plt.title(f"b=free {min_years_strong} yrs plus")
    
    
    ax2 = fig.add_subplot(1,3,2)
    plt.hist(hindcast_Fphat_short.pvals_0.dropna(),density = True,bins = 20)
    plt.ylim(0,7)
    plt.xlabel("p value")
    plt.title(f"b=0 {min_years_strong} yrs plus")
    
    ax3 = fig.add_subplot(1,3,3)
    plt.hist(hindcast_Fphat_exp_short.pvals.dropna(),density = True,bins = 20)
    plt.ylim(0,7)
    plt.xlabel("p value")
    plt.title("b=free exponential")

    
    plt.suptitle(f"{country_save}")
    plt.show()
    
    
    
    
    #plot maps
    
    base_cmap = plt.cm.get_cmap("viridis")
    color_list = base_cmap(np.linspace(0,1,10))
    
    discrete_viridis = ListedColormap(color_list, name = "viridis")
    
    s = 3
    fontsize = 12
    
    fig = plt.figure(figsize=(8, 22))
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(3, 1, 1, projection=proj)
    
    # Add map features
    ax1.coastlines(zorder = 2)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax1.scatter( #plot the negligable at 5% lvl points
        new_df_short.longitude,
        new_df_short.latitude,
        c = hindcast_Fphat_short.pvals,
        s = s,
        cmap = discrete_viridis,
        norm = norm,zorder = 1
    )
    
    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
    cb.set_label('p value', fontsize=14)  
    cb.ax.tick_params(labelsize=12)
    
    
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize-2}
    gl.ylabel_style = {'size': fontsize-2}
    plt.title(f'{country_save} p value, b = free. more than {min_years_strong} years', fontsize=16)
    
    
    ax2 = fig.add_subplot(3, 1, 2, projection=proj)
    ax2.coastlines(zorder = 2)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax2.scatter( #plot the negligable at 5% lvl points
        new_df_short.longitude,
        new_df_short.latitude,
        c = hindcast_Fphat_short.pvals_0,
        s = s,
        cmap = discrete_viridis,
        norm = norm,zorder = 1
    )
    
    
    gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize-2}
    gl.ylabel_style = {'size': fontsize-2}
    
    
    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
    cb.set_label('p value', fontsize=14)  
    cb.ax.tick_params(labelsize=12)
    
    plt.title(f'{country_save} p value, b = 0', fontsize=16)
    
    ax3 = fig.add_subplot(3, 1, 3, projection=proj)
    ax3.coastlines(zorder = 2)
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sc = ax3.scatter( #plot the negligable at 5% lvl points
        new_df_short.longitude,
        new_df_short.latitude,
        c = hindcast_Fphat_short.pvals_0 - hindcast_Fphat_short.pvals,
        s = s,
        cmap = 'seismic',
        norm = norm,zorder = 1
    )
    
    
    gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize-2}
    gl.ylabel_style = {'size': fontsize-2}
    
    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
    cb.set_label('delta p value', fontsize=14)  
    cb.ax.tick_params(labelsize=12)
    
    plt.title('p_0 - p_free', fontsize=16)
    
    plt.show()
    
    # make list of significance
    significance = 0.05
    sig_list = hindcast_Fphat_short.pvals > significance #True/1 = insignificant
    sig_list_0 = hindcast_Fphat_short.pvals_0 > significance
    
    hindcast_Fphat_short["sig"] = sig_list.replace({True: 1, False: 0})
    hindcast_Fphat_short["sig_0"] = sig_list_0.replace({True: 1, False: 0})
    
    changes_pvals = hindcast_Fphat_short.sig + hindcast_Fphat_short.sig_0*2 # 0 means both sig, 1 means free insig but 0 sig, 2 means free sig then 0 insig, 3 means both insig
    
    
    # plot map showing sig vs not sig
    
    norm = mcolors.Normalize(vmin=0, vmax=1)
    s = 3
    fontsize = 12
    
    fig = plt.figure(figsize=(10, 18))
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(3, 1, 1, projection=proj)
    
    # Add map features
    ax1.coastlines(zorder = 2)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax1.scatter( #plot the negligable at 5% lvl points
        new_df_short.longitude,
        new_df_short.latitude,
        c = sig_list,
        s = s,
        cmap = discrete_viridis,
        norm = norm,zorder = 1
    )
    
    
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize-2}
    gl.ylabel_style = {'size': fontsize-2}
    
    
    legend_elements = [
        Patch(facecolor=base_cmap(1.0), label=f'insignificant at {significance*100}%'),  # default matplotlib colors
        Patch(facecolor=base_cmap(0), label=f'significant at {significance*100}%'),
    ]
    
    plt.legend(handles=legend_elements)
    
    
    
    plt.title(f'{country_save} p value, b = free. more than {min_years_strong} years', fontsize=16)
    
    
    ax2 = fig.add_subplot(3, 1, 2, projection=proj)
    ax2.coastlines(zorder = 2)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax2.scatter( #plot the negligable at 5% lvl points
        new_df_short.longitude,
        new_df_short.latitude,
        c = sig_list_0,
        s = s,
        cmap = discrete_viridis,
        norm = norm,zorder = 1
    )
    
    
    gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize-2}
    gl.ylabel_style = {'size': fontsize-2}
    
    
    plt.legend(handles=legend_elements)
    
    plt.title(f'{country_save} p value, b = 0', fontsize=16)
    
    
    
    base_cmap = plt.cm.get_cmap("rainbow") #new cmap for ax3
    
    ax3 = fig.add_subplot(3, 1, 3, projection=proj)
    ax3.coastlines(zorder = 2)
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    
    norm = mcolors.Normalize(vmin=0, vmax=3)
    sc = ax3.scatter( #plot the negligable at 5% lvl points
        new_df_short.longitude,
        new_df_short.latitude,
        c = changes_pvals,
        s = s,
        cmap = 'rainbow',
        norm = norm,zorder = 1
    )
    
    
    gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize-2}
    gl.ylabel_style = {'size': fontsize-2}
    
    
    legend_elements = [
        Patch(facecolor=base_cmap(0), label=f'both significant at {significance*100}% ({np.sum(changes_pvals==0)*100/len(changes_pvals):.0f}% stations)'),  # default matplotlib colors
        Patch(facecolor=base_cmap(1/3), label=f'free insignificant, 0 significant ({np.sum(changes_pvals==1)*100/len(changes_pvals):.0f}% stations)'),
        Patch(facecolor=base_cmap(2/3), label=f'free significant, 0 insignificant ({np.sum(changes_pvals==2)*100/len(changes_pvals):.0f}% stations)'),
        Patch(facecolor=base_cmap(1.0), label=f'both insignificant ({np.sum(changes_pvals==3)*100/len(changes_pvals):.0f}% stations)'),
    ]
    
    plt.legend(handles=legend_elements)
    
    plt.title('changes in significance', fontsize=16)
    
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

else:
    print("there isn't any data that long")
    perc_different5_0 = len(hindcast_Fphat[hindcast_Fphat.pvals_0<0.05])/len(hindcast_Fphat)
    perc_different10_0 = len(hindcast_Fphat[hindcast_Fphat.pvals_0<0.1])/len(hindcast_Fphat)
    
    
    perc_different5 = len(hindcast_Fphat[hindcast_Fphat.pvals<0.05])/len(hindcast_Fphat)
    perc_different10 = len(hindcast_Fphat[hindcast_Fphat.pvals<0.1])/len(hindcast_Fphat)
    
    print(f"b = 0: percentage of stations where F_phat different in {country} at 5% level: {perc_different5_0 *100:.1f}%")
    print(f"b = free: percentage of stations where F_phat different in {country} at 5% level: {perc_different5 *100:.1f}%")
    
    
    print(f"b = 0: percentage of stations where F_phat different in {country} at 10% level: {perc_different10_0 *100:.1f}%")
    print(f"b = free: percentage of stations where F_phat different in {country} at 10% level: {perc_different10 *100:.1f}%")
    




###############################################################################
# CUTTING OUT RESOLUTION CHANGES
resolution_df = pd.read_csv(f"{drive}:/outputs/resolutions/{country_save}/resolution_info.csv")

hindcast_Fphat_short = hindcast_Fphat[resolution_df.n_mins==1]
new_df_short = new_df[resolution_df.n_mins==1]

hindcast_Fphat_exp_short = hindcast_Fphat_exp[resolution_df.n_mins==1]

norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = 'plasma'
# plot comparisons of the two period F_phat values
variables = ["kappa","b","lambda","a"]
for vari in variables: 
    df_small = hindcast_Fphat_short[[f"{vari}1",f"{vari}2",f"{vari}1_0",f"{vari}2_0"]]
    corr_table = df_small.corr()
    df_small_exp = hindcast_Fphat_exp_short[[f"{vari}1",f"{vari}2"]]
    corr_table_exp = df_small_exp.corr()
    
    poly_model = odr.polynomial(1)  # using first order polynomial model
    data = odr.Data(hindcast_Fphat_short[f"{vari}1"].dropna(),hindcast_Fphat_short[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y = poly(hindcast_Fphat_short[f"{vari}1"].dropna())
    
    data = odr.Data(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),hindcast_Fphat_exp_short[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y_exp = poly(hindcast_Fphat_exp_short[f"{vari}1"].dropna())
    
    if vari != "b":    
        data = odr.Data(hindcast_Fphat_short[f"{vari}1_0"].dropna(),hindcast_Fphat_short[f"{vari}2_0"].dropna())
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()  # running ODR fitting
        poly = np.poly1d(output.beta[::-1])
        poly_y_0 = poly(hindcast_Fphat_short[f"{vari}1_0"].dropna())
    
    
    
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(1,3,1)
    sc = ax1.scatter(hindcast_Fphat_short[f"{vari}1"],hindcast_Fphat_short[f"{vari}2"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    
    ax1.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax1.plot(hindcast_Fphat_short[f"{vari}1"].dropna(),poly_y,label = "best fit",color = "r")
    ax1.set_xlabel(f"{vari}1")
    ax1.set_ylabel(f"{vari}2")
    ax1.set_title(f"free b. corr = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}")
    plt.legend()
    
    ax2 = fig.add_subplot(1,3,2)
    sc = ax2.scatter(hindcast_Fphat_short[f"{vari}1_0"],hindcast_Fphat_short[f"{vari}2_0"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    ax2.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    
    if vari != "b":    
        ax2.plot(hindcast_Fphat_short[f"{vari}1_0"].dropna(),poly_y_0,label = "best fit",color = "r")
        
    ax2.set_xlabel(f"{vari}1_0")
    ax2.set_ylabel(f"{vari}2_0")
    ax2.set_title(f"b = 0. corr = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}")
    
    ax3 = fig.add_subplot(1,3,3)
    sc = ax3.scatter(hindcast_Fphat_exp_short[f"{vari}1"],hindcast_Fphat_exp_short[f"{vari}2"],
                s=3,c = hindcast_Fphat_exp_short.pvals,
                norm = norm, cmap = cmap)#, marker = "*" if val_info.cleaned_years>=30 else ".")
    
    ax3.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax3.plot(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),poly_y_exp,label = "best fit",color = "r")
    ax3.set_xlabel(f"{vari}1")
    ax3.set_ylabel(f"{vari}2")
    ax3.set_title(f"free b exponential. corr = {corr_table_exp[f"{vari}1"][f"{vari}2"]:.2f}")
    
    
    cbar_ax = fig.add_subplot([0.15, -0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('p-value', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.suptitle(f"{country_save}. resolution doesn't change")
    plt.show()



fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot(1,3,1)
plt.hist(hindcast_Fphat_short.pvals.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title(f"b=free only one resolution")


ax2 = fig.add_subplot(1,3,2)
plt.hist(hindcast_Fphat_short.pvals_0.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title(f"b=0 one resolution")

ax3 = fig.add_subplot(1,3,3)
plt.hist(hindcast_Fphat_exp.pvals.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title("b=free exponential")

plt.suptitle(f"{country_save}")
plt.show()




#plot maps

base_cmap = plt.cm.get_cmap("viridis")
color_list = base_cmap(np.linspace(0,1,10))

discrete_viridis = ListedColormap(color_list, name = "viridis")

s = 3
fontsize = 12

fig = plt.figure(figsize=(8, 22))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 2)
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = hindcast_Fphat_short.pvals,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)

# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
plt.title(f'{country_save} p value, b = free. resolution doesnt change', fontsize=16)


ax2 = fig.add_subplot(3, 1, 2, projection=proj)
ax2.coastlines(zorder = 2)
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = hindcast_Fphat_short.pvals_0,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)

plt.title(f'{country_save} p value, b = 0', fontsize=16)

ax3 = fig.add_subplot(3, 1, 3, projection=proj)
ax3.coastlines(zorder = 2)
ax3.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.Normalize(vmin=-1, vmax=1)
sc = ax3.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = hindcast_Fphat_short.pvals_0 - hindcast_Fphat_short.pvals,
    s = s,
    cmap = 'seismic',
    norm = norm,zorder = 1
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}

# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('delta p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)

plt.title('p_0 - p_free', fontsize=16)

plt.show()

# make list of significance
significance = 0.05
sig_list = hindcast_Fphat_short.pvals > significance #True/1 = insignificant
sig_list_0 = hindcast_Fphat_short.pvals_0 > significance

hindcast_Fphat_short["sig"] = sig_list.replace({True: 1, False: 0})
hindcast_Fphat_short["sig_0"] = sig_list_0.replace({True: 1, False: 0})

changes_pvals = hindcast_Fphat_short.sig + hindcast_Fphat_short.sig_0*2 # 0 means both sig, 1 means free insig but 0 sig, 2 means free sig then 0 insig, 3 means both insig


# plot map showing sig vs not sig

norm = mcolors.Normalize(vmin=0, vmax=1)
s = 3
fontsize = 12

fig = plt.figure(figsize=(10, 18))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 2)
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = sig_list,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


legend_elements = [
    Patch(facecolor=base_cmap(1.0), label=f'insignificant at {significance*100}%'),  # default matplotlib colors
    Patch(facecolor=base_cmap(0), label=f'significant at {significance*100}%'),
]

plt.legend(handles=legend_elements)



plt.title(f'{country_save} p value, b = free. resolution doesnt change', fontsize=16)


ax2 = fig.add_subplot(3, 1, 2, projection=proj)
ax2.coastlines(zorder = 2)
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = sig_list_0,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


plt.legend(handles=legend_elements)

plt.title(f'{country_save} p value, b = 0', fontsize=16)



base_cmap = plt.cm.get_cmap("rainbow") #new cmap for ax3

ax3 = fig.add_subplot(3, 1, 3, projection=proj)
ax3.coastlines(zorder = 2)
ax3.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.Normalize(vmin=0, vmax=3)
sc = ax3.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = changes_pvals,
    s = s,
    cmap = 'rainbow',
    norm = norm,zorder = 1
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


legend_elements = [
    Patch(facecolor=base_cmap(0), label=f'both significant at {significance*100}% ({np.sum(changes_pvals==0)*100/len(changes_pvals):.0f}% stations)'),  # default matplotlib colors
    Patch(facecolor=base_cmap(1/3), label=f'free insignificant, 0 significant ({np.sum(changes_pvals==1)*100/len(changes_pvals):.0f}% stations)'),
    Patch(facecolor=base_cmap(2/3), label=f'free significant, 0 insignificant ({np.sum(changes_pvals==2)*100/len(changes_pvals):.0f}% stations)'),
    Patch(facecolor=base_cmap(1.0), label=f'both insignificant ({np.sum(changes_pvals==3)*100/len(changes_pvals):.0f}% stations)'),
]

plt.legend(handles=legend_elements)

plt.title('changes in significance', fontsize=16)

plt.show()



###############################################################################
#resolution doesn't change AND long yrs only


hindcast_Fphat_short = hindcast_Fphat[(resolution_df.n_mins==1)&(val_info.cleaned_years>=min_years_strong)]
new_df_short = new_df[(resolution_df.n_mins==1)&(val_info.cleaned_years>=min_years_strong)]
hindcast_Fphat_exp_short = hindcast_Fphat_exp[(resolution_df.n_mins==1)&(val_info.cleaned_years>=min_years_strong)]

norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = 'plasma'
# plot comparisons of the two period F_phat values
variables = ["kappa","b","lambda","a"]
for vari in variables: 
    df_small = hindcast_Fphat_short[[f"{vari}1",f"{vari}2",f"{vari}1_0",f"{vari}2_0"]]
    corr_table = df_small.corr()
    df_small_exp = hindcast_Fphat_exp_short[[f"{vari}1",f"{vari}2"]]
    corr_table_exp = df_small_exp.corr()
    
    
    poly_model = odr.polynomial(1)  # using first order polynomial model
    data = odr.Data(hindcast_Fphat_short[f"{vari}1"].dropna(),hindcast_Fphat_short[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y = poly(hindcast_Fphat_short[f"{vari}1"].dropna())
    
    data = odr.Data(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),hindcast_Fphat_exp_short[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y_exp = poly(hindcast_Fphat_exp_short[f"{vari}1"].dropna())
    
    
    
    if vari != "b":    
        data = odr.Data(hindcast_Fphat_short[f"{vari}1_0"].dropna(),hindcast_Fphat_short[f"{vari}2_0"].dropna())
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()  # running ODR fitting
        poly = np.poly1d(output.beta[::-1])
        poly_y_0 = poly(hindcast_Fphat_short[f"{vari}1_0"].dropna())
    
    
    
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(1,3,1)
    sc = ax1.scatter(hindcast_Fphat_short[f"{vari}1"],hindcast_Fphat_short[f"{vari}2"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    
    ax1.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax1.plot(hindcast_Fphat_short[f"{vari}1"].dropna(),poly_y,label = "best fit",color = "r")
    ax1.set_xlabel(f"{vari}1")
    ax1.set_ylabel(f"{vari}2")
    ax1.set_title(f"free b. corr = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}")
    plt.legend()
    
    ax2 = fig.add_subplot(1,3,2)
    sc = ax2.scatter(hindcast_Fphat_short[f"{vari}1_0"],hindcast_Fphat_short[f"{vari}2_0"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    ax2.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    
    if vari != "b":    
        ax2.plot(hindcast_Fphat_short[f"{vari}1_0"].dropna(),poly_y_0,label = "best fit",color = "r")
        
    ax2.set_xlabel(f"{vari}1_0")
    ax2.set_ylabel(f"{vari}2_0")
    ax2.set_title(f"b = 0. corr = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}")
    
    ax3 = fig.add_subplot(1,3,3)
    sc = ax3.scatter(hindcast_Fphat_exp_short[f"{vari}1"],hindcast_Fphat_exp_short[f"{vari}2"],
                s=3,c = hindcast_Fphat_exp_short.pvals,
                norm = norm, cmap = cmap)#, marker = "*" if val_info.cleaned_years>=30 else ".")
    
    ax3.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax3.plot(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),poly_y_exp,label = "best fit",color = "r")
    ax3.set_xlabel(f"{vari}1")
    ax3.set_ylabel(f"{vari}2")
    ax3.set_title(f"free b exponential. corr = {corr_table_exp[f"{vari}1"][f"{vari}2"]:.2f}")
    
    
    cbar_ax = fig.add_subplot([0.15, -0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('p-value', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.suptitle(f"{country_save}. resolution doesn't change and years longer than {min_years_strong}")
    plt.show()



fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot(1,3,1)
plt.hist(hindcast_Fphat_short.pvals.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title(f"b=free only one resolution and {min_years_strong} yrs plus")


ax2 = fig.add_subplot(1,3,2)
plt.hist(hindcast_Fphat_short.pvals_0.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title(f"b=0 one resolution and {min_years_strong} yrs plus")

ax3 = fig.add_subplot(1,3,3)
plt.hist(hindcast_Fphat_exp.pvals.dropna(),density = True,bins = 20)
plt.ylim(0,7)
plt.xlabel("p value")
plt.title("b=free exponential")

plt.suptitle(f"{country_save}")
plt.show()




#plot maps

base_cmap = plt.cm.get_cmap("viridis")
color_list = base_cmap(np.linspace(0,1,10))

discrete_viridis = ListedColormap(color_list, name = "viridis")

s = 3
fontsize = 12

fig = plt.figure(figsize=(8, 22))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 2)
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = hindcast_Fphat_short.pvals,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)

# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
plt.title(f'{country_save} p value, b = free. resolution doesnt change and years longer than {min_years_strong}', fontsize=16)


ax2 = fig.add_subplot(3, 1, 2, projection=proj)
ax2.coastlines(zorder = 2)
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = hindcast_Fphat_short.pvals_0,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)

plt.title(f'{country_save} p value, b = 0', fontsize=16)

ax3 = fig.add_subplot(3, 1, 3, projection=proj)
ax3.coastlines(zorder = 2)
ax3.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.Normalize(vmin=-1, vmax=1)
sc = ax3.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = hindcast_Fphat_short.pvals_0 - hindcast_Fphat_short.pvals,
    s = s,
    cmap = 'seismic',
    norm = norm,zorder = 1
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}

# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15)
cb.set_label('delta p value', fontsize=14)  
cb.ax.tick_params(labelsize=12)

plt.title('p_0 - p_free', fontsize=16)

plt.show()

# make list of significance
significance = 0.05
sig_list = hindcast_Fphat_short.pvals > significance #True/1 = insignificant
sig_list_0 = hindcast_Fphat_short.pvals_0 > significance

hindcast_Fphat_short["sig"] = sig_list.replace({True: 1, False: 0})
hindcast_Fphat_short["sig_0"] = sig_list_0.replace({True: 1, False: 0})

changes_pvals = hindcast_Fphat_short.sig + hindcast_Fphat_short.sig_0*2 # 0 means both sig, 1 means free insig but 0 sig, 2 means free sig then 0 insig, 3 means both insig


# plot map showing sig vs not sig

norm = mcolors.Normalize(vmin=0, vmax=1)
s = 3
fontsize = 12

fig = plt.figure(figsize=(10, 18))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 2)
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = sig_list,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


legend_elements = [
    Patch(facecolor=base_cmap(1.0), label=f'insignificant at {significance*100}%'),  # default matplotlib colors
    Patch(facecolor=base_cmap(0), label=f'significant at {significance*100}%'),
]

plt.legend(handles=legend_elements)



plt.title(f'{country_save} p value, b = free. resolution doesnt change and years longer than {min_years_strong}', fontsize=16)


ax2 = fig.add_subplot(3, 1, 2, projection=proj)
ax2.coastlines(zorder = 2)
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = sig_list_0,
    s = s,
    cmap = discrete_viridis,
    norm = norm,zorder = 1
)


gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


plt.legend(handles=legend_elements)

plt.title(f'{country_save} p value, b = 0', fontsize=16)



base_cmap = plt.cm.get_cmap("rainbow") #new cmap for ax3

ax3 = fig.add_subplot(3, 1, 3, projection=proj)
ax3.coastlines(zorder = 2)
ax3.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.Normalize(vmin=0, vmax=3)
sc = ax3.scatter( #plot the negligable at 5% lvl points
    new_df_short.longitude,
    new_df_short.latitude,
    c = changes_pvals,
    s = s,
    cmap = 'rainbow',
    norm = norm,zorder = 1
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}


legend_elements = [
    Patch(facecolor=base_cmap(0), label=f'both significant at {significance*100}% ({np.sum(changes_pvals==0)*100/len(changes_pvals):.0f}% stations)'),  # default matplotlib colors
    Patch(facecolor=base_cmap(1/3), label=f'free insignificant, 0 significant ({np.sum(changes_pvals==1)*100/len(changes_pvals):.0f}% stations)'),
    Patch(facecolor=base_cmap(2/3), label=f'free significant, 0 insignificant ({np.sum(changes_pvals==2)*100/len(changes_pvals):.0f}% stations)'),
    Patch(facecolor=base_cmap(1.0), label=f'both insignificant ({np.sum(changes_pvals==3)*100/len(changes_pvals):.0f}% stations)'),
]

plt.legend(handles=legend_elements)

plt.title('changes in significance', fontsize=16)

plt.show()

#################################################################################
# drop outliers
from scipy.stats import zscore

def fit_odr_with_outlier_removal(x, y, model, beta0=None, threshold=2.5, max_iter=5):
    x = np.array(x)
    y = np.array(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    for _ in range(max_iter):
        data = odr.Data(x, y)
        odr_obj = odr.ODR(data, model, beta0=beta0)
        output = odr_obj.run()

        # Handle odr.polynomial (beta[0] + beta[1]*x + beta[2]*x^2 + ...)
        y_pred = np.polyval(output.beta[::-1], x)

        residuals = y - y_pred
        z = np.abs(zscore(residuals))

        new_mask = z < threshold
        if np.all(new_mask):
            break

        x = x[new_mask]
        y = y[new_mask]

    return output, x, y



threshold = 2.5

hindcast_Fphat_short = hindcast_Fphat[val_info.cleaned_years>=min_years_strong]
hindcast_Fphat_exp_short = hindcast_Fphat_exp[val_info.cleaned_years>=min_years_strong]
new_df_short = new_df[val_info.cleaned_years>=min_years_strong]

variables = ["kappa","b","lambda","a"]
for vari in variables: 
    varis = [f"{vari}1",f"{vari}2",f"{vari}1_0",f"{vari}2_0"]
    df_small = hindcast_Fphat[varis]
    
    
    df_small_exp = hindcast_Fphat_exp[[f"{vari}1",f"{vari}2"]]
    
    corr_table = df_small.corr()
    corr_table_exp = df_small_exp.corr()
    
    
    poly_model = odr.polynomial(1)  # using first order polynomial model
    output, x_fit, y = fit_odr_with_outlier_removal(
        hindcast_Fphat_short[f"{vari}1"],
        hindcast_Fphat_short[f"{vari}2"],
        poly_model,
        beta0=[0, 0],
        threshold = threshold        )
    poly = np.poly1d(output.beta[::-1])
    poly_y = poly(x_fit)
    
    
    poly_model = odr.polynomial(1)  # using first order polynomial model
    data = odr.Data(hindcast_Fphat_short[f"{vari}1"].dropna(),hindcast_Fphat_short[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y2 = poly(hindcast_Fphat_short[f"{vari}1"].dropna())
    
    poly_model = odr.polynomial(1)  # using first order polynomial model
    output, x_fit_exp, y_exp = fit_odr_with_outlier_removal(
        hindcast_Fphat_exp_short[f"{vari}1"],
        hindcast_Fphat_exp_short[f"{vari}2"],
        poly_model,
        beta0=[0, 0],
        threshold = threshold
        )
    poly = np.poly1d(output.beta[::-1])
    poly_y_exp = poly(x_fit_exp)
    
    
    data = odr.Data(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),hindcast_Fphat_exp_short[f"{vari}2"].dropna())
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()  # running ODR fitting
    poly = np.poly1d(output.beta[::-1])
    poly_y_exp2 = poly(hindcast_Fphat_exp_short[f"{vari}1"].dropna())
    
    
    
    if vari != "b":    
        
        poly_model = odr.polynomial(1)  # using first order polynomial model
        output, x_fit_0, y_0 = fit_odr_with_outlier_removal(
            hindcast_Fphat_short[f"{vari}1_0"],
            hindcast_Fphat_short[f"{vari}2_0"],
            poly_model,
            beta0=[0, 0],
            threshold = threshold        )
        poly = np.poly1d(output.beta[::-1])
        poly_y_0 = poly(x_fit_0)
        
        
        data = odr.Data(hindcast_Fphat_short[f"{vari}1_0"].dropna(),hindcast_Fphat_short[f"{vari}2_0"].dropna())
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()  # running ODR fitting
        poly = np.poly1d(output.beta[::-1])
        poly_y_02 = poly(hindcast_Fphat_short[f"{vari}1_0"].dropna())
        
        
    
    
    
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(1,3,1)
    sc = ax1.scatter(hindcast_Fphat_short[f"{vari}1"],hindcast_Fphat_short[f"{vari}2"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    
    
    ax1.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax1.plot(x_fit,poly_y,label = "best fit, outliers removed",color = "r")
    ax1.plot(hindcast_Fphat_short[f"{vari}1"].dropna(),poly_y2,label = "best fit")
    ax1.set_xlabel(f"{vari}1")
    ax1.set_ylabel(f"{vari}2")
    ax1.set_title(f"free b. corr = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}")
    plt.legend()
    
    ax2 = fig.add_subplot(1,3,2)
    sc = ax2.scatter(hindcast_Fphat_short[f"{vari}1_0"],hindcast_Fphat_short[f"{vari}2_0"],
                s=3,c = hindcast_Fphat_short.pvals,
                norm = norm, cmap = cmap)
    ax2.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    
    if vari != "b": 
        ax2.plot(x_fit_0,poly_y_0,label = "best fit, outliers removed",color = "r")
        ax2.plot(hindcast_Fphat_short[f"{vari}1_0"].dropna(),poly_y_02,label = "best fit")
        
    ax2.set_xlabel(f"{vari}1_0")
    ax2.set_ylabel(f"{vari}2_0")
    ax2.set_title(f"b = 0. corr = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}")
    
    ax3 = fig.add_subplot(1,3,3)
    sc = ax3.scatter(hindcast_Fphat_exp_short[f"{vari}1"],hindcast_Fphat_exp_short[f"{vari}2"],
                s=3,c = hindcast_Fphat_exp_short.pvals,
                norm = norm, cmap = cmap)#, marker = "*" if val_info.cleaned_years>=30 else ".")
    
    ax3.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax3.plot(x_fit_exp,poly_y_exp,label = "best fit, outliers remove",color = "r")
    ax3.plot(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),poly_y_exp2,label = "best fit")
    ax3.set_xlabel(f"{vari}1")
    ax3.set_ylabel(f"{vari}2")
    ax3.set_title(f"free b exponential. corr = {corr_table_exp[f"{vari}1"][f"{vari}2"]:.2f}")
    
    
    cbar_ax = fig.add_subplot([0.15, -0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('p-value', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.suptitle(f"{country_save}. years longer than {min_years_strong}")
    plt.tight_layout()
    plt.show()
    
    
    
    fig = plt.figure(figsize = (12,5))
    ax1 = fig.add_subplot(1,3,1)
    sc = ax1.scatter(x_fit,y,
                s=3)
    
    ax1.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax1.plot(x_fit,poly_y,label = "best fit, outliers removed",color = "r")
    #ax1.plot(hindcast_Fphat_short[f"{vari}1"].dropna(),poly_y2,label = "best fit")
    ax1.set_xlabel(f"{vari}1")
    ax1.set_ylabel(f"{vari}2")
    ax1.set_title(f"free b. corr = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}")
    plt.legend()
    
    ax2 = fig.add_subplot(1,3,2)
    
    if vari != "b": 
        sc = ax2.scatter(x_fit_0,y_0,
                    s=3)
    ax2.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    
    if vari != "b":    
        ax2.plot(x_fit_0,poly_y_0,label = "best fit, outliers removed",color = "r")
        
    ax2.set_xlabel(f"{vari}1_0")
    ax2.set_ylabel(f"{vari}2_0")
    ax2.set_title(f"b = 0. corr = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}")
    
    ax3 = fig.add_subplot(1,3,3)
    sc = ax3.scatter(x_fit_exp,y_exp,
                s=3)#, marker = "*" if val_info.cleaned_years>=30 else ".")
    
    ax3.plot([np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],[np.min(hindcast_Fphat_short[f"{vari}1"]),np.max(hindcast_Fphat_short[f"{vari}1"])*1.1],label = "line of equality",linestyle = "--")
    ax3.plot(x_fit_exp,poly_y_exp,label = "best fit, outliers remove",color = "r")
    #ax3.plot(hindcast_Fphat_exp_short[f"{vari}1"].dropna(),poly_y_exp2,label = "best fit")
    ax3.set_xlabel(f"{vari}1")
    ax3.set_ylabel(f"{vari}2")
    ax3.set_title(f"free b exponential. corr = {corr_table_exp[f"{vari}1"][f"{vari}2"]:.2f}")
    
    
    plt.suptitle(f"{country_save}. years longer than {min_years_strong} outliers removed")
    plt.tight_layout()
    plt.show()



















