# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 07:34:54 2024

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
import matplotlib.colors as mcolors
import glob


from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning

# Suppress IterationLimitWarning
warnings.simplefilter("ignore", IterationLimitWarning)

drive='F' #name of drive
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet

name_col = 'ppt'
temp_name_col = 't2m'

maxlat = 30
maxlon = -79

US_main_loc = np.array([24, -125, 56, -66])
J_loc = np.array([24, 122.9, 45.6, 145.8]) 
H_loc = np.array([18.8,-160,22.3,-154.8])
PR_loc = np.array([17.6,-67.3,18.5,-64.7])
Is_loc = np.array([27, 34, 34, 36])
F_loc = np.array([25, -83, 31, -78])




countries = ['Japan','US'] #Puerto RIco
n_stations = 5

info = []

for c in countries:
    dd = pd.read_csv(drive+':/metadata/'+c+'_fulldata.csv', dtype={'station': str})
    dd.startdate = pd.to_datetime(dd.startdate)
    dd.enddate = pd.to_datetime(dd.enddate)
    dd = dd[dd['startdate']>=min_startdate]
    info.append(dd)
    
info_full = pd.concat(info,axis=0)    




info_tropics_J = info[0][info[0].latitude<maxlat]
info_tropics_J = info_tropics_J[info_tropics_J.cleaned_years>=20]

info_tropics_US = info[1][info[1].latitude<maxlat]
info_tropics_US = info_tropics_US[info_tropics_US.longitude<maxlon]
info_tropics_US = info_tropics_US[info_tropics_US.cleaned_years>=20]

info_hawaii = info_tropics_US[info_tropics_US.latitude>H_loc[0]]
info_hawaii = info_hawaii[info_hawaii.latitude<H_loc[2]]
info_hawaii = info_hawaii[info_hawaii.longitude>H_loc[1]]
info_hawaii = info_hawaii[info_hawaii.longitude<H_loc[3]]


info_florida = info_tropics_US[info_tropics_US.latitude>F_loc[0]]
info_florida = info_florida[info_florida.latitude<F_loc[2]]
info_florida = info_florida[info_florida.longitude>F_loc[1]]
info_florida = info_florida[info_florida.longitude<F_loc[3]]


J_sort = info_tropics_J.sort_values(by=['cleaned_years'],ascending=0) #sort by size so can choose top sizes

H_sort = info_hawaii.sort_values(by=['cleaned_years'],ascending=0) #sort by size so can choose top sizes

F_sort = info_florida.sort_values(by=['cleaned_years'],ascending=0)

J_selected = J_sort[0:n_stations]
H_selected = H_sort[0:n_stations]
F_selected = F_sort[0:n_stations]


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()

ax1.scatter(J_selected.longitude,J_selected.latitude, color = 'red')

plt.xlim(J_loc[1]-2,J_loc[3]+2)
plt.ylim(J_loc[0]-2,J_loc[2]+2)
plt.show()


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()

ax1.scatter(H_selected.longitude,H_selected.latitude, color = 'red')

plt.xlim(H_loc[1]-2,H_loc[3]+2)
plt.ylim(H_loc[0]-2,H_loc[2]+2)
plt.show()



fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()

ax1.scatter(F_selected.longitude,F_selected.latitude, color = 'red')

plt.xlim(F_loc[1]-2,F_loc[3]+2)
plt.ylim(F_loc[0]-2,F_loc[2]+2)
plt.show()


#FLORIDA
saved_files_US = glob.glob(drive+':/US_temp/*') #temp files already saved
T_files_US = sorted(glob.glob(drive+':/ERA5_land/US*/*'))
nans = xr.open_dataarray(T_files_US[0])[0] 
nans = np.invert(np.isnan(nans)).astype(int)
g_phats = [0]*n_stations
F_phats = [0]*n_stations
thr = [0]*n_stations
ns = [0]*n_stations

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.9],
        alpha = 0.05,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )


for i in np.arange(0,n_stations):
    code = F_selected.station[F_selected.index[i]]
    path = f"{drive}:/US/US_{code}.txt"
    G,data_meta = read_GSDR_file(path,name_col)
    
    target_lat = F_selected.latitude[F_selected.index[i]] #define targets
    target_lon = F_selected.longitude[F_selected.index[i]]
    start_date = F_selected.startdate[F_selected.index[i]]-pd.Timedelta(days=1) #adding an extra day either end to be safe
    end_date = F_selected.enddate[F_selected.index[i]]+pd.Timedelta(days=1)
    
    save_path = f"{drive}:/US_temp\\US_{code}.nc"
    if save_path not in saved_files_US:
        print(f'file {save_path} not made yet')
        T_ERA = make_T_timeseries(target_lat,target_lon,start_date,end_date,T_files_US,nans)
        if len(T_ERA) == 0:
            print('skip')
        else:
            T_ERA.to_netcdf(save_path) #save file with same name format as GSDR
            saved_files_US.append(save_path)  # Update saved_files to include the newly saved file
    else:
        print(f"File {save_path} already exists. Skipping save.")
        T_ERA = xr.load_dataarray(save_path)
    
    
    data = G 
    data = S.remove_incomplete_years(data, name_col)
    t_data = (T_ERA.squeeze()-273.15).to_dataframe()
    print(f'{data_meta.latitude},{data_meta.longitude}')
    print(t_data[0:5])
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
    
    
    
    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() 
    T = dict_ordinary["60"]["T"].to_numpy()  
    
    
    # Number of threshold 
    thr[i] = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
    
    
    #TENAX MODEL HERE
    #magnitude model
    F_phats[i], loglik, _, _ = S.magnitude_model(P, T, thr[i])
    #temperature model
    g_phats[i] = S.temperature_model(T)
    # M is mean n of ordinary events
    ns[i] = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    
    
    eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phats[i],thr[i],eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f}) \n κ_0 = {F_phats[i][0]:.3f}, b = {F_phats[i][1]:.3f}, λ_0 = {F_phats[i][2]:.3f}, a = {F_phats[i][3]:.3f}')
    plt.show()
    
    #recalculate for non zero bs forcing to zero
    if F_phats[i][1] != 0:
        F_phat2, __, _, _ = S2.magnitude_model(P, T, thr[i])
        TNX_FIG_magn_model(P,T,F_phats[i],thr[i],eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
        
        #plot new
        percentile_lines = inverse_magnitude_model(F_phat2,eT,qs)
        i=0
        
        plt.plot(eT,percentile_lines[n],'--b',alpha = 0.6, label = "Magnitude model, b=0")
        i=1
        while n<np.size(qs):
            plt.plot(eT,percentile_lines[n],'--b',alpha = 0.6) 
            i=i+1    
        
        plt.ylabel('60-minute precipitation (mm)')
        plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f}) \n κ_0 = {F_phats[i][0]:.3f}, b = {F_phats[i][1]:.3f}, λ_0 = {F_phats[i][2]:.3f}, a = {F_phats[i][3]:.3f}')
        plt.show()
    
    ##############################################################################
    TNX_FIG_temp_model(T, g_phats[i],4,eT,xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,5/(np.max(T)-np.min(T))])
    plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f}) \n μ = {g_phats[i][0]:.1f}, σ = {g_phats[i][1]:.1f}')
    plt.show()
    
    
    Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
    iTs = np.arange(-2.5,37.5,1.5) #idk why we need a different T range here 
    S.n_monte_carlo = np.size(P)*S.niter_smev
    _, T_mc, P_mc = S.model_inversion(F_phats[i], g_phats[i], ns[i], Ts,gen_P_mc = True,gen_RL=False) 
    
    
    scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phats[i],S.niter_smev,eT,iTs,xlimits = [np.min(T)-3,np.max(T)+3])
    plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f}) \n κ_0 = {F_phats[i][0]:.3f}, b = {F_phats[i][1]:.3f}, λ_0 = {F_phats[i][2]:.3f}, a = {F_phats[i][3]:.3f}')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    
    season_separations = [5, 10]
    months = dict_ordinary["60"]["oe_time"].dt.month
    winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
    summer_inds = months.index[(months<season_separations[1]+1)&(months>season_separations[0]-1)]
    T_winter = T[winter_inds]
    T_summer = T[summer_inds]


    g_phat_winter = S.temperature_model(T_winter,beta = 2)
    g_phat_summer = S.temperature_model(T_summer,beta = 2)


    winter_pdf = gen_norm_pdf(eT, g_phat_winter[0], g_phat_winter[1], 2)
    summer_pdf = gen_norm_pdf(eT, g_phat_summer[0], g_phat_summer[1], 2)

    combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))


    #fig 3


    TNX_FIG_temp_model(T=T_summer, g_phat=g_phat_summer,beta=2,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Summer')
    TNX_FIG_temp_model(T=T_winter, g_phat=g_phat_winter,beta=2,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Winter')
    TNX_FIG_temp_model(T=T, g_phat=g_phats[i],beta=4,eT=eT,obscol='k',valcol='k',obslabel = None,vallabel = 'Annual',xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,7/(np.max(T)-np.min(T))])
    plt.plot(eT,combined_pdf,'m',label = 'Combined summer and winter')
    plt.legend()
    plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f}) \n μ_s = {g_phat_summer[0]:.1f}, σ_s = {g_phat_summer[1]:.1f},μ_w = {g_phat_winter[0]:.1f}, σ_w = {g_phat_winter[1]:.1f}')
    plt.show()
    
    #fig 4 
    S.n_monte_carlo = 20000 # set number of MC for getting RL
    RL, _, P_check = S.model_inversion(F_phats[i], g_phats[i], ns[i], Ts) 
    AMS = dict_AMS['60'] # yet the annual maxima
    TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+10])
    plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f})')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()

    
    
    #TENAX MODEL VALIDATION
    yrs = dict_ordinary["60"]["oe_time"].dt.year
    yrs_unique = np.unique(yrs)
    midway = yrs_unique[int(np.ceil(np.size(yrs_unique)/2))-1] # -1 to adjust indexing because this returns a sort of length

    #DEFINE FIRST PERIOD
    P1 = P[yrs<=midway]
    T1 = T[yrs<=midway]
    AMS1 = AMS[AMS['year']<=midway]
    n_ordinary_per_year1 = n_ordinary_per_year[n_ordinary_per_year.index<=midway]
    n1 = n_ordinary_per_year1.sum() / len(n_ordinary_per_year1)

    #DEFINE SECOND PERIOD
    P2 = P[yrs>midway]
    T2 = T[yrs>midway]
    AMS2 = AMS[AMS['year']>midway]
    n_ordinary_per_year2 = n_ordinary_per_year[n_ordinary_per_year.index>midway]
    n2 = n_ordinary_per_year2.sum() / len(n_ordinary_per_year2)


    g_phat1 = S.temperature_model(T1)
    g_phat2 = S.temperature_model(T2)


    F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr[i])
    RL1, _, _ = S.model_inversion(F_phat1, g_phat1, n1, Ts)
       

    F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr[i])
    RL2, _, _ = S.model_inversion(F_phat2, g_phat2, n2, Ts)   

    if F_phats[i][1]==0: #check if b parameter is 0 (shape=shape_0*b
        dof=3
        alpha1=1; # b parameter is not significantly different from 0; 3 degrees of freedom for the LR test
    else: 
        dof=4
        alpha1=0  # b parameter is significantly different from 0; 4 degrees of freedom for the LR test




    #check magnitude model the same in both periods
    lambda_LR = -2*( loglik - (loglik1+loglik2) )
    pval = chi2.sf(lambda_LR, dof)
    if pval > S.alpha:
        print(f"p={pval}. Magnitude models not  different at {S.alpha*100}% significance.")
    else:
        print(f"p={pval}. Magnitude models are different at {S.alpha*100}% significance.")

    #modelling second model based on first magnitude and changes in mean/std
    mu_delta = np.mean(T2)-np.mean(T1)
    sigma_factor = np.std(T2)/np.std(T1)

    g_phat2_predict = [g_phat1[0]+mu_delta, g_phat1[1]*sigma_factor]
    RL2_predict, _,_ = S.model_inversion(F_phat1,g_phat2_predict,n2,Ts)


    #fig 7a

    TNX_FIG_temp_model(T=T1, g_phat=g_phat1,beta=4,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Temperature model '+str(yrs_unique[0])+'-'+str(midway))
    TNX_FIG_temp_model(T=T2, g_phat=g_phat2_predict,beta=4,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Temperature model '+str(midway+1)+'-'+str(yrs_unique[-1]),xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,5/(np.max(T)-np.min(T))]) # model based on temp ave and std changes
    plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f})')
    plt.show() #this is slightly different in code and paper I think.. using predicted T vs fitted T

    #fig 7b

    TNX_FIG_valid(AMS1,S.return_period,RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'The TENAX model '+str(yrs_unique[0])+'-'+str(midway),obslabel='Observed annual maxima '+str(yrs_unique[0])+'-'+str(midway),ylimits = [0,np.max(AMS.AMS)+10])
    TNX_FIG_valid(AMS2,S.return_period,RL2_predict,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'The predicted TENAX model '+str(midway+1)+'-'+str(yrs_unique[-1]),obslabel='Observed annual maxima '+str(midway+1)+'-'+str(yrs_unique[-1]),ylimits = [0,np.max(AMS.AMS)+10])
    plt.title(f'Florida. ({F_selected.latitude.iloc[i]:.1f},{F_selected.longitude.iloc[i]:.1f})')
    plt.show()
    
    
    
#HAWAII

g_phats = [0]*n_stations
F_phats = [0]*n_stations
thr = [0]*n_stations
ns = [0]*n_stations

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.9],
        alpha = 0.05,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )


for i in np.arange(0,n_stations):
    code = H_selected.station[H_selected.index[i]]
    path = f"{drive}:/US/US_{code}.txt"
    G,data_meta = read_GSDR_file(path,name_col)
    
    target_lat = H_selected.latitude[H_selected.index[i]] #define targets
    target_lon = H_selected.longitude[H_selected.index[i]]
    start_date = H_selected.startdate[H_selected.index[i]]-pd.Timedelta(days=1) #adding an extra day either end to be safe
    end_date = H_selected.enddate[H_selected.index[i]]+pd.Timedelta(days=1)
    
    save_path = f"{drive}:/US_temp\\US_{code}.nc"
    if save_path not in saved_files_US:
        print(f'file {save_path} not made yet')
        T_ERA = make_T_timeseries(target_lat,target_lon,start_date,end_date,T_files_US,nans)
        if len(T_ERA) == 0:
            print('skip')
        else:
            T_ERA.to_netcdf(save_path) #save file with same name format as GSDR
            saved_files_US.append(save_path)  # Update saved_files to include the newly saved file
    else:
        print(f"File {save_path} already exists. Skipping save.")
        T_ERA = xr.load_dataarray(save_path)
    
    
    data = G 
    data = S.remove_incomplete_years(data, name_col)
    t_data = (T_ERA.squeeze()-273.15).to_dataframe()
    print(f'{data_meta.latitude},{data_meta.longitude}')
    print(t_data[0:5])
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
    
    
    
    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() 
    T = dict_ordinary["60"]["T"].to_numpy()  
    
    
    # Number of threshold 
    thr[i] = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
    
    
    #TENAX MODEL HERE
    #magnitude model
    F_phats[i], loglik, _, _ = S.magnitude_model(P, T, thr[i])
    #temperature model
    g_phats[i] = S.temperature_model(T)
    # M is mean n of ordinary events
    ns[i] = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    
    
    eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phats[i],thr[i],eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f}) \n κ_0 = {F_phats[i][0]:.3f}, b = {F_phats[i][1]:.3f}, λ_0 = {F_phats[i][2]:.3f}, a = {F_phats[i][3]:.3f}')
    plt.show()
    
    #recalculate for non zero bs forcing to zero
    if F_phats[i][1] != 0:
        F_phat2, __, _, _ = S2.magnitude_model(P, T, thr[i])
        TNX_FIG_magn_model(P,T,F_phats[i],thr[i],eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
        
        #plot new
        percentile_lines = inverse_magnitude_model(F_phat2,eT,qs)
        i=0
        
        plt.plot(eT,percentile_lines[n],'--b',alpha = 0.6, label = "Magnitude model, b=0")
        i=1
        while n<np.size(qs):
            plt.plot(eT,percentile_lines[n],'--b',alpha = 0.6) 
            i=i+1    
        
        plt.ylabel('60-minute precipitation (mm)')
        plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f}) \n κ_0 = {F_phats[i][0]:.3f}, b = {F_phats[i][1]:.3f}, λ_0 = {F_phats[i][2]:.3f}, a = {F_phats[i][3]:.3f}')
        plt.show()
    
    ##############################################################################
    TNX_FIG_temp_model(T, g_phats[i],4,eT,xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,5/(np.max(T)-np.min(T))])
    plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f}) \n μ = {g_phats[i][0]:.1f}, σ = {g_phats[i][1]:.1f}')
    plt.show()
    
    
    Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
    iTs = np.arange(-2.5,37.5,1.5) #idk why we need a different T range here 
    S.n_monte_carlo = np.size(P)*S.niter_smev
    _, T_mc, P_mc = S.model_inversion(F_phats[i], g_phats[i], ns[i], Ts,gen_P_mc = True,gen_RL=False) 
    
    
    scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phats[i],S.niter_smev,eT,iTs,xlimits = [np.min(T)-3,np.max(T)+3])
    plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f}) \n κ_0 = {F_phats[i][0]:.3f}, b = {F_phats[i][1]:.3f}, λ_0 = {F_phats[i][2]:.3f}, a = {F_phats[i][3]:.3f}')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    
    season_separations = [5, 10]
    months = dict_ordinary["60"]["oe_time"].dt.month
    winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
    summer_inds = months.index[(months<season_separations[1]+1)&(months>season_separations[0]-1)]
    T_winter = T[winter_inds]
    T_summer = T[summer_inds]


    g_phat_winter = S.temperature_model(T_winter,beta = 2)
    g_phat_summer = S.temperature_model(T_summer,beta = 2)


    winter_pdf = gen_norm_pdf(eT, g_phat_winter[0], g_phat_winter[1], 2)
    summer_pdf = gen_norm_pdf(eT, g_phat_summer[0], g_phat_summer[1], 2)

    combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))


    #fig 3


    TNX_FIG_temp_model(T=T_summer, g_phat=g_phat_summer,beta=2,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Summer')
    TNX_FIG_temp_model(T=T_winter, g_phat=g_phat_winter,beta=2,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Winter')
    TNX_FIG_temp_model(T=T, g_phat=g_phats[i],beta=4,eT=eT,obscol='k',valcol='k',obslabel = None,vallabel = 'Annual',xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,7/(np.max(T)-np.min(T))])
    plt.plot(eT,combined_pdf,'m',label = 'Combined summer and winter')
    plt.legend()
    plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f}) \n μ_s = {g_phat_summer[0]:.1f}, σ_s = {g_phat_summer[1]:.1f},μ_w = {g_phat_winter[0]:.1f}, σ_w = {g_phat_winter[1]:.1f}')
    plt.show()
    
    
    #fig 4 
    S.n_monte_carlo = 20000 # set number of MC for getting RL
    RL, _, P_check = S.model_inversion(F_phats[i], g_phats[i], ns[i], Ts) 
    AMS = dict_AMS['60'] # yet the annual maxima
    TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+10])
    plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f})')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()

    
    
    #TENAX MODEL VALIDATION
    yrs = dict_ordinary["60"]["oe_time"].dt.year
    yrs_unique = np.unique(yrs)
    midway = yrs_unique[int(np.ceil(np.size(yrs_unique)/2))-1] # -1 to adjust indexing because this returns a sort of length

    #DEFINE FIRST PERIOD
    P1 = P[yrs<=midway]
    T1 = T[yrs<=midway]
    AMS1 = AMS[AMS['year']<=midway]
    n_ordinary_per_year1 = n_ordinary_per_year[n_ordinary_per_year.index<=midway]
    n1 = n_ordinary_per_year1.sum() / len(n_ordinary_per_year1)

    #DEFINE SECOND PERIOD
    P2 = P[yrs>midway]
    T2 = T[yrs>midway]
    AMS2 = AMS[AMS['year']>midway]
    n_ordinary_per_year2 = n_ordinary_per_year[n_ordinary_per_year.index>midway]
    n2 = n_ordinary_per_year2.sum() / len(n_ordinary_per_year2)


    g_phat1 = S.temperature_model(T1)
    g_phat2 = S.temperature_model(T2)


    F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr[i])
    RL1, _, _ = S.model_inversion(F_phat1, g_phat1, n1, Ts)
       

    F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr[i])
    RL2, _, _ = S.model_inversion(F_phat2, g_phat2, n2, Ts)   

    if F_phats[i][1]==0: #check if b parameter is 0 (shape=shape_0*b
        dof=3
        alpha1=1; # b parameter is not significantly different from 0; 3 degrees of freedom for the LR test
    else: 
        dof=4
        alpha1=0  # b parameter is significantly different from 0; 4 degrees of freedom for the LR test




    #check magnitude model the same in both periods
    lambda_LR = -2*( loglik - (loglik1+loglik2) )
    pval = chi2.sf(lambda_LR, dof)
    if pval > S.alpha:
        print(f"p={pval}. Magnitude models not  different at {S.alpha*100}% significance.")
    else:
        print(f"p={pval}. Magnitude models are different at {S.alpha*100}% significance.")

    #modelling second model based on first magnitude and changes in mean/std
    mu_delta = np.mean(T2)-np.mean(T1)
    sigma_factor = np.std(T2)/np.std(T1)

    g_phat2_predict = [g_phat1[0]+mu_delta, g_phat1[1]*sigma_factor]
    RL2_predict, _,_ = S.model_inversion(F_phat1,g_phat2_predict,n2,Ts)


    #fig 7a

    TNX_FIG_temp_model(T=T1, g_phat=g_phat1,beta=4,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Temperature model '+str(yrs_unique[0])+'-'+str(midway))
    TNX_FIG_temp_model(T=T2, g_phat=g_phat2_predict,beta=4,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Temperature model '+str(midway+1)+'-'+str(yrs_unique[-1]),xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,5/(np.max(T)-np.min(T))]) # model based on temp ave and std changes
    plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f})')
    plt.show() #this is slightly different in code and paper I think.. using predicted T vs fitted T

    #fig 7b

    TNX_FIG_valid(AMS1,S.return_period,RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'The TENAX model '+str(yrs_unique[0])+'-'+str(midway),obslabel='Observed annual maxima '+str(yrs_unique[0])+'-'+str(midway))
    TNX_FIG_valid(AMS2,S.return_period,RL2_predict,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'The predicted TENAX model '+str(midway+1)+'-'+str(yrs_unique[-1]),obslabel='Observed annual maxima '+str(midway+1)+'-'+str(yrs_unique[-1]),ylimits = [0,np.max(AMS.AMS)+10])
    plt.title(f'Hawaii. ({H_selected.latitude.iloc[i]:.1f},{H_selected.longitude.iloc[i]:.1f})')
    plt.show()



