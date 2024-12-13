# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:48:33 2024

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

import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning

# Suppress IterationLimitWarning
warnings.simplefilter("ignore", IterationLimitWarning)


drive = 'D'
countries = ['Belgium','Germany','Japan','UK'] #'ISD','Finland','US','Norway','Portugal','Ireland'

min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
min_yrs = 10 

#initializing
df_parameters_lst = []
TENAX_uses = []
df_parameters_neg_lst = []
val_infos = []
full_infos = []

for country in countries:
    #load parameters (b etc)
    df_savename = drive + ':/outputs/'+country+'\\parameters.csv'
    df_parameters = pd.read_csv(df_savename)
    df_parameters_lst.append(df_parameters)
    
    
    #load TENAX parameters (alpha etc)
    TENAX_use = pd.read_csv(drive + ':/outputs/'+country+'/TENAX_parameters.csv')
    TENAX_uses.append(TENAX_use)
    
    #load extra bs
    save_path_neg = drive + ':/outputs/'+country+'\\parameters_neg.csv'
    df_parameters_neg = pd.read_csv(save_path_neg)
    df_parameters_neg_lst.append(df_parameters_neg)
    
    #load info
    info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
    info.startdate = pd.to_datetime(info.startdate)
    info.enddate = pd.to_datetime(info.enddate)

    val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min
    val_info = val_info[val_info['startdate']>=min_startdate]
    
    val_infos.append(val_info)
    full_infos.append(info)
    
    lon_lims = [np.trunc(np.min(df_parameters.longitude/2.5))*2.5,np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
    lat_lims = [np.trunc(np.min(df_parameters.latitude/2.5))*2.5,np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
    
    new_df = df_parameters[['latitude','longitude','b']].copy()
    mask = new_df['b'] == 0

    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    s=3
    ####################################################
    #plot at 5% sig
    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')

    # Choosing cmap
    norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())

    for lon, lat in zip(df_parameters.longitude[df_parameters.thr == 0], 
                        df_parameters.latitude[df_parameters.thr == 0]):
        square = patches.Rectangle(
            (lon - 0.5, lat - 0.5),  # Bottom-left corner of the square
            1,  # Width (1 degree)
            1,  # Height (1 degree)
            color='r',
            alpha = 0.2,
            label="no ERA data within 1 deg" if 'no ERA data within 1 deg' not in ax1.get_legend_handles_labels()[1] else ""
        )
        ax1.add_patch(square)
    sc = ax1.scatter(
        df_parameters.longitude[df_parameters.b==0],
        df_parameters.latitude[df_parameters.b==0],
        s = s,
        color = 'darkgrey',  
    )

    sc = ax1.scatter(
        df_parameters.longitude[df_parameters.b!=0],
        df_parameters.latitude[df_parameters.b!=0],
        c=df_parameters.b[df_parameters.b!=0],
        s = s,
        cmap='seismic',  
        norm=norm
    )

    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
    cb.set_label('b', fontsize=14)  
    cb.ax.tick_params(labelsize=12)

    # Set x and y ticks
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

    plt.title(f'GSDR: {country}. b at {TENAX_use.alpha[0]} sig level', fontsize=16)
    plt.legend()
    plt.show()

    ############################################

    ## PLOT ALL b 2
    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')

    # Choosing cmap
    norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())

    sc = ax1.scatter( #plot the negligable at 5% lvl points
        new_df.longitude,
        new_df.latitude,
        c = new_df.b,
        s = s,
        cmap = 'seismic',
        norm = norm
    )

    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
    cb.set_label('b', fontsize=14)  
    cb.ax.tick_params(labelsize=12)

    # Set x and y ticks
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


    plt.title(f'GSDR: {country}. b at 0 sig level', fontsize=16)
    plt.show()
    #THIS SHOWS THE LOCATION OF THE STATION, NOT THE ERA DATA

    ###############################################################
    #Average temps
    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax1.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=df_parameters.mu,
        s = s,
        cmap='Reds',  
    )




    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
    cb.set_label('μ (°C)', fontsize=14)  
    cb.ax.tick_params(labelsize=12)

    # Set x and y ticks
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


    plt.title(f'GSDR: {country}. μ', fontsize=16)
    plt.show()





###############################################################################

#redoing UK because different startdate
min_startdate = dt.datetime(1981,1,1)

val_info = val_info[val_info['startdate']>=min_startdate]

val_infos[3] = val_info




#Plot the data lengths
for n in np.arange(1,len(countries)):
    val_info = val_infos[n]
    country = countries[n]
    
    lon_lims = [np.trunc(np.min(val_info.longitude/2.5))*2.5,np.ceil(np.max(val_info.longitude/2.5))*2.5]
    lat_lims = [np.trunc(np.min(val_info.latitude/2.5))*2.5,np.ceil(np.max(val_info.latitude/2.5))*2.5]
        
    
    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax1.scatter(
        val_info.longitude,
        val_info.latitude,
        c=val_info.cleaned_years,
        s = s,
        cmap='inferno_r',  
    )


    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
    cb.set_label('Years', fontsize=14)  
    cb.ax.tick_params(labelsize=12)

    # Set x and y ticks
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


    plt.title(f'GSDR: {country}. Cleaned years', fontsize=16)
    plt.show()


germany_comb = df_parameters_lst[1]
germany_comb['cleaned_years'] = full_infos[1]['cleaned_years']
germany_comb.station = germany_comb['station'].apply(lambda x: f'{int(x):0{5}}')



japan_comb = df_parameters_lst[2]
japan_comb['cleaned_years'] = full_infos[2]['cleaned_years']
japan_comb.station = japan_comb['station'].apply(lambda x: f'{int(x):0{5}}')

UK_comb = df_parameters_lst[3]
UK_comb['cleaned_years'] = full_infos[3]['cleaned_years']


germany_20 = germany_comb[germany_comb.cleaned_years>=20]
japan_20 = japan_comb[japan_comb.cleaned_years>=20]
UK_20 = UK_comb[UK_comb.cleaned_years>=20]

germany_pos = germany_20[germany_20.b>0]
japan_pos = japan_20[japan_20.b>0]
UK_pos = UK_20[UK_20.b>0]

germany_neg = germany_20[germany_20.b<0]
japan_neg = japan_20[japan_20.b<0]
UK_neg = UK_20[UK_20.b<0]

germany_0 = germany_20[germany_20.b==0]
japan_0 = japan_20[japan_20.b==0]
UK_0 = UK_20[UK_20.b==0]



datasets = [germany_pos,germany_0,germany_neg,japan_pos,japan_0,japan_neg,UK_pos,UK_0,UK_neg]
cmaps = ['autumn_r',None,'Blues']
val_info_new = val_infos[1:]
titles = ['germany_pos','germany_0','germany_neg','japan_pos','japan_0','japan_neg','UK_pos','UK_0','UK_neg']

for n in np.arange(0,9):
    val_info = val_info_new[n//3]
    daset = datasets[n]
    
    lon_lims = [np.trunc(np.min(val_info.longitude/2.5))*2.5,np.ceil(np.max(val_info.longitude/2.5))*2.5]
    lat_lims = [np.trunc(np.min(val_info.latitude/2.5))*2.5,np.ceil(np.max(val_info.latitude/2.5))*2.5]
        
    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax1.scatter(
        daset.longitude,
        daset.latitude,
        c=daset.b,
        cmap=cmaps[n%3],  
    )
    
    
    # Add a colorbar at the bottom
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
    cb.set_label('b', fontsize=14)  
    cb.ax.tick_params(labelsize=12)
    
    # Set x and y ticks
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  
    
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    
    
    plt.title(titles[n], fontsize=16)
    plt.show()



####################################################################################
# SELECT SAMPLES

germany_pos_sel = germany_pos
germany_0_sel = germany_0[4:8]
germany_neg_sel = germany_neg[1:3]

japan_pos_sel = japan_pos.iloc[list(range(0, 1)) + [-2]]
japan_0_sel = japan_0.iloc[[4] + [200]+[270]+[-4]]
japan_neg_sel = japan_neg.iloc[[4] + [195]]

UK_pos_sel = UK_pos.iloc[list(range(1, 3))]
UK_0_sel = UK_0.iloc[[2] + [15] + [20] + [21]]
UK_neg_sel = UK_neg.iloc[[0] + [2]]



###################################################################################
# PLOT SAMPLE LOCATIONS
lon_lims = [np.trunc(np.min(val_info_new[0].longitude/2.5))*2.5,np.ceil(np.max(val_info_new[0].longitude/2.5))*2.5]
lat_lims = [np.trunc(np.min(val_info_new[0].latitude/2.5))*2.5,np.ceil(np.max(val_info_new[0].latitude/2.5))*2.5]

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

plt.scatter(germany_pos_sel.longitude,germany_pos_sel.latitude,color = 'r',label = 'positive')
plt.scatter(germany_0_sel.longitude,germany_0_sel.latitude,color = 'g',label = '0')
plt.scatter(germany_neg_sel.longitude,germany_neg_sel.latitude,color = 'b',label = 'negative')

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.legend()
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

plt.show()


lon_lims = [np.trunc(np.min(val_info_new[1].longitude/2.5))*2.5,np.ceil(np.max(val_info_new[1].longitude/2.5))*2.5]
lat_lims = [np.trunc(np.min(val_info_new[1].latitude/2.5))*2.5,np.ceil(np.max(val_info_new[1].latitude/2.5))*2.5]

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

plt.scatter(japan_pos_sel.longitude,japan_pos_sel.latitude,color = 'r',label = 'positive')
plt.scatter(japan_0_sel.longitude,japan_0_sel.latitude,color = 'g',label = '0')
plt.scatter(japan_neg_sel.longitude,japan_neg_sel.latitude,color = 'b',label = 'negative')

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.legend()
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

plt.show()



lon_lims = [np.trunc(np.min(val_info_new[2].longitude/2.5))*2.5,np.ceil(np.max(val_info_new[2].longitude/2.5))*2.5]
lat_lims = [np.trunc(np.min(val_info_new[2].latitude/2.5))*2.5,np.ceil(np.max(val_info_new[2].latitude/2.5))*2.5]

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

plt.scatter(UK_pos_sel.longitude,UK_pos_sel.latitude,color = 'r',label = 'positive')
plt.scatter(UK_0_sel.longitude,UK_0_sel.latitude,color = 'g',label = '0')
plt.scatter(UK_neg_sel.longitude,UK_neg_sel.latitude,color = 'b',label = 'negative')

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.legend()
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.show()

###################################################################################
# TENAX TIME
name_col = 'ppt' 
temp_name_col = "t2m"


code_str = ['DE_']*8+['JP_']*8+['UK_']*8
countrys = ['Germany']*8+['Japan']*8+['UK']*8

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )

S2 = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 1,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )


mashed_selects = pd.concat([germany_pos_sel,germany_0_sel,germany_neg_sel,
                          japan_pos_sel,japan_0_sel,japan_neg_sel,
                          UK_pos_sel,UK_0_sel,UK_neg_sel])

for n in np.arange(0,len(mashed_selects)):
        
    first_file_code = mashed_selects.station.iloc[n]
    file_name = drive + ':/'+countrys[n]+'/'+code_str[n]+first_file_code+'.txt'
    save_path = drive + ':/'+countrys[n]+'_temp\\'+code_str[n] + first_file_code + '.nc'
    
    g_phat = [mashed_selects.mu.iloc[n],mashed_selects.sigma.iloc[n]]
    F_phat = [mashed_selects.kappa.iloc[n],mashed_selects.b.iloc[n],mashed_selects['lambda'].iloc[n],mashed_selects.a.iloc[n]]
    thr = mashed_selects.thr.iloc[n]
    
    G,data_meta = read_GSDR_file(file_name,name_col)
    
    T_ERA = xr.load_dataarray(save_path)
    
    data = G 
    data = S.remove_incomplete_years(data, name_col)
    t_data = (T_ERA.squeeze()-273.15).to_dataframe()
    df_arr = np.array(data[name_col])
    df_dates = np.array(data.index)
    
    #extract indexes of ordinary events
    #these are time-wise indexes =>returns list of np arrays with np.timeindex
    
    S.separation = 6
    idx_ordinary6=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
    arr_vals6,arr_dates6,n_ordinary_per_year6=S.remove_short(idx_ordinary6)
    
    S.separation = 24
    idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
    arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
    
    plt.plot(n_ordinary_per_year6, label = '6 hour separation')
    plt.plot(n_ordinary_per_year, label = '24 hour separation')
    plt.legend()
    plt.title(f'Ordinary events per year {countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f})')
    plt.show()
    
    
    plt.plot(data)
    plt.scatter(arr_dates[:,0],arr_vals,color = 'r')
    plt.scatter(arr_dates[:,1],arr_vals,color = 'g')
    plt.title(f'24 hr separation {countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f})')
    plt.xlim(dt.datetime(n_ordinary_per_year.index[5],7,1),dt.datetime(n_ordinary_per_year.index[5],8,1))
    plt.show()
    
    plt.plot(data)
    plt.scatter(arr_dates6[:,0],arr_vals6,color = 'r')
    plt.scatter(arr_dates6[:,1],arr_vals6,color = 'g')
    plt.title(f'6 hr separation {countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f})')
    plt.xlim(dt.datetime(n_ordinary_per_year6.index[5],7,1),dt.datetime(n_ordinary_per_year6.index[5],8,1))
    plt.show()
    
    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
    dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
    
    df_arr_t_data = np.array(t_data[temp_name_col])
    df_dates_t_data = np.array(t_data.index)
    
    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)

    
    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() 
    T = dict_ordinary["60"]["T"].to_numpy()  
    
    
    ns = n_ordinary_per_year.sum() / len(n_ordinary_per_year) 
    
    ##############################################################################
    eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f}) \n κ_0 = {F_phat[0]:.3f}, b = {F_phat[1]:.3f}, λ_0 = {F_phat[2]:.3f}, a = {F_phat[3]:.3f}')
    plt.show()
    
    #recalculate for non zero bs forcing to zero
    if F_phat[1] != 0:
        F_phat2, __, _, _ = S2.magnitude_model(P, T, thr)
        TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
        
        #plot new
        percentile_lines = inverse_magnitude_model(F_phat2,eT,qs)
        i=0
        
        plt.plot(eT,percentile_lines[i],'--b',alpha = 0.6, label = "Magnitude model, b=0")
        i=1
        while i<np.size(qs):
            plt.plot(eT,percentile_lines[i],'--b',alpha = 0.6) 
            i=i+1    
        
        plt.ylabel('60-minute precipitation (mm)')
        plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f}) \n κ_0 = {F_phat[0]:.3f}, b = {F_phat[1]:.3f}, λ_0 = {F_phat[2]:.3f}, a = {F_phat[3]:.3f}')
        plt.show()
    
    ##############################################################################
    TNX_FIG_temp_model(T, g_phat,4,eT,xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,2.5/(np.max(T)-np.min(T))])
    plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f}) \n μ = {g_phat[0]:.1f}, σ = {g_phat[1]:.1f}')
    plt.show()
    
    TNX_FIG_temp_model(T, g_phat,6,eT,xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,2.5/(np.max(T)-np.min(T))])
    plt.title(f'{countrys[n]} Beta = 6. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f}) \n μ = {g_phat[0]:.1f}, σ = {g_phat[1]:.1f}')
    plt.show()
     
    
    Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
    iTs = np.arange(-2.5,37.5,1.5) #idk why we need a different T range here 
    S.n_monte_carlo = np.size(P)*S.niter_smev
    _, T_mc, P_mc = S.model_inversion(F_phat, g_phat, ns, Ts,gen_P_mc = True,gen_RL=False) 
    
    
    scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phat,S.niter_smev,eT,iTs,xlimits = [np.min(T)-3,np.max(T)+3])
    plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f}) \n κ_0 = {F_phat[0]:.3f}, b = {F_phat[1]:.3f}, λ_0 = {F_phat[2]:.3f}, a = {F_phat[3]:.3f}')
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
    TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,obscol='k',valcol='k',obslabel = None,vallabel = 'Annual',xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,4/(np.max(T)-np.min(T))])
    plt.plot(eT,combined_pdf,'m',label = 'Combined summer and winter')
    plt.legend()
    plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[n]:.1f},{mashed_selects.longitude.iloc[n]:.1f}) \n μ_s = {g_phat_summer[0]:.1f}, σ_s = {g_phat_summer[1]:.1f},μ_w = {g_phat_winter[0]:.1f}, σ_w = {g_phat_winter[1]:.1f}')
    plt.show()


    #fig 4 
    S.n_monte_carlo = 20000 # set number of MC for getting RL
    RL, _, P_check = S.model_inversion(F_phat, g_phat, ns, Ts) 
    AMS = dict_AMS['60'] # yet the annual maxima
    TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+10])
    plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[i]:.1f},{mashed_selects.longitude.iloc[i]:.1f})')
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


    F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr)
    RL1, _, _ = S.model_inversion(F_phat1, g_phat1, n1, Ts)
       

    F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr)
    RL2, _, _ = S.model_inversion(F_phat2, g_phat2, n2, Ts)   

    if F_phat[1]==0: #check if b parameter is 0 (shape=shape_0*b
        dof=3
        alpha1=1; # b parameter is not significantly different from 0; 3 degrees of freedom for the LR test
    else: 
        dof=4
        alpha1=0  # b parameter is significantly different from 0; 4 degrees of freedom for the LR test




    
    #modelling second model based on first magnitude and changes in mean/std
    mu_delta = np.mean(T2)-np.mean(T1)
    sigma_factor = np.std(T2)/np.std(T1)

    g_phat2_predict = [g_phat1[0]+mu_delta, g_phat1[1]*sigma_factor]
    RL2_predict, _,_ = S.model_inversion(F_phat1,g_phat2_predict,n2,Ts)


    #fig 7a

    TNX_FIG_temp_model(T=T1, g_phat=g_phat1,beta=4,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Temperature model '+str(yrs_unique[0])+'-'+str(midway))
    TNX_FIG_temp_model(T=T2, g_phat=g_phat2_predict,beta=4,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Temperature model '+str(midway+1)+'-'+str(yrs_unique[-1]),xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,5/(np.max(T)-np.min(T))]) # model based on temp ave and std changes
    plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[i]:.1f},{mashed_selects.longitude.iloc[i]:.1f})')
    plt.show() #this is slightly different in code and paper I think.. using predicted T vs fitted T

    #fig 7b

    TNX_FIG_valid(AMS1,S.return_period,RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'The TENAX model '+str(yrs_unique[0])+'-'+str(midway),obslabel='Observed annual maxima '+str(yrs_unique[0])+'-'+str(midway),ylimits = [0,np.max(AMS.AMS)+10])
    TNX_FIG_valid(AMS2,S.return_period,RL2_predict,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'The predicted TENAX model '+str(midway+1)+'-'+str(yrs_unique[-1]),obslabel='Observed annual maxima '+str(midway+1)+'-'+str(yrs_unique[-1]),ylimits = [0,np.max(AMS.AMS)+10])
    plt.title(f'{countrys[n]}. ({mashed_selects.latitude.iloc[i]:.1f},{mashed_selects.longitude.iloc[i]:.1f})')
    plt.show()






