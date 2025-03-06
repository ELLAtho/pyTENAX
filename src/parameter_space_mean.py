# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:56:48 2025

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



drive = 'D'
alpha_set = 0

radius = 50 #radius in km



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

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = alpha_set,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )

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

# for some reason in germany there is one less row...
    



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



#get matrix of the distances between each station
distance_savename = f"{drive}:/outputs/{country_save}\\distances_matrix.npy"

if distance_savename not in glob.glob(f"{drive}:/outputs/{country_save}\\*"):
    print("calculating distances")          
    distances_matrix = np.zeros((len(new_df.latitude),len(new_df.latitude)))
    
    for i in range(len(new_df.latitude)):
        for j in range(len(new_df.latitude)):
            distances_matrix[i,j] = calculate_distance(new_df.latitude.iloc[i], 
                                                       new_df.longitude.iloc[i], 
                                                       new_df.latitude.iloc[j], 
                                                       new_df.longitude.iloc[j])
    
    np.save(f"{drive}:/outputs/{country_save}\\distances_matrix", distances_matrix)
else:
    distances_matrix = np.load(distance_savename)


#calculate b as average
b2 = [0]*len(new_df.latitude)
n_stations_in_group = [0]*len(new_df.latitude)

for i in range(len(new_df.latitude)):
    if pd.isna(new_df.b.iloc[i]):
        b2[i] = np.nan
        n_stations_in_group[i] = np.nan
    else:
        station_distances = distances_matrix[i,:]
        close_locs = np.where(station_distances<=radius*1000)
        
        b2[i] = np.mean(new_df.b.iloc[close_locs])
        n_stations_in_group[i] = len(close_locs[0])
        
plt.hist(n_stations_in_group)
plt.xlabel("number of stations within radius")
plt.title(f"{country_save}. radius = {radius}km")


################################################################################
s=3
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]


fig = plt.figure(figsize=(20, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

# Choosing cmap
if df_parameters.b.min() == 0:
    norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
else:
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


plt.title(f'GSDR: {ERA_country}. b at 0 sig level', fontsize=16)



ax2 = fig.add_subplot(1, 2, 2, projection=proj)

# Add map features
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')

# Choosing cmap
if df_parameters.b.min() == 0:
    norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
else:
    norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())

sc = ax2.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = b2,
    s = s,
    cmap = 'seismic',
    norm = norm
)



# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('b', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'b moving average. radius = {radius}km', fontsize=16)
plt.show()



df_savename = f"{drive}:/outputs/{country_save}\\parameters_rolling{radius}.csv"
saved_output_files = glob.glob(drive + ':/outputs/'+country_save+'/*')

if df_savename not in saved_output_files: #read in files and create t time series and do TENAX if it hasnt been done already
    print('TENAX not done yet on '+country_save+' with rolling b. making data.')
    
    saved_files = glob.glob(drive+':/'+country+'_temp/*') #temp files already saved
    
    F_phats = [0]*len(files_sel)
    RL = [0]*len(files_sel)
    
    start_time = [0]*len(files_sel)
    
    for i in np.arange(0, len(files_sel)):
        start_time[i] = time.time() 
        #read in ppt data
        if 'code_str' in locals():
            G,data_meta = read_GSDR_file(files_sel[i],name_col)
        else:
            G = pd.read_csv(files_sel[i])
            G['prec_time'] = pd.to_datetime(G['prec_time'])
            G.set_index('prec_time', inplace=True)
            
        ######################################################################
        #read in T data
        if 'code_str' in locals():
            save_path = drive + ':/'+country+'_temp\\'+code_str + str(val_info.station[val_info.index[i]]) + '.nc'
        else:
            save_path = drive + ':/'+country+'_temp\\'+str(val_info.station[val_info.index[i]]) + '.nc'
        
        
        # Check if file already exists before saving
        
        if save_path not in saved_files:
            print(f'file {save_path} not there')
            T_ERA = []
            
        else:
            print(f"File {save_path} already exists. Skipping loading.")
            T_ERA = xr.load_dataarray(save_path)
            
            #####################################################################
        #TENAX 
        if len(T_ERA) == 0: # dont do tenax if no T data saved
            print('skip')
            F_phats[i] = np.array([np.nan,np.nan,np.nan,np.nan])
            RL[i] = np.nan
        else:
            data = G 
            data = S.remove_incomplete_years(data, name_col)
            t_data = (T_ERA.squeeze()-273.15).to_dataframe()
            
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
            
            AMS = dict_AMS['60']
            
            
            df_arr_t_data = np.array(t_data[temp_name_col])
            df_dates_t_data = np.array(t_data.index)
            
            dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
            
            
            
            # Your data (P, T arrays) and threshold thr=3.8
            P = dict_ordinary["60"]["ordinary"].to_numpy() 
            T = dict_ordinary["60"]["T"].to_numpy()  
            
            
            # Number of threshold 
            thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
            
            
            n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
            
            AMS_sort = AMS.sort_values(by=['AMS'])['AMS']
            plot_pos = np.arange(1,np.size(AMS_sort)+1)/(1+np.size(AMS_sort))
            
            eRP = 1/(1-plot_pos)
            S.return_period = eRP
            
            #TENAX MODEL HERE
            #magnitude model
            F_phats_norm, loglik, _, _ = S.magnitude_model(P, T, thr)
            F_phats[i], loglik, _, _ = S.magnitude_model(P, T, thr, b_set = b2[i])
            #temperature model
            g_phat = S.temperature_model(T)
            
            T_min = g_phat[0] - 2.5 * g_phat[1]
            T_max = g_phat[0] + 2.5 * g_phat[1]
            Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
            
            RL[i], __, __ = S.model_inversion(F_phats[i], g_phat, n, Ts)
            
            
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(files_sel)-i)*time_taken/60
            print(save_path)
            print(files_sel[i])
            print(f"b exp: {F_phats[i]}. normal {F_phats_norm}")
            print(RL[i])
            print(f"{i}/{len(files_sel)}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        
    
    
    df_parameters_rolling = pd.DataFrame({'station':val_info.station,'latitude':val_info.latitude,'longitude':val_info.longitude,
                                       'kappa':np.array(F_phats)[:,0],'b':np.array(F_phats)[:,1],'lambda':np.array(F_phats)[:,2],'a':np.array(F_phats)[:,3],
                                       'return_levels': RL
                                       })
    df_parameters_rolling.to_csv(df_savename,index=False) #save calculated parameters
    

else:
    print('TENAX already done! reading in data')
    df_parameters_rolling = pd.read_csv(df_savename) 
    nan_locs = df_parameters_rolling.b[df_parameters_rolling.b.isna()].index
    replace_range = np.arange(0,len(df_parameters_rolling))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
        df_parameters_rolling.at[j, "return_levels"] = np.fromstring(df_parameters_rolling.return_levels.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        

RL_df = pd.read_csv(f"{drive}:/outputs/{country_save}/return_levels.csv", dtype={'station': str})
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
    



# CHECKS
for j in np.arange(1012,1019):
    plot_pos = np.arange(1,np.size(RL_df.obs_AMS.iloc[j])+1)/(1+np.size(RL_df.obs_AMS.iloc[j]))
    
    eRP = 1/(1-plot_pos)
    
    TNX_FIG_valid(RL_df.obs_AMS.iloc[j],eRP,RL_df.return_levels_b0.iloc[j],TENAXlabel = 'b=0',obslabel='AMS')
    plt.plot(eRP,RL_df.return_levels.iloc[j],"r",label = "free")
    plt.plot(eRP,df_parameters_rolling.return_levels.iloc[j],'g',label = "rolling ave b")
    #plt.plot(eRP,RL_df.return_levels_5.iloc[j],"g",alpha = 0.5, label = "b = 5% sig")
    if "return_levels_bexp" in RL_df.columns:
        plt.plot(eRP,RL_df.return_levels_bexp.iloc[j],"m", label = f"exp")
    
    plt.ylim(0,np.max(RL_df.return_levels.iloc[j])+5)
    plt.xlim(1,np.max(eRP)+2)
    
    plt.legend()
    plt.title(f"{j} lat: {df_parameters_rolling.latitude.iloc[j]}, lon: {df_parameters_rolling.longitude.iloc[j]}. station {df_parameters_rolling.station.iloc[j]}. b_ave = {df_parameters_rolling.b.iloc[j]:.3f}")
    plt.show()
    
    

