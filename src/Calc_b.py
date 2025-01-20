# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:14:46 2024

@author: ellar

runs trhough precipitation files, makes time series if not created already, calculates TENAX parameters if not already calculated and saved
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



drive = 'D'

# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


# country = 'Belgium'
# ERA_country = 'Germany' #country where the era files are
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# country_save = 'Belgium'
# code_str = 'BE_'
# name_len = 8 #how long the numbers are at the end of the files
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet


# country = 'US' 
# ERA_country = 'Puerto_rico'
# country_save = 'Puerto_rico'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 17.6,-67.3,18.5,-64.7
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9

# country = 'US' 
# ERA_country = 'Hawaii'
# country_save = 'Hawaii'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 18.8,-160,22.3,-154.8  #Hawaii
# name_len = 6
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9

country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9


# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9

# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany2'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.7


# country = 'Portugal' 
# ERA_country = 'Portugal'
# country_save = 'Portugal'
# code_str = 'PT_'
# minlat,minlon,maxlat,maxlon = 36.9,-9.5,42.1, -5
# name_len = 0
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet


# country = 'UK' 
# ERA_country = 'UK'
# country_save = 'UK'
# code_str = 'UK_'
# name_len = 0
# min_startdate = dt.datetime(1981,1,1) #this is for if havent read all ERA5 data yet
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
val_info = val_info[val_info['startdate']>=min_startdate]

if 'minlat' in locals():
    
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
    
else:
    pass


files = glob.glob(drive+':/'+country+'/*') #list of files in country folder
files_sel = [files[i] for i in val_info.index]


S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = 0.05,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )


#make empty lists to read into

df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'
saved_output_files = glob.glob(drive + ':/outputs/'+country_save+'/*')

if df_savename not in saved_output_files: #read in files and create t time series and do TENAX if it hasnt been done already
    print('TENAX not done yet on '+country_save+'. making data.')
    
    T_files = sorted(glob.glob(drive+':/ERA5_land/'+ERA_country+'*/*')) #make list of era5 files
    saved_files = glob.glob(drive+':/'+country+'_temp/*') #temp files already saved
    
    g_phats = [0]*len(files_sel)
    F_phats = [0]*len(files_sel)
    thr = [0]*len(files_sel)
    ns = [0]*len(files_sel)
    
    nans = xr.open_dataarray(T_files[0])[0] 
    nans = np.invert(np.isnan(nans)).astype(int)
    
    saved_counter = 0
    
    start_time = [0]*len(files_sel)
    
    for i in np.arange(0, len(files_sel)):
        start_time[i] = time.time() 
        #read in ppt data
        G,data_meta = read_GSDR_file(files_sel[i],name_col)
        
        print(G[0:3])
        ######################################################################
        #read in T data
        target_lat = val_info.latitude[val_info.index[i]] #define targets
        target_lon = val_info.longitude[val_info.index[i]]
        start_date = val_info.startdate[val_info.index[i]]-pd.Timedelta(days=1) #adding an extra day either end to be safe
        end_date = val_info.enddate[val_info.index[i]]+pd.Timedelta(days=1)
        
        save_path = drive + ':/'+country+'_temp\\'+code_str + str(val_info.station[val_info.index[i]]) + '.nc'
        # Check if file already exists before saving
        
        if save_path not in saved_files:
            print(f'file {save_path} not made yet')
            T_ERA = make_T_timeseries(target_lat,target_lon,start_date,end_date,T_files,nans)
            if len(T_ERA) == 0:
                print('skip')
            else:
                T_ERA.to_netcdf(save_path) #save file with same name format as GSDR
                saved_files.append(save_path)  # Update saved_files to include the newly saved file
                saved_counter = saved_counter+1
        else:
            print(f"File {save_path} already exists. Skipping save.")
            T_ERA = xr.load_dataarray(save_path)
            
            #####################################################################
        #TENAX 
        if len(T_ERA) == 0: # dont do tenax if no T data saved
            print('skip')
            F_phats[i] = np.array([np.nan,np.nan,np.nan,np.nan])
            g_phats[i] = np.array([np.nan,np.nan])
            ns[i] = pd.Series(np.nan)
            thr[i] = np.nan
        else:
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
            
            
            
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(files_sel)-i)*time_taken/60
            print(f"{i}/{len(files_sel)}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        
    
    T_temp = []
    
    df_parameters = pd.DataFrame({'station':val_info.station,'latitude':val_info.latitude,'longitude':val_info.longitude,'mu':np.array(g_phats)[:,0],'sigma':np.array(g_phats)[:,1],'kappa':np.array(F_phats)[:,0],'b':np.array(F_phats)[:,1],'lambda':np.array(F_phats)[:,2],'a':np.array(F_phats)[:,3],'thr':np.array(thr),'n_events_per_yr':np.array(ns)[:,0]})
    df_parameters.to_csv(df_savename) #save calculated parameters
    
    TENAX_use = pd.DataFrame({'alpha':[S.alpha],'beta':[S.beta],'left_censoring':[S.left_censoring[1]],'event_duration':[60],'n_monte_carlo':[S.n_monte_carlo],'niter_smev':[S.niter_smev],'min_startdate':min_startdate,'min_years':min_yrs})
    TENAX_use.to_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters


else:
    print('TENAX already done! reading in data')
    df_parameters = pd.read_csv(df_savename) 
    TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters

########################

#RUN WITH FREE b
S.alpha = 0
df_parameters_neg = df_parameters[df_parameters.b==0].copy()

length_neg = len(df_parameters_neg.b)
if name_len!=0:
    df_parameters_neg.station = df_parameters_neg['station'].apply(lambda x: f'{int(x):0{name_len}}') #need to edit this according to file
else:
    pass

F_phats2 = [0]*length_neg
start_time = [0]*length_neg

save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'

if save_path_neg not in saved_output_files:
    print('making the extra bs')
    for i in np.arange(0,length_neg):
        start_time[i] = time.time()
        read_path = drive + ':/'+country+'_temp\\'+code_str + str(df_parameters_neg.station[df_parameters_neg.index[i]]) + '.nc'
        read_path_ppt = drive + ':/'+country+'\\'+code_str + str(df_parameters_neg.station[df_parameters_neg.index[i]]) + '.txt'
        T_ERA = xr.load_dataarray(read_path)
        
        G,data_meta = read_GSDR_file(read_path_ppt,name_col)
        
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
        thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
        
        
        #TENAX MODEL HERE
        #magnitude model
        F_phats2[i], loglik, _, _ = S.magnitude_model(P, T, thr)
        #temperature model
          
        
        
        
        time_taken = (time.time()-start_time[i-9])/10
        time_left = (length_neg-i)*time_taken/60
        print(F_phats2[i])
        print(f"{i}/{length_neg}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
    
    
    print('finished making b again')
    
    
    df_parameters_neg['kappa2']= np.array(F_phats2)[:,0]
    df_parameters_neg['b2'] = np.array(F_phats2)[:,1]
    df_parameters_neg['lambda2'] = np.array(F_phats2)[:,2]
    df_parameters_neg['a2'] = np.array(F_phats2)[:,3]
    
    
    df_parameters_neg.to_csv(save_path_neg)
else:
    print('bs already made. loading')
    df_parameters_neg = pd.read_csv(save_path_neg)


##########################################################################

b_zero = df_parameters.b[df_parameters.b==0].count()
total = df_parameters.b.count()
perc_sig = 100*(1-b_zero/total)
print(f'Percent of stations with significant b: {perc_sig:.0f}%')

b_pos = df_parameters.b[df_parameters.b>0].count()
perc_pos = 100*(b_pos/total)
b_neg = df_parameters.b[df_parameters.b<0].count()
perc_neg = 100*(b_neg/total)
print(f'Percent of stations with significant positive b: {perc_pos:.0f}%')
print(f'Percent of stations with significant negative b: {perc_neg:.0f}%')


non_calc = df_parameters['b'].isna().sum()
print(f'Number of stations without ERA data: {non_calc} out of {len(df_parameters)} stations')


new_df = df_parameters[['latitude','longitude','b']].copy()
mask = new_df['b'] == 0

new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()



#PLOTS
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]

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
if df_parameters.b.min() == 0:
    norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
else:
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

plt.title(f'GSDR: {ERA_country}. b at {TENAX_use.alpha[0]} sig level', fontsize=16)
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


plt.title(f'GSDR: {ERA_country}. μ', fontsize=16)
plt.show()


###############################################################
#scale param
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.TwoSlopeNorm(vmin=0,vcenter = (df_parameters["lambda"].max()/2) , vmax=df_parameters["lambda"].max())
sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=df_parameters["lambda"],
    s = s,
    cmap='YlGnBu',  
    norm=norm
)




# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('λ (mm/hr)', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. λ', fontsize=16)
plt.show()

#######################################################
#scale param
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.TwoSlopeNorm(vmin=0,vcenter = (df_parameters.kappa.max()/2) , vmax=df_parameters.kappa.max())
sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=df_parameters.kappa,
    s = s,
    cmap='hsv',
)




# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('κ', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. κ', fontsize=16)
plt.show()

#######################################################
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=df_parameters.a,
    s = s,
    cmap='hsv',  
)




# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('a', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. a', fontsize=16)
plt.show()

#######################################################



def kendall_pval(x,y):
    return kendalltau(x,y)[1]

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def spearmanr_pval(x,y):
    return spearmanr(x,y)[1]

correlation_variables = df_parameters[df_parameters.columns[-8:-1]].copy()
correlation_variables['thr'] = correlation_variables['thr'].replace(0, np.nan)

correlations = correlation_variables.corr() #r values correlations between all the parameters but b is zero if non significant
P_vals = correlation_variables.corr(method=pearsonr_pval) #p values for sig test

correlations_b = {}
P_vals_b = {}
for col1 in correlation_variables.columns: #correlations and p values for the new non-zero b
    corr_value = correlation_variables[col1].corr(new_df.b)
    correlations_b[(col1, 'b')] = corr_value
    P_vals_b[(col1, 'b')] = correlation_variables[col1].corr(new_df.b,method=pearsonr_pval)
    

#############################################################################
varis = ['mu','sigma','kappa', 'lambda', 'a', 'thr']
for name in varis:

    coeffs=np.polyfit(correlation_variables[name].dropna(),new_df.b.dropna(),1)
    delt = (np.max(correlation_variables[name])-np.min(correlation_variables[name]))/10
    x = np.arange(np.min(correlation_variables[name]),np.max(correlation_variables[name])+delt,delt)
    y = coeffs[0]*x+coeffs[1]
    
    
    plt.scatter(correlation_variables[name],new_df.b,alpha = val_info.cleaned_years/np.max(val_info.cleaned_years))
    plt.plot(x,y,color = 'r',label = f'y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
    plt.xlabel(name)
    plt.ylabel('b')
    r_val = correlations_b[(name, "b")]
    p_val = P_vals_b[(name, "b")]
    plt.text(np.min(df_parameters[name]),np.min(new_df.b),f'r = {r_val:.3f}\n p = {p_val:.5f}')
    plt.legend()
    plt.show()
##################################################



#BOXPLOT

plt.boxplot([new_df.b.copy().dropna(),df_parameters.b.copy().dropna()],vert=False)
plt.xlabel('b')
plt.yticks([1,2],['b allowed to be non sig','b forced to zero if not sig'])
plt.title(f'{ERA_country}')
plt.show()




