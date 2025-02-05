# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:31:05 2025

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
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, pearsonr, spearmanr
import matplotlib.patches as mpatches


drive = 'D'

# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
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



name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 

#READ IN META INFO FOR COUNTRY
info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})


info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

#select stations


val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min

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


S = TENAX(
        return_period = [2,5,10,20,50,100, 200],  #for some reason it doesnt like calculating RP =<1
        durations = [60, 180],
        left_censoring = [0, 0.95], #get top 5%
        alpha = 0.05,
    )



save_name = f"{drive}:/outputs/{country_save}\\timings_top_5.csv"
saved_output_files = glob.glob(f'{drive}:/outputs/{country_save}/*')
if save_name not in saved_output_files:
    print('Not done the months thing yet, doing it')
        
    month_freq = [0]*np.size(files_sel)
    start_time = [0]*np.size(files_sel)
    
    
    for i in np.arange(0,np.size(files_sel)):
        start_time[i] = time.time()
        data,data_meta = read_GSDR_file(files_sel[i],name_col)    
        data = S.remove_incomplete_years(data, name_col)
        
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
        
        thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
        
        
        years = dict_ordinary['60'].year.unique()
        
        above_thr_dict = dict_ordinary['60'].copy()[dict_ordinary['60']["ordinary"]>=thr]
        above_thr_dict['month'] = above_thr_dict.oe_time.dt.month
        month_freq[i] = above_thr_dict['month'].value_counts()/len(above_thr_dict)
        
        if (i+1)%50 == 0:
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (np.size(files_sel)-i)*time_taken/60
            print(f"{i}/{np.size(files_sel)}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
        
        
    month_freq_full = [0]*np.size(files_sel)
    for i in np.arange(0,np.size(files_sel)):
        start_time[i] = time.time()
        month_freq_new = pd.DataFrame(month_freq[i].copy())
        month_freq_new['latitude'] = val_info.latitude[val_info.index[i]]
        month_freq_new['longitude'] = val_info.longitude[val_info.index[i]]
        month_freq_new['station'] = val_info.station[val_info.index[i]]
        month_freq_full[i] = month_freq_new
        
        if (i+1)%50 == 0:
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (np.size(files_sel)-i)*time_taken/60
            print(f"{i}/{np.size(files_sel)}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
    
    
    month_freq_full_df = pd.concat(month_freq_full)
    
    month_freq_full_df.to_csv(save_name)

else:
    print('loading months data')
    month_freq_full_df = pd.read_csv(save_name)
    

# get top 3 months for each station
month_1 = []
month_2 = []
month_3 = []

for station_no in month_freq_full_df.station.unique():
    df_cut = month_freq_full_df[month_freq_full_df.station == station_no]
    
    
    month_1.append({
        'station' : station_no,
        'month' : df_cut.month[df_cut.index[0]],
        'fraction' : df_cut['count'][df_cut.index[0]],
        'latitude' : df_cut.latitude[df_cut.index[0]],
        'longitude' : df_cut.longitude[df_cut.index[0]]
        })
    
    month_2.append({
        'station' : station_no,
        'month' : df_cut.month[df_cut.index[1]],
        'fraction' : df_cut['count'][df_cut.index[1]],
        'latitude' : df_cut.latitude[df_cut.index[1]],
        'longitude' : df_cut.longitude[df_cut.index[1]]
        })
    month_3.append({
        'station' : station_no,
        'month' : df_cut.month[df_cut.index[2]],
        'fraction' : df_cut['count'][df_cut.index[2]],
        'latitude' : df_cut.latitude[df_cut.index[2]],
        'longitude' : df_cut.longitude[df_cut.index[2]]
        })
    

month_1 = pd.DataFrame(month_1)
month_2 = pd.DataFrame(month_2)
month_3 = pd.DataFrame(month_3)


##################################################
# PLOTS


months = ['jan','feb','mar', 'apr', 'may', 'jun','jul','aug','sep','oct','nov','dec']
colors = ['tab:pink','r','tab:orange','y','g','lime','c','b','m','tab:brown','tab:gray','k']


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

legend_patches = []

for month_no in month_1.month.unique():
    df_cut_month = month_1[month_1.month == month_no]
    for _, row in df_cut_month.iterrows():
        plt.arrow(
            row.longitude,
            row.latitude,
            row.fraction * np.sin(month_no * np.pi / 6) * 3, 
            row.fraction * np.cos(month_no * np.pi / 6) * 3,
            color=colors[month_no - 1],
            linewidth=2,  
            head_width=0.3,
            head_length=0.4,  
            length_includes_head=True
        )
    legend_patches.append(mpatches.Patch(color=colors[month_no - 1], label=months[month_no - 1]))

plt.legend(handles=legend_patches)
plt.title('month with the most top 5% events')
plt.show()


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

legend_patches = []

for month_no in month_1.month.unique():
    df_cut_month = month_1[month_1.month == month_no]
    for _, row in df_cut_month.iterrows():
        plt.arrow(
            row.longitude,
            row.latitude,
            row.fraction * np.sin(month_no * np.pi / 6) * 3, 
            row.fraction * np.cos(month_no * np.pi / 6) * 3,
            color='k',
            linewidth=2,  
            head_width=0.3,
            head_length=0.4,  
            length_includes_head=True
        )
    legend_patches.append(mpatches.Patch(color=colors[month_no - 1], label=months[month_no - 1]))

plt.legend(handles=legend_patches)
plt.title('month with the most top 5% events')
plt.show()



fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

for month_no in month_2.month.unique():
    df_cut_month = month_2[month_2.month == month_no]
    sc = ax1.scatter( #plot the negligable at 5% lvl points
        df_cut_month.longitude,
        df_cut_month.latitude,
        s = 5,
        c = colors[month_no-1],
        label = months[month_no-1]
    )
plt.legend()
plt.title('month with the second most top 5% events')
plt.show()


fig, ax = plt.subplots()

for station_no in month_freq_full_df.station.unique():
    df_cut = month_freq_full_df[month_freq_full_df.station == station_no]
    top_month = df_cut.month.iloc[0]
    df_cut_sort = df_cut.sort_values(by = 'month')
    plt.plot(df_cut_sort.month,df_cut_sort['count'],color = colors[top_month-1],alpha = 0.1)
    

plt.xticks(np.arange(1,13),months) #put months as the x labels
plt.ylabel('Fraction of extreme ppt this month') 
custom_lines = [[Line2D([0], [0], color=colors[i], lw=4)][0] for i in np.arange(0,12)]
ax.legend(custom_lines, months)   
plt.title(country)
plt.show()
 
 







