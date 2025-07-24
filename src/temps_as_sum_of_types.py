# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 10:28:53 2025

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
from scipy.signal import find_peaks

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
from scipy.stats import norm, skewnorm, skew, kurtosis
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

drive = 'D'



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
df = pd.read_csv(save_name,dtype = {0:str})


df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'
df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 

#merging the dataframes to ensure station consistency
missing_rows = pd.merge(df_parameters.station, df.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
else:
    pass

min_yrs = 10

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
val_info = val_info.reset_index()


# read in data for storm types with matching stations
station_ids = val_info.station.to_numpy()


drop_id = []
storm_files = []


#make list of filenames only first
for i in range(len(station_ids)):
    file_name = f"D:/NSF_CausesData/NSF_CausesData\\USC00{station_ids[i]}.csv"
    if file_name in glob.glob("D:/NSF_CausesData/NSF_CausesData/*"):
        storm_files.append(file_name)
    else:
        drop_id.append(i)
        
matched_info = val_info[~val_info.index.isin(drop_id)].reset_index()

# for i in range(len(station_ids)):
#     file_name = f"D:/NSF_CausesData/NSF_CausesData\\USC00{station_ids[i]}.csv"
#     if file_name in glob.glob("D:/NSF_CausesData/NSF_CausesData/*"):
#         storm_types.append(pd.read_csv(file_name,dtype={' date': str}))
#     else:
#         drop_id.append(i)


s = 3
fontsize = 15


n_lat = len(region_lats)-1
n_lon = len(region_lons)-1


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1,1,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c="g",
    alpha = 0.2,
    s = s,
    label = "all stations"
)

sc = ax.scatter(
    matched_info.longitude,
    matched_info.latitude,
    c="r",
    s = s, 
    label = "matched stations"
)

for lat_i in range(n_lat-1):
    ax.plot([minlon-3,maxlon+3],[region_lats[lat_i+1],region_lats[lat_i+1]],  'r', linewidth=2, transform=ccrs.PlateCarree())

for lon_i in range(n_lon-1):
    ax.plot([region_lons[lon_i+1],region_lons[lon_i+1]],[minlat-3,maxlat+3],  'r', linewidth=2, transform=ccrs.PlateCarree())


plt.legend()
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

plt.xlim(-125,-70)
plt.ylim(25,50)

plt.show()

################################################################################
# Plot the stuff by region
region = 0
mask = (matched_info.latitude >= region_lats[region])&(
    matched_info.latitude < region_lats[region+1])&(
        matched_info.longitude >= region_lons[region])&(
            matched_info.longitude < region_lons[region+1])

            
mask = (matched_info.latitude >= region_lats[region])&( #florida
    matched_info.latitude < 32)&(
        matched_info.longitude >= -85)&(
            matched_info.longitude < -60)


matched_info_mask = matched_info[mask]

storm_types = []
ord_events = []
storm_types_events = [] # for the event types matching the days of the ordinary event data
combed_events_stuff = []


for i in range(len(matched_info_mask)):
    station = matched_info.station.iloc[i]
    file_name = f"D:/NSF_CausesData/NSF_CausesData\\USC00{station}.csv"
    storm_types.append(pd.read_csv(file_name,dtype={' date': str}))
    
    T_ = np.genfromtxt(f"D:/ordinary_events/US_main/T_{station}.csv")
    P_ = np.genfromtxt(f"D:/ordinary_events/US_main/P_{station}.csv")
    times = pd.read_csv(f"D:/ordinary_events/US_main/time_{station}.csv",parse_dates = ["oe_time"])
    
    
    oe = pd.DataFrame({
        "oe_time": times.oe_time,
        "T" : T_,
        "P" : P_
        })
    
    oe["date"] = pd.to_datetime(oe.oe_time).dt.strftime('%Y%m%d')
    
    
    storm_types_events.append(storm_types[i][storm_types[i][" date"].isin(oe.date)])
    ord_events.append(oe)
    
    oe = oe[oe.date.isin(storm_types_events[i][" date"])]
    combed_events_stuff.append(pd.concat([oe.reset_index(),storm_types_events[i].reset_index()],axis = 1))
    
    #plt.plot(eTs_df_stations.drop(columns = "station").iloc[i].to_numpy(),kernel_df.iloc[i][1:],label = station)
    

# do the temp stufff babs
S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0,
        min_ev_dur = 60,
        beta = 2
    )

colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

eT = np.arange(-12,45)
eT1 = np.arange(-12,45,2) 

for i in range(len(combed_events_stuff)):
    df_now = combed_events_stuff[i]
    
    kde  = gaussian_kde(df_now["T"].to_numpy())
    prob = kde(eT)
    
    causes = df_now[" cause_1"].unique()
    g_phat = []
    n_events = []
    pdf_values = []
    
    plt.plot(eT,prob, color = "r", linewidth = 3,label = "kernel density")
    eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(df_now["T"].to_numpy(), bins=eT_edges, density=True)
    plt.plot(eT1, hist, '--', color = "r")
    
    
    for j in range(len(causes)):
        little_df = df_now[df_now[" cause_1"] == causes[j]]
        T = little_df["T"].to_numpy()
        
        g_phat.append(S.temperature_model(T))
        n_events.append(len(T))
        pdf_values.append(gen_norm_pdf(eT, g_phat[j][0], g_phat[j][1], S.beta) * (len(T)/len(df_now)))
        
        eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
        hist, bin_edges = np.histogram(T, bins=eT_edges, density=True)
        plt.plot(eT1, hist * (len(T)/len(df_now)), '--', color = colours[j])
        
        
        plt.plot(eT,pdf_values[j], color = colours[j], label = causes[j])
    
    plt.plot(eT,np.sum(pdf_values,axis = 0), color = "k", linewidth = 3,label = "sum")
    plt.title(f"betas = {S.beta}. station: {matched_info.station.iloc[i]} \n ({matched_info.latitude.iloc[i]:.1f},{matched_info.longitude.iloc[i]:.1f})")
    plt.legend()
    plt.show()


S.beta = 4      
for i in range(len(combed_events_stuff)):
    df_now = combed_events_stuff[i]
    
    eT = np.arange(-12,45)
    
    kde  = gaussian_kde(df_now["T"].to_numpy())
    prob = kde(eT)
    
    causes = df_now[" cause_1"].unique()
    g_phat = []
    n_events = []
    pdf_values = []
    
    plt.plot(eT,prob, color = "r", linewidth = 3,label = "kernel density")
    
    eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(df_now["T"].to_numpy(), bins=eT_edges, density=True)
    plt.plot(eT1, hist, '--', color = "r")
    
    
    for j in range(len(causes)):
        little_df = df_now[df_now[" cause_1"] == causes[j]]
        T = little_df["T"].to_numpy()
        
        g_phat.append(S.temperature_model(T))
        n_events.append(len(T))
        pdf_values.append(gen_norm_pdf(eT, g_phat[j][0], g_phat[j][1], S.beta) * (len(T)/len(df_now)))
        
        hist, bin_edges = np.histogram(T, bins=eT_edges, density=True)
        plt.plot(eT1, hist * (len(T)/len(df_now)), '--', color = colours[j])
        
        plt.plot(eT,pdf_values[j],label = causes[j])
    plt.plot(eT,np.sum(pdf_values,axis = 0), color = "k", linewidth = 3,label = "sum")
    plt.title(f"betas = {S.beta}. station: {matched_info.station.iloc[i]}")
    plt.legend()
    plt.show()   

## skewed
for i in range(len(combed_events_stuff)):
    df_now = combed_events_stuff[i]
    print(f"i: {i} {len(df_now[[" cause_1"," cause_2"]][df_now[" cause_1"] != df_now[" cause_2"]])}")
    
    eT = np.arange(-12,45)
    
    fig = plt.figure(figsize = (15,5))
    ax1 = fig.add_subplot(1,3,1)  
    
    kde  = gaussian_kde(df_now["T"].to_numpy())
    prob = kde(eT)
    
    causes = df_now[" cause_1"].unique()
    g_phat = []
    n_events = []
    pdf_values = []
    
      
    ax1.plot(eT,prob, color = "r", linewidth = 3,label = "kernel density")
    eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(df_now["T"].to_numpy(), bins=eT_edges, density=True)
    plt.plot(eT1, hist, '--', color = "r")
    
    
    for j in range(len(causes)):
        little_df = df_now[df_now[" cause_1"] == causes[j]]
        T = little_df["T"].to_numpy()
        
        
        try:
            g_phat.append(S.temperature_model(T,method = "skewnorm"))
            n_events.append(len(T))
            pdf_values.append(skewnorm.pdf(eT, *g_phat[j]) * (len(T)/len(df_now)))
            
            eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
            hist, bin_edges = np.histogram(T, bins=eT_edges, density=True)
            plt.plot(eT1, hist * (len(T)/len(df_now)), '--', color = colours[j])
            
            plt.plot(eT,pdf_values[j],color = colours[j],label = causes[j])
        # process the result here
        except RuntimeError:
            plt.text(0,0.05,f"runtime err on {causes[j]}")
            g_phat.append([np.nan,np.nan,np.nan])
            n_events.append(len(T))
            pdf_values.append(np.zeros(len(eT)))
            continue  # skip to the next iteration
        
        
    plt.plot(eT,np.nansum(pdf_values,axis = 0), color = "k", linewidth = 3,label = "sum")
    ax1.set_title(f"skewnorm. station: {matched_info.station.iloc[i]}")
    plt.legend()
    
    
    summer_df = df_now[(df_now.oe_time.dt.month >= 5)&(df_now.oe_time.dt.month <= 10)]
    winter_df = df_now[(df_now.oe_time.dt.month < 5)|(df_now.oe_time.dt.month > 10)]
    
    
    ax2 = fig.add_subplot(1,3,2)
    kde  = gaussian_kde(summer_df["T"].to_numpy())
    prob = kde(eT)
    
    causes = summer_df[" cause_1"].unique()
    g_phat = []
    n_events = []
    pdf_values = []
    
      
    ax2.plot(eT,prob, color = "r", linewidth = 3,label = "kernel density")
    eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(summer_df["T"].to_numpy(), bins=eT_edges, density=True)
    plt.plot(eT1, hist, '--', color = "r")
    
    
    for j in range(len(causes)):
        little_df = summer_df[summer_df[" cause_1"] == causes[j]]
        T = little_df["T"].to_numpy()
        
        
        try:
            g_phat.append(S.temperature_model(T,method = "skewnorm"))
            n_events.append(len(T))
            pdf_values.append(skewnorm.pdf(eT, *g_phat[j]) * (len(T)/len(summer_df)))
            
            eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
            hist, bin_edges = np.histogram(T, bins=eT_edges, density=True)
            plt.plot(eT1, hist * (len(T)/len(summer_df)), '--', color = colours[j])
            
            plt.plot(eT,pdf_values[j],color = colours[j],label = causes[j])
        # process the result here
        except RuntimeError:
            plt.text(0,0.05,f"runtime err on {causes[j]}")
            g_phat.append([np.nan,np.nan,np.nan])
            n_events.append(len(T))
            pdf_values.append(np.zeros(len(eT)))
            continue  # skip to the next iteration
        
        
    plt.plot(eT,np.nansum(pdf_values,axis = 0), color = "k", linewidth = 3,label = "sum")
    ax2.set_title(f"skewnorm. station: {matched_info.station.iloc[i]}. summer only")
    plt.legend()
    
    ax3 = fig.add_subplot(1,3,3)
    kde  = gaussian_kde(winter_df["T"].to_numpy())
    prob = kde(eT)
    
    causes = winter_df[" cause_1"].unique()
    g_phat = []
    n_events = []
    pdf_values = []
    
      
    ax3.plot(eT,prob, color = "r", linewidth = 3,label = "kernel density")
    eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(winter_df["T"].to_numpy(), bins=eT_edges, density=True)
    plt.plot(eT1, hist, '--', color = "r")
    
    
    for j in range(len(causes)):
        little_df = winter_df[winter_df[" cause_1"] == causes[j]]
        T = little_df["T"].to_numpy()
        
        
        try:
            g_phat.append(S.temperature_model(T,method = "skewnorm"))
            n_events.append(len(T))
            pdf_values.append(skewnorm.pdf(eT, *g_phat[j]) * (len(T)/len(winter_df)))
            
            eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
            hist, bin_edges = np.histogram(T, bins=eT_edges, density=True)
            plt.plot(eT1, hist * (len(T)/len(winter_df)), '--', color = colours[j])
            
            plt.plot(eT,pdf_values[j],color = colours[j],label = causes[j])
        # process the result here
        except RuntimeError:
            plt.text(0,0.05,f"runtime err on {causes[j]}")
            g_phat.append([np.nan,np.nan,np.nan])
            n_events.append(len(T))
            pdf_values.append(np.zeros(len(eT)))
            continue  # skip to the next iteration
        
        
    plt.plot(eT,np.nansum(pdf_values,axis = 0), color = "k", linewidth = 3,label = "sum")
    ax3.set_title(f"skewnorm. station: {matched_info.station.iloc[i]}. winter only")
    plt.legend()
    
    
    
    
    plt.show()  


################################################################################
## plot comparison
method = "norm"
S.beta = 2

for i in range(len(combed_events_stuff)):
    df_now = combed_events_stuff[i]
    print(f"i: {i} {len(df_now[[" cause_1"," cause_2"]][df_now[" cause_1"] != df_now[" cause_2"]])}")
    
    eT = np.arange(-12,45)
    
    
    kde  = gaussian_kde(df_now["T"].to_numpy())
    prob = kde(eT)
    
    causes = df_now[" cause_1"].unique()
    g_phat = []
    n_events = []
    pdf_values = []
    
    fig = plt.figure(figsize = (11,5))
    ax1 = fig.add_subplot(1,2,1)
    
    ax1.plot(eT,prob, color = "k", linewidth = 3,label = "kernel density")
    g_phat_full = S.temperature_model(df_now["T"].to_numpy(),method = method)
    if method == "skewnorm":
        pdf_values_full = skewnorm.pdf(eT, *g_phat_full)
    else:
        pdf_values_full = gen_norm_pdf(eT, *g_phat_full,S.beta)
    
    plt.plot(eT,pdf_values_full, label = "skew fit for all")
    
    
    eT_edges = np.concatenate([np.array([eT1[0]-(eT1[1]-eT1[0])/2]),(eT1 + (eT1[1]-eT1[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(df_now["T"].to_numpy(), bins=eT_edges, density=True)
    plt.plot(eT1, hist, '--', color = "k")
    
    
    for j in range(len(causes)):
        little_df = df_now[df_now[" cause_1"] == causes[j]]
        T = little_df["T"].to_numpy()
        
        
        try:
            g_phat.append(S.temperature_model(T,method = method))
            n_events.append(len(T))
            if method == "skewnorm":
                pdf_values.append(skewnorm.pdf(eT, *g_phat[j]) * (len(T)/len(df_now)))
            else:
                pdf_values.append(gen_norm_pdf(eT, *g_phat[j],S.beta)* (len(T)/len(df_now)))
            
        # process the result here
        except RuntimeError:
            g_phat.append([np.nan,np.nan,np.nan])
            n_events.append(len(T))
            pdf_values.append(np.zeros(len(eT)))
            continue  # skip to the next iteration
        
        
    plt.plot(eT,np.nansum(pdf_values,axis = 0), color = "r",label = "sum of storm types")
    plt.title(f"{method}. station: {matched_info.station.iloc[i]}\n ({matched_info.latitude.iloc[i]:.1f},{matched_info.longitude.iloc[i]:.1f})")
    plt.ylim(0,0.1)
    
    
    summer_df = df_now[(df_now.oe_time.dt.month >= 5)&(df_now.oe_time.dt.month <= 10)]
    winter_df = df_now[(df_now.oe_time.dt.month < 5)|(df_now.oe_time.dt.month > 10)]
    
    
    T_winter = winter_df["T"].to_numpy()
    g_phat_winter = S.temperature_model(T_winter,method = method)
    if method == "skewnorm":
        pdf_values_winter = skewnorm.pdf(eT, *g_phat_winter)
    else:
        pdf_values_winter = gen_norm_pdf(eT,*g_phat_winter,S.beta)
    
    T_summer = summer_df["T"].to_numpy()
    g_phat_summer = S.temperature_model(T_summer,method = method)
    pdf_values_summer = skewnorm.pdf(eT, *g_phat_summer)
    if method == "skewnorm":
        pdf_values_summer = skewnorm.pdf(eT, *g_phat_summer)
    else:
        pdf_values_summer = gen_norm_pdf(eT,*g_phat_summer,S.beta)
    
    
    
    summer_winter_sum = pdf_values_summer * (len(T_summer)/len(df_now)) + pdf_values_winter * (len(T_winter)/len(df_now))
    
    plt.plot(eT,summer_winter_sum, color = "b",label = "sum of summer winter")
    
    
    plt.legend()
    
    
    ax2 = fig.add_subplot(1,2,2)
    full_temp_xr = xr.load_dataarray(f"D:/US_temp/US_{matched_info.station.iloc[i]}.nc")
    full_temp = full_temp_xr.to_numpy() - 273.15
    full_temp_24hr = full_temp_xr.to_pandas().resample("d").mean() - 273.15
    full_temp_rolling = full_temp_xr.to_pandas().rolling("d").mean() - 273.15
    
    
    
    kde_FT  = gaussian_kde(full_temp)
    prob_FT = kde_FT(eT)
    
    kde_FT_24hr  = gaussian_kde(full_temp_24hr)
    prob_FT_24hr = kde_FT_24hr(eT)
    
    kde_FT_rolling  = gaussian_kde(full_temp_rolling)
    prob_FT_rolling = kde_FT_rolling(eT)
    
    ax2.plot(eT,prob_FT,label= "full temperature distribution")
    ax2.plot(eT,prob_FT_24hr,label= "full temperature distribution, 24 hour mean")
    ax2.plot(eT,prob_FT_rolling,label= "full temperature distribution, rolling mean")
    
    ax2.plot(eT,prob,label= "storms temperature distribution")
    
    plt.title("full temperature distribution (kernel density)")
    plt.legend()
    plt.ylim(0,0.1)
    
    
    plt.show()  














