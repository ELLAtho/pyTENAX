# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:15:13 2025

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
from scipy.stats import norm, skewnorm
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

drive = 'D'
alpha_set = 0



# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 50


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 30



country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30

# country = 'UK' 
# ERA_country = 'UK'
# country_save = 'UK'
# code_str = 'UK_'
# name_len = 0
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 50



name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 

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

eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
eTs = eTs_df.drop(columns = "station").to_numpy()

n_peaks = [0]*len(df)
n_peaks01 = [0]*len(df)
prominences = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan]]*len(df))
diffs = np.array([[np.nan,np.nan,np.nan,np.nan]]*len(df))

for i in range(len(df)):
    x, y = eTs[i],df.iloc[i][1:]
    peaks = find_peaks(y,prominence = 0.000)
    peaks01 = find_peaks(y,prominence = 0.001)
    n_peaks[i] = len(peaks[0])
    n_peaks01[i] = len(peaks01[0])
    for j in range(n_peaks[i]):
        prominences[i][j] = peaks[1]["prominences"][j]
    for j in np.arange(0,n_peaks[i]-1):
        diffs[i][j] = x[peaks[0][j+1]] -x[peaks[0][j]] 
        
    # if i % 50 == 0:
    #     print(i)

peaks_df = pd.DataFrame({
    "n_peaks" : n_peaks,
    "n_peaks01" : n_peaks01,
    "prominence_1" : prominences[:,0],
    "prominence_2" : prominences[:,1],
    "prominence_3" : prominences[:,2],
    "prominence_4" : prominences[:,3],
    "prominence_5" : prominences[:,4],
    "diff_1" : diffs[:,0],
    "diff_2" : diffs[:,1],
    "diff_3" : diffs[:,2],
    "diff_4" : diffs[:,3],

    
    })

cmap = 'plasma'
bounds = [0.5,1.5,2.5,3.5,4.5]  # 3 discrete levels
norm = mcolors.BoundaryNorm(bounds, plt.get_cmap(cmap).N)
s = 3

fig = plt.figure(figsize=(15, 10))

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=peaks_df.n_peaks,
    cmap=cmap,
    norm = norm,
    s = s
)
ax1.set_title("prominence = 0")


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 2, 2, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=peaks_df.n_peaks01,
    cmap=cmap,
    norm = norm,
    s = s
)
ax1.set_title("prominence = 0.001")
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
cbar = plt.colorbar(sc, shrink = 0.2, cax=cbar_ax, ticks=[1, 2, 3, 4])

plt.show()




###############################################################################
S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0,
        min_ev_dur = 60,
        beta = 4
    )
#Skewness
save_name_skew = f"{drive}:/outputs/{country_save}\\temp_skew.csv"
output_files = glob.glob(f"{drive}:/outputs/{country_save}/*")

if save_name_skew not in output_files:
    print("temp skewed not calculated yet. here we gooooooo")
    

    g_phat_skew = [0] * len(df)
    start_time = [0] * len(df)
    
    for i in np.arange(0, len(df)):
        start_time[i] = time.time()
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
                
            
            if 'code_str' in locals():
                G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
            else:
                G = pd.read_csv(f"{file_name}.csv")
                G['prec_time'] = pd.to_datetime(G['prec_time'])
                G.set_index('prec_time', inplace=True)
                
                
            data = S.remove_incomplete_years(G, name_col)
            
            T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
            
            if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                print('skip')
                T = [np.nan]
            else:
                T_ERA = xr.load_dataarray(T_path)
                t_data = (T_ERA.squeeze()-273.15).to_dataframe()
                
        
                df_arr = np.array(data[name_col])
                df_dates = np.array(data.index)
                
                #extract indexes of ordinary events
                #these are time-wise indexes =>returns list of np arrays with np.timeindex
                idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
                    
                
                #get ordinary events by removing too short events
                #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
                _,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
                
                #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
                dict_ordinary, _ = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
                
                
                
                df_arr_t_data = np.array(t_data[temp_name_col])
                df_dates_t_data = np.array(t_data.index)
                
                if type(df_dates_t_data[0]) != np.datetime64:
                        
                    df_dates_t_data = pd.Series([item[0] for item in df_dates_t_data])
                    df_dates_t_data = np.array(df_dates_t_data)
                else:
                    pass
                
                dicts, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                
                #g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]] 
                
                
                # Your data (P, T arrays) and threshold thr=3.8
                P = dicts["60"]["ordinary"].to_numpy() 
                T = dicts["60"]["T"].to_numpy()  
                
                np.savetxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv",T)
                np.savetxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv",P)
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv")
        
        if len(T) <= 2:
            g_phat_skew[i] = [np.nan]*3
        else:
            g_phat_skew[i] = S.temperature_model(T, method = "skewnorm")
        
        
        if i%50 == 0: 
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(df)-i)*time_taken/60
            print(f"{i}/{len(df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
        
    skew_df = pd.DataFrame({
        'station': df_parameters.station,
        "skewness" : np.array(g_phat_skew)[:,0],
        "g_phat1" : np.array(g_phat_skew)[:,1],
        "g_phat2" : np.array(g_phat_skew)[:,2],
        })
    skew_df.to_csv(save_name_skew,index=False)
else:
    skew_df = pd.read_csv(save_name_skew, dtype={"station":str})
        
    


###############################################################################
#plot

lon_lims = [truncate_neg(np.min(df_parameters.longitude),5),np.ceil(np.max(df_parameters.longitude/5))*5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]


cmap = "seismic"
s = 3
norm = mcolors.Normalize(vmin=np.min(skew_df.skewness)*0.6, vmax=np.min(skew_df.skewness)*-0.6)
fig = plt.figure(figsize=(10, 10))

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=skew_df.skewness,
    cmap=cmap,
    norm = norm,
    s = s
)
ax1.set_title("skewness")
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
plt.colorbar(sc,extend = "both")


proj = ccrs.PlateCarree()

cmap = 'plasma'
bounds = [0.5,1.5,2.5,3.5,4.5]  # 3 discrete levels
norm = mcolors.BoundaryNorm(bounds, plt.get_cmap(cmap).N)

ax1 = fig.add_subplot(2, 1, 2, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=peaks_df.n_peaks01,
    cmap=cmap,
    norm = norm,
    s = s
)
ax1.set_title("number of peaks (prominence = 0.001)")
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
plt.colorbar(sc,ticks=[1, 2, 3, 4])

plt.show()

print(f"max skew: {np.max(skew_df.skewness)}")
print(f"min skew: {np.min(skew_df.skewness)}")




n_peak = 2
sel_lat = [40,50]
sel_lon = [122,140]
peak1 = df_parameters[(peaks_df.n_peaks01 == n_peak)&
                      (df_parameters.latitude.between(sel_lat[0],sel_lat[1]))&
                      (df_parameters.longitude.between(sel_lon[0],sel_lon[1]))]
df1 = df[(peaks_df.n_peaks01 == n_peak)&
                      (df_parameters.latitude.between(sel_lat[0],sel_lat[1]))&
                      (df_parameters.longitude.between(sel_lon[0],sel_lon[1]))]
eTs_df1 = eTs_df[(peaks_df.n_peaks01 == n_peak)&
                      (df_parameters.latitude.between(sel_lat[0],sel_lat[1]))&
                      (df_parameters.longitude.between(sel_lon[0],sel_lon[1]))]

# for i in range(6):
#     oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{peak1.station.iloc[i]}.csv"
#     T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{peak1.station.iloc[i]}.csv")
#     P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{peak1.station.iloc[i]}.csv")
#     oe_time = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{peak1.station.iloc[i]}.csv",parse_dates = ["oe_time"])

    
    
#     #SPLITTING INTO SUMMER/WINTER
#     season_separations = [5, 10]
#     day_separations = [dt.timedelta(100), dt.timedelta(300)]
#     months = oe_time["oe_time"].dt.month
#     years = oe_time["oe_time"].dt.year
#     jans = pd.to_datetime(years.astype(str) + '-01-01') #make dataframe with 1st jan of each year
#     days_since_jan = oe_time["oe_time"] - jans
    
    
    
#     # winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
#     # summer_inds = months.index[(months<season_separations[1]+1)&(months>season_separations[0]-1)]
    
#     winter_inds = days_since_jan.index[(days_since_jan>day_separations[1]) | (days_since_jan<=day_separations[0])]
#     summer_inds = days_since_jan.index[(days_since_jan<=day_separations[1])&(days_since_jan>day_separations[0])]
    
    
#     T_winter = T[winter_inds]
#     T_summer = T[summer_inds]


#     g_phat_winter = S.temperature_model(T_winter,beta = 2)
#     g_phat_summer = S.temperature_model(T_summer,beta = 2)
    
    
#     g_phat_winter_skew = S.temperature_model(T_winter,method = "skewnorm")
#     g_phat_summer_skew = S.temperature_model(T_summer,method = "skewnorm")

#     eT = np.arange(np.min(T),np.max(T)+4)
#     winter_pdf = gen_norm_pdf(eT, g_phat_winter[0], g_phat_winter[1], 2)
#     summer_pdf = gen_norm_pdf(eT, g_phat_summer[0], g_phat_summer[1], 2)
    
#     winter_pdf_skew = skewnorm.pdf(eT, *g_phat_winter_skew)
#     summer_pdf_skew = skewnorm.pdf(eT, *g_phat_summer_skew)

#     combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))
#     combined_pdf_skew = (winter_pdf_skew*np.size(T_winter)+summer_pdf_skew*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))

    
    
    
#     g_phat_skew = S.temperature_model(T, method = "skewnorm")
    
    
#     TNX_FIG_temp_model(T, g_phat_skew, 4, eT, obscol='r',valcol='b',
#                            obslabel = 'observations',
#                            vallabel = 'skewed normal',
#                            xlimits = [np.min(T)-3,np.max(T)+3],
#                            method = "skewnorm")
#     plt.plot(eTs_df1.iloc[i][1:],df1.iloc[i][1:],color = "r", label = "kernel density")
#     S.beta = 4
#     g_phat = S.temperature_model(T)
#     plt.plot(eT,gen_norm_pdf(eT, g_phat[0], g_phat[1], 4),label = "beta = 4")
#     S.beta = 6
#     g_phat6 = S.temperature_model(T)
#     plt.plot(eT,gen_norm_pdf(eT, g_phat[0], g_phat[1], 6),label = "beta = 6")
#     plt.plot(eT,combined_pdf,label = "summer and winter")
#     plt.plot(eT,combined_pdf_skew,label = "summer and winter skewnorms",color = "m")
#     plt.ylim(0,np.max(df1.iloc[i][1:])+0.01)
    
#     plt.legend()
#     plt.title(f"{country_save}. station {peak1.station.iloc[i]}. lat {peak1.latitude.iloc[i]}. lon {peak1.longitude.iloc[i]}")
#     plt.show()



###############################################################################
# choosing stations to look at specifically

minlat_spec, minlon_spec, maxlat_spec, maxlon_spec = 36, -100, 50, -90       #30, -125, 40, -115


mask = (df_parameters.latitude < maxlat_spec)&(
    df_parameters.longitude < maxlon_spec)&(
        df_parameters.latitude > minlat_spec)&(
            df_parameters.longitude > minlon_spec) & (
                peaks_df.n_peaks >1)&(
                    val_info.cleaned_years > 30)

                    
                    
                    
info_masked = val_info[mask]
df_parameters_masked = df_parameters[mask]
stations_choose = df_parameters_masked.station
stations = df_parameters.station


# plot locations of stations chosen
fontsize = 12

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1,1,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)



sc = ax.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c="g",
    alpha = 0.5,
    s = s,
)

sc = ax.scatter(
    df_parameters_masked.longitude,
    df_parameters_masked.latitude,
    c="r",
    s = s, 
)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

plt.show()


###############################################################################

colspecs = [(0,11), (12,20), (21,30), (31,37), (38,40), (41,71)]
names    = ["station_id","latitude","longitude","elevation","state","name"]

database_meta = pd.read_fwf("D:\\NSF_CausesData\\metadata.txt", colspecs=colspecs, names=names)
country_meta = database_meta[(database_meta.latitude>=minlat)&(database_meta.longitude>=minlon)&(database_meta.latitude<=maxlat)&(database_meta.longitude<=maxlon)]



fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1,1,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)



sc = ax.scatter(
    country_meta.longitude,
    country_meta.latitude,
    c="g",
    alpha = 0.5,
    s = s,
)

sc = ax.scatter(
    df_parameters_masked.longitude,
    df_parameters_masked.latitude,
    c="r",
    s = s, 
)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

plt.show()

# choose location
meta_masked = database_meta[
    (database_meta.latitude < maxlat_spec)&(
        database_meta.longitude < maxlon_spec) &(
            database_meta.latitude > minlat_spec)&(
                database_meta.longitude > minlon_spec)]

meta_stations = pd.concat([database_meta[(database_meta.latitude == df_parameters_masked.latitude.iloc[i])&(database_meta.longitude == df_parameters_masked.longitude.iloc[i])] for i in range(len(df_parameters_masked))])

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1,1,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    meta_masked.longitude,
    meta_masked.latitude,
    c="g",
    alpha = 0.2,
    s = s,
    label = "GHCNd stations"
)

sc = ax.scatter(
    df_parameters_masked.longitude,
    df_parameters_masked.latitude,
    c="r",
    s = s, 
    label = "selected SW double stations"
)

sc = ax.scatter(
    meta_stations.longitude,
    meta_stations.latitude,
    c="b",
    s = s, 
    label = "GHCNd and selected"
)


plt.legend()
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

plt.show()


###############################################################################
# load in the storm types
storm_types = []
drop_id = []

for i in range(len(meta_stations)):
    file_name = f"D:/NSF_CausesData/NSF_CausesData\\{meta_stations.station_id.iloc[i]}.csv"
    if file_name in glob.glob("D:/NSF_CausesData/NSF_CausesData/*"):
        storm_types.append(pd.read_csv(file_name,dtype={' date': str}))
    else:
        drop_id.append(i)
meta_stations = meta_stations.drop(labels = meta_stations.index[drop_id]) #drop rows that aren't in the database

fig = plt.figure(figsize=(5, 5))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1,1,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    meta_masked.longitude,
    meta_masked.latitude,
    c="y",
    alpha = 0.1,
    s = s,
    label = "GHCNd stations"
)



for i in range(len(meta_stations)):
    
    sc = ax.scatter(
        meta_stations.longitude.iloc[i],
        meta_stations.latitude.iloc[i],
        s = 10, 
        label = f"{meta_stations.station_id.iloc[i]}"
    )


plt.legend()
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}
gl.xformatter = LongitudeFormatter(degree_symbol="° ")
gl.yformatter = LatitudeFormatter(degree_symbol="° ")

plt.show()



df_parameters_stations = pd.concat([df_parameters_masked[(df_parameters_masked.latitude == meta_stations.latitude.iloc[i])&(df_parameters_masked.longitude == meta_stations.longitude.iloc[i])] for i in range(len(meta_stations))])
kernel_df = df[df.station.isin(df_parameters_stations.station)]
eTs_df_stations = eTs_df[eTs_df.station.isin(df_parameters_stations.station)]


#load in events data
ord_events = []
storm_types_events = [] # for the event types matching the days of the ordinary event data
combed_events_stuff = []


for i in range(len(meta_stations)):
    
    station = df_parameters_stations.station.iloc[i]
    
    T_ = np.genfromtxt(f"D:/ordinary_events/US_main/T_{station}.csv")
    P_ = np.genfromtxt(f"D:/ordinary_events/US_main/P_{station}.csv")
    time = pd.read_csv(f"D:/ordinary_events/US_main/time_{station}.csv",parse_dates = ["oe_time"])
    
    
    oe = pd.DataFrame({
        "oe_time": time.oe_time,
        "T" : T_,
        "P" : P_
        })
    
    oe["date"] = pd.to_datetime(oe.oe_time).dt.strftime('%Y%m%d')
    
    
    storm_types_events.append(storm_types[i][storm_types[i][" date"].isin(oe.date)])
    ord_events.append(oe)
    
    oe = oe[oe.date.isin(storm_types_events[i][" date"])]
    combed_events_stuff.append(pd.concat([oe.reset_index(),storm_types_events[i].reset_index()],axis = 1))
    
    plt.plot(eTs_df_stations.drop(columns = "station").iloc[i].to_numpy(),kernel_df.iloc[i][1:],label = station)
    
plt.legend()
plt.show()

S.beta = 2
for i in range(len(meta_stations)):
    df_now = combed_events_stuff[i]
    
    causes = df_now[" cause_1"].unique()
    g_phat = []
    n_events = []
    pdf_values = []
    
    for j in range(len(causes)):
        little_df = df_now[df_now[" cause_1"] == causes[j]]
        T = little_df["T"].to_numpy()
        
        g_phat.append(S.temperature_model(T))
        n_events.append(len(T))
        pdf_values.append(gen_norm_pdf(np.arange(-12,35), g_phat[j][0], g_phat[j][1], 2) * (len(T)/len(df_now)))
        plt.plot(np.arange(-12,35),pdf_values[j],label = causes[j])
    plt.plot(eTs_df_stations.drop(columns = "station").iloc[i].to_numpy(),kernel_df.iloc[i][1:],label = "kernel density")
    plt.plot(np.arange(-12,35),np.sum(pdf_values,axis = 0),label = "sum")
    plt.title(f"{meta_stations.station_id.iloc[i]}")
    plt.legend()
    plt.show()
    
        
        
    

