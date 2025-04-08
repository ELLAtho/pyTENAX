# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:46:27 2025

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
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import to_rgba



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
# region_lats = [minlat,51,maxlat]
# region_lons = [minlon,9,maxlon]


country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30
region_lats = [minlat,31,35,41.3,maxlat]
region_lons = [minlon,maxlon]



# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 30
# region_lats = [minlat,37.5,maxlat]
# region_lons = [minlon,-116,-105,-90,maxlon]

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

missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    new_df = new_df.drop(missing_rows.index)
else:
    pass



## Read in the 
save_name = f"{drive}:/outputs/{country_save}\\average_temp_shape.csv"
df = pd.read_csv(save_name,dtype = {0:str})



average_filename = f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv"
eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(average_filename,dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()



peaks_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\peaks.csv")
save_name_skew = f"{drive}:/outputs/{country_save}\\temp_skew.csv"
skew_df = pd.read_csv(save_name_skew, dtype={"station":str})
AIC_savename = f"{drive}:/outputs/{country_save}\\AIC.csv"
AIC_df = pd.read_csv(AIC_savename,dtype = {"station":str})
skew_FRMSE_savename = f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv"
skew_FRMSE_df = pd.read_csv(skew_FRMSE_savename,dtype = {"station":str})
save_name = f"{drive}:/outputs/{country_save}\\temp_FRMSE.csv"
temp_FRMSE_df4 = pd.read_csv(save_name,dtype = {"station":str})
save_name = f"{drive}:/outputs/{country_save}\\temp_FRMSE6.csv"
temp_FRMSE_df6 = pd.read_csv(save_name,dtype = {"station":str})



FRMSE_6_4 =  temp_FRMSE_df6.FRMSE - temp_FRMSE_df4.FRMSE
FRMSE_skew_4 =  skew_FRMSE_df.FRMSE - temp_FRMSE_df4.FRMSE
FRMSE_6_4_20 =  temp_FRMSE_df6.FRMSE_upper_perc - temp_FRMSE_df4.FRMSE_upper_perc
FRMSE_skew_4_20 =  skew_FRMSE_df.FRMSE_upper_perc - temp_FRMSE_df4.FRMSE_upper_perc



AIC_6_4 = AIC_df.AIC6 - AIC_df.AIC
AIC_skew_4 = AIC_df.AIC_skew - AIC_df.AIC

#do rolling mean of the AIC differences
distance_savename = f"{drive}:/outputs/{country_save}\\distances_matrix.npy"
distances_matrix = np.load(distance_savename)

#calculate b as average
radius = 80


[0]*len(new_df.latitude)
FRMSE_6_4_roll =  [0]*len(new_df.latitude)
FRMSE_skew_4_roll =  [0]*len(new_df.latitude)
FRMSE_6_4_20_roll =  [0]*len(new_df.latitude)
FRMSE_skew_4_20_roll =  [0]*len(new_df.latitude)

AIC_6_4_roll = [0]*len(new_df.latitude)
AIC_skew_4_roll = [0]*len(new_df.latitude)

for i in range(len(new_df.latitude)):
    if pd.isna(new_df.b.iloc[i]):
        
        
        FRMSE_6_4_roll[i] = np.nan
        FRMSE_skew_4_roll[i] = np.nan
        FRMSE_6_4_20_roll[i] = np.nan
        FRMSE_skew_4_20_roll[i] = np.nan

        AIC_6_4_roll[i] = np.nan
        AIC_skew_4_roll[i] = np.nan
        
        # FRMSE_6_4_roll
        # FRMSE_skew_4_roll
        # FRMSE_6_4_20_roll
        # FRMSE_skew_4_20_roll

        # AIC_6_4_roll 
        # AIC_skew_4_roll
    else:
        station_distances = distances_matrix[i,:]
        close_locs = np.where(station_distances<=radius*1000)
        
        FRMSE_6_4_roll[i] = np.mean(FRMSE_6_4.iloc[close_locs])
        FRMSE_skew_4_roll[i] = np.mean(FRMSE_skew_4.iloc[close_locs])
        FRMSE_6_4_20_roll[i] = np.mean(FRMSE_6_4_20.iloc[close_locs])
        FRMSE_skew_4_20_roll[i] = np.mean(FRMSE_skew_4_20.iloc[close_locs])

        AIC_6_4_roll[i] = np.mean(AIC_6_4.iloc[close_locs])
        AIC_skew_4_roll[i] = np.mean(AIC_skew_4.iloc[close_locs])

lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
s = 5


fig = plt.figure(figsize=(15, 16))
norm = mcolors.Normalize(vmin=-0.3, vmax=0.3)
cmap = 'seismic'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(3, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = FRMSE_skew_4,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title(f"skew - beta = 4")
plt.colorbar(sc,extend = "both")


ax2 = fig.add_subplot(3, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = FRMSE_skew_4_20,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title(f"skew - beta = 4 upper 20%")
plt.colorbar(sc,extend = "both")

ax3 = fig.add_subplot(3, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = FRMSE_skew_4_roll,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax3.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax3.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax3.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax3.set_title(f"skew - beta = 4")
plt.colorbar(sc,extend = "both")



ax4 = fig.add_subplot(3, 2, 4, projection=proj)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax4.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = FRMSE_skew_4_20_roll,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax4.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax4.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax4.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax4.set_title("skew - beta = 4. upper 20%")
plt.colorbar(sc,extend = "both")


cmap = 'plasma'
bounds = [0.5,1.5,2.5,3.5,4.5]  # 3 discrete levels
norm = mcolors.BoundaryNorm(bounds, plt.get_cmap(cmap).N)

ax5 = fig.add_subplot(3, 2, 5, projection=proj)
ax5.coastlines()
ax5.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax5.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=peaks_df.n_peaks01,
    cmap=cmap,
    norm = norm,
    s = s
)
ax5.set_title("number of peaks (prominence = 0.001)")
ax5.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,5), crs=proj)
ax5.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
plt.colorbar(sc,ticks=[1, 2, 3, 4])

cmap = "seismic"
s = 3
norm = mcolors.Normalize(vmin=np.min(skew_df.skewness)*0.6, vmax=np.min(skew_df.skewness)*-0.6)
ax6 = fig.add_subplot(3, 2, 6, projection=proj)

# Add map features
ax6.coastlines()
ax6.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax6.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=skew_df.skewness,
    cmap=cmap,
    norm = norm,
    s = s
)
ax6.set_title("skewness")
ax6.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,5), crs=proj)
ax6.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
plt.colorbar(sc,extend = "both")




plt.show()

################################################################################
#lat lon regions
n_lat = len(region_lats)-1
n_lon = len(region_lons)-1

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

for lat_i in range(n_lat-1):
    ax1.plot([minlon-3,maxlon+3],[region_lats[lat_i+1],region_lats[lat_i+1]],  'r', linewidth=2, transform=ccrs.PlateCarree())

for lon_i in range(n_lon-1):
    ax1.plot([region_lons[lon_i+1],region_lons[lon_i+1]],[minlat-3,maxlat+3],  'r', linewidth=2, transform=ccrs.PlateCarree())

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
plt.show()


interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df_parameters)
for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        


if country == "Japan":
    
    fig = plt.figure(figsize=(4 * 5, n_lat * 5))

    axs = []  # We'll manually fill this with Axes
    for row in range(n_lat):
        row_axes = []
        for col in range(4):
            index = row * 4 + col + 1  # subplot index is 1-based
            if col % 2 == 1:  # Map subplot (1, 3)
                ax = fig.add_subplot(n_lat, 4, index, projection=proj)
            else:  # Regular line plot (0, 2)
                ax = fig.add_subplot(n_lat, 4, index)
            row_axes.append(ax)
        axs.append(row_axes)

    for lat_i in range(n_lat):
        for pwr in range(2):
            mask = (
                (df_parameters.latitude <= region_lats[lat_i + 1]) &
                (df_parameters.latitude > region_lats[lat_i]) &
                ((pd.DataFrame(FRMSE_skew_4_20_roll)[0]) * (-1) ** pwr > 0)
            )

            interp_y_region = np.array(interp_y)[mask]
            aves_region = np.array(aves)[mask]
            loc_region = df_parameters[mask]

            ax_line = axs[n_lat - 1 - lat_i][pwr * 2]
            for i in range(len(interp_y_region)):
                if not np.isnan(aves_region[i]):
                    ax_line.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")

            ax_line.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
            words = "worse" if pwr == 0 else "better"
            ax_line.set_title(f"Lat: {region_lats[lat_i]}–{region_lats[lat_i + 1]} | Skew {words}")
            ax_line.set_ylim(0, 0.5)

            ax_map = axs[n_lat - 1 - lat_i][pwr * 2 + 1]
            ax_map.coastlines()
            ax_map.add_feature(cfeature.BORDERS, linestyle=':')
            ax_map.scatter(loc_region.longitude, loc_region.latitude, transform=ccrs.PlateCarree())
            ax_map.set_ylim(minlat,maxlat)
            ax_map.set_xlim(minlon,maxlon)

    plt.tight_layout()
    plt.show()
    
    


