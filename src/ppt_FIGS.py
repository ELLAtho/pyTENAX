# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:13:49 2025

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
from scipy.stats import ttest_ind

import datetime as dt
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

import xarray as xr
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from matplotlib import cm
import alphashape
from shapely.geometry import Polygon
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator

drive = 'D'


###############################################################################
# 3a and 3b
country = "germany"
country_save = "germany"
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15

info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
station = "03811"

T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{station}.csv",parse_dates = ["oe_time"])


S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
        beta = 4
    )


g_phat = S.temperature_model(T)
thr = np.quantile(P,S.left_censoring[1])
n = len(T)/info[info.station == station].cleaned_years


F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)

eT = np.arange(np.min(T),np.max(T)+4,1)


fontsize = 14

qs = [.85,.99]
TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [eT[0],eT[-1]])
plt.ylabel("Hourly precipitation (mm)",fontsize = fontsize)
plt.xlabel("T (°C)",fontsize = fontsize)
plt.xticks(fontsize = fontsize-2)
plt.yticks(fontsize = fontsize-2)
plt.legend(fontsize = fontsize)
plt.title("The magnitude model",fontsize = fontsize)
plt.show()

TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
plt.xlabel("T (°C)",fontsize = fontsize)
plt.ylabel("pdf",fontsize = fontsize)
plt.xticks(fontsize = fontsize-2)
plt.yticks(fontsize = fontsize-2)
plt.legend(fontsize = fontsize)
plt.title("The temperature model",fontsize = fontsize)
plt.show()


elevation = xr.load_dataarray(f"{drive}:/extras/elevation_europe/elev_ens_0.1deg_reg_v30.0e.nc")
fig = plt.figure(figsize=(4, 4))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS)
cnt = plt.pcolormesh(elevation.longitude,elevation.latitude,elevation
                     ,cmap = "terrain",alpha = 0.5,
                   transform=ccrs.PlateCarree())


plt.scatter(info[info.station == station].longitude,
            info[info.station == station].latitude, 
            s = 300,
            c = "r",
            marker = "x"
            )
plt.xlim(minlon,maxlon)
plt.ylim(minlat,maxlat)
ax1.set_xticks(np.arange(minlon,maxlon+1,1), crs=proj)
ax1.set_yticks(np.arange(minlat,maxlat+1,1), crs=proj)
plt.title("Station location")
plt.xticks(fontsize = fontsize-2)
plt.yticks(fontsize = fontsize-2)


plt.show()
####################################################################################
# fig 4c

file_name = f"{drive}:/{country}/DE_{station}"
P_full,data_meta = read_GSDR_file(f"{file_name}.txt","ppt")
T_path = f"{drive}:/{country}_temp\\DE_{station}.nc"
T_ERA = xr.load_dataarray(T_path)
T_full = (T_ERA.squeeze()-273.15).to_dataframe()

dates_range = [dt.datetime(1998,7,15,1),dt.datetime(1998,8,15,1)]


T_range = T[times.oe_time.between(dates_range[0],dates_range[1])]
P_range = P[times.oe_time.between(dates_range[0],dates_range[1])]
times_range = times[times.oe_time.between(dates_range[0],dates_range[1])]


fig = plt.figure(figsize = (3.5,3.5))
ax1 = fig.add_subplot(2,1,1)
plt.plot(P_full,zorder = 1)
plt.scatter(times_range,P_range,color = "r", marker = "x",zorder = 2)
plt.xlim(dates_range)
plt.ylim(0,15)

# for i in range(2,len(T_range)):
#     plt.text(times_range.oe_time.iloc[i]-dt.timedelta(days = 2),P_range[i]+0.5,f"P = {P_range[i]} mm/hr")


ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax1.tick_params(labelbottom=False)


plt.ylabel("Precipitation (mm/hr)")
ax1.set_title("Event separation = 24 hours")


ax2 = fig.add_subplot(2,1,2)
plt.plot(T_full.t2m,zorder = 1)
plt.scatter(times_range,T_range,color = "r", marker = "x",zorder = 2)

for i in range(len(T_range)):
    plt.plot([times_range.oe_time.iloc[i]-dt.timedelta(days = 1),times_range.oe_time.iloc[i]],[T_range[i],T_range[i]],color = "r")


# for i in range(2,len(T_range)):
#     plt.text(times_range.oe_time.iloc[i]-dt.timedelta(days = 2),T_range[i]+0.5,f"T = {T_range[i]:.1f} °C ")

plt.xlim(dates_range)
plt.ylim(5,35)

ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}°C"))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax2.tick_params(axis='x', which='both', top=True, labeltop=False)

plt.ylabel("Temperature")

plt.subplots_adjust(hspace=0)
plt.show()



################################################################################
# 5b1 (Germany)

df_germany = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape.csv",dtype = {0:str})

eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv",dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()

interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df_germany)
for i in np.arange(0,len(df_germany)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df_germany.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        

temp_aves_proper = np.nanmean(interp_y,axis =0)

T_mc = randdf(S.n_monte_carlo, np.vstack([temp_aves_proper, interp_x]), 'pdf').T

g_phat = S.temperature_model(T_mc)
pdf4 = gen_norm_pdf(interp_x,g_phat[0],g_phat[1],4)

fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1,1,1)

for i in np.arange(0,len(df_germany)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax.plot(interp_x ,interp_y[i],alpha = 0.01,color = "b")
plt.ylim(0,0.5)
plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")

ax.set_title("Germany temperature distributions")
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")
plt.show()

#5b1.1
fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1,1,1)

for i in np.arange(0,len(df_germany)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax.plot(interp_x ,interp_y[i],alpha = 0.01,color = "b")
plt.ylim(0,0.5)
plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")

ax.plot(interp_x,pdf4,color = "lime",label = "temp model beta = 4")

plt.legend()
ax.set_title("Germany temperature distributions")
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")
plt.show()

# 5a1 USA
country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
df_US = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape.csv",dtype = {0:str})
df_parameters_US = pd.read_csv(drive + ':/outputs/'+country_save+'\\parameters.csv', dtype={'station': str}) 


eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv",dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()

interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df_US)
for i in np.arange(0,len(df_US)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df_US.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        


peaks_df_US = pd.read_csv(f"{drive}:/outputs/{country_save}\\peaks.csv")
skew_df_US = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_skew.csv", dtype={"station":str})


temp_aves_proper = np.nanmean(interp_y,axis =0)
fig = plt.figure(figsize = (2,2))

mask = ((skew_df_US.skewness > 0) &
    (peaks_df_US.n_peaks01 == 1) &
    (df_parameters_US.longitude < -110))

interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters_US[mask]

ax_line = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax_line.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax_line.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
ax_line.set_title(u"USA temperature distributions \n West coast single peak")
ax_line.set_ylim(0, 0.5)
ax_line.set_xlabel("(T - μ)/σ")
ax_line.set_ylabel("Probability density")

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 0.5  # Smaller alpha = tighter wrap. Try tuning this value.
shape_a1 = alphashape.alphashape(points, alpha)



fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "b",
               transform=ccrs.PlateCarree())
ax_map.set_xlim(minlon, maxlon)
ax_map.set_ylim(minlat, maxlat)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.tight_layout()
plt.show()


# 5a2 USA

skew_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}/old_parameters_couple_month_missing\\temp_FRMSE_skew.csv",dtype = {"station":str})
temp_FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}/old_parameters_couple_month_missing\\temp_FRMSE.csv",dtype = {"station":str})

FRMSE_skew_4_20 =  skew_FRMSE_df.FRMSE_upper_perc - temp_FRMSE_df4.FRMSE_upper_perc

mask = ((skew_df_US.skewness > 0) &
    (peaks_df_US.n_peaks01 != 1) &
    (df_parameters_US.longitude < -90) &
    (FRMSE_skew_4_20 > 0)) #this is positive skew and 2 peaks and skew bad fit


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters_US[mask]

fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
ax.set_title(u"USA temperature distributions \n South mountains")
ax.set_ylim(0, 0.5)
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 0.3  # Smaller alpha = tighter wrap. Try tuning this value.
shape_a2 = alphashape.alphashape(points, alpha)


fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "r",
               transform=ccrs.PlateCarree())
ax_map.set_xlim(minlon, maxlon)
ax_map.set_ylim(minlat, maxlat)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.tight_layout()
plt.show()

# 5a3 USA

mask = ((skew_df_US.skewness < 0) &
    (peaks_df_US.n_peaks01 == 1) &
    (df_parameters_US.longitude > -110)
    ) #negative skew, one peak


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters_US[mask]

fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
ax.set_title(u"USA temperature distributions \n South East")
ax.set_ylim(0, 0.5)
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 0.5  # Smaller alpha = tighter wrap. Try tuning this value.
shape_a3 = alphashape.alphashape(points, alpha)


fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "r",
               transform=ccrs.PlateCarree())
ax_map.set_xlim(minlon, maxlon)
ax_map.set_ylim(minlat, maxlat)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.tight_layout()
plt.show()



# 5a4 USA

mask = ((skew_df_US.skewness < 0) &
    (peaks_df_US.n_peaks01 != 1) &
    (df_parameters_US.longitude > -100) &
    (df_parameters_US.latitude > 35)
    ) #negative skew, one peak


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters_US[mask]

fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
ax.set_title(u"USA temperature distributions \n North East")
ax.set_ylim(0, 0.5)
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 0.5  # Smaller alpha = tighter wrap. Try tuning this value.
shape_a4 = alphashape.alphashape(points, alpha)


fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "r",
               transform=ccrs.PlateCarree())
ax_map.set_xlim(minlon, maxlon)
ax_map.set_ylim(minlat, maxlat)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.tight_layout()
plt.show()



# 5c1 Japan

country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
 
df_JP = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape.csv",dtype = {0:str})
df_parameters_JP = pd.read_csv(drive + ':/outputs/'+country_save+'\\parameters.csv', dtype={'station': str}) 


eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv",dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()

interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df_JP)
for i in np.arange(0,len(df_JP)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df_JP.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        


peaks_df_JP = pd.read_csv(f"{drive}:/outputs/{country_save}\\peaks.csv")
skew_df_JP = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_skew.csv", dtype={"station":str})


temp_aves_proper = np.nanmean(interp_y,axis =0)
fig = plt.figure(figsize = (2,2))

mask = ((df_parameters_JP.latitude < 30)) # The islands South

interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters_JP[mask]

ax_line = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax_line.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax_line.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
ax_line.set_title(u"Japan temperature distributions \n South islands")
ax_line.set_ylim(0, 0.5)
ax_line.set_xlabel("(T - μ)/σ")
ax_line.set_ylabel("Probability density")

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 1.0  # Smaller alpha = tighter wrap. Try tuning this value.
shape_c1 = alphashape.alphashape(points, alpha)


fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "b",
               transform=ccrs.PlateCarree())
ax_map.set_xlim(minlon, maxlon)
ax_map.set_ylim(minlat, maxlat)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.tight_layout()
plt.show()


# 5c2 Japan

skew_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv",dtype = {"station":str})
temp_FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE.csv",dtype = {"station":str})

FRMSE_skew_4_20 =  skew_FRMSE_df.FRMSE_upper_perc - temp_FRMSE_df4.FRMSE_upper_perc

mask = ((skew_df_JP.skewness < 0) &
    (peaks_df_JP.n_peaks01 == 1) &
    (df_parameters_JP.latitude > 31)) #1 peak, neg skew


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters_JP[mask]

fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
ax.set_title(u"Japan temperature distributions \n Eastern side")
ax.set_ylim(0, 0.5)
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 1.0  # Smaller alpha = tighter wrap. Try tuning this value.
shape_c2 = alphashape.alphashape(points, alpha)


fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "r",
               transform=ccrs.PlateCarree())
ax_map.set_xlim(minlon, maxlon)
ax_map.set_ylim(minlat, maxlat)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.tight_layout()
plt.show()


# 5c3 Japan

skew_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv",dtype = {"station":str})
temp_FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE.csv",dtype = {"station":str})

FRMSE_skew_4_20 =  skew_FRMSE_df.FRMSE_upper_perc - temp_FRMSE_df4.FRMSE_upper_perc

mask = ((skew_df_JP.skewness > 0) &
    (peaks_df_JP.n_peaks01 != 1) &
    (df_parameters_JP.latitude > 31)) #this is positive skew and 2 peaks and skew bad fit


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters_JP[mask]


T_mc = randdf(S.n_monte_carlo, np.vstack([np.nanmean(interp_y_region, axis=0), interp_x]), 'pdf').T

g_phat = S.temperature_model(T_mc)
g_phat6 = S.temperature_model(T_mc,beta = 6)
g_phat8 = S.temperature_model(T_mc,beta = 8)

pdf4 = gen_norm_pdf(interp_x,g_phat[0],g_phat[1],4)
pdf6 = gen_norm_pdf(interp_x,g_phat6[0],g_phat6[1],6)
pdf8 = gen_norm_pdf(interp_x,g_phat8[0],g_phat8[1],8)



fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r", label = "mean")
ax.plot(interp_x,pdf4,color = "lime",label = "temp model beta = 4")
# ax.plot(interp_x,pdf6,color = "lime",label = "temp model beta = 6")
# ax.plot(interp_x,pdf8,color = "lime",label = "temp model beta = 8")


ax.set_title(u"Japan temperature distributions \n Western side")
ax.set_ylim(0, 0.5)
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")
plt.legend()
plt.show() 

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 1.0  # Smaller alpha = tighter wrap. Try tuning this value.
shape_c3 = alphashape.alphashape(points, alpha)

#version 2
fig = plt.figure(figsize = (2,2))
ax = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
# ax.plot(interp_x,pdf4,label = "temp model beta = 4")
# ax.plot(interp_x,pdf6,label = "temp model beta = 6")
ax.set_title(u"Japan temperature distributions \n Western side")
ax.set_ylim(0, 0.5)
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")
plt.show() 


fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "r",
               transform=ccrs.PlateCarree()
               )

if isinstance(shape_c3, Polygon):
    x, y = shape_c3.exterior.xy
    ax_map.plot(x, y, color='b', linewidth=2, label='Alpha Shape', transform=ccrs.PlateCarree())


ax_map.set_xlim(minlon, maxlon)
ax_map.set_ylim(minlat, maxlat)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.tight_layout()
plt.show()

###############################################################################

colors = ['m', 'g']
# fig 5a
# USA map
lon_lims = [truncate_neg(np.min(df_parameters_US.longitude),5),np.ceil(np.max(df_parameters_US.longitude/5))*5]
lat_lims = [truncate_neg(np.min(df_parameters_US.latitude),2.5),np.ceil(np.max(df_parameters_US.latitude/2.5))*2.5]
fontsize = 15

cmap = "seismic"
s = 3
norm = mcolors.Normalize(vmin=np.min(skew_df_US.skewness)*0.4, vmax=np.min(skew_df_US.skewness)*-0.4)
fig = plt.figure(figsize=(7, 3))

# delta_skew = skew_df_US.skewness/(np.sqrt(1 + (skew_df_US.skewness)**2))
# act_skew_US = ((4 - np.pi) / 2) * (delta_skew * np.sqrt(2 / np.pi)) / ((1 - 2 * delta_skew**2 / np.pi) ** (3 / 2))

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters_US.longitude,
    df_parameters_US.latitude,
    c=skew_df_US.skewness,
    cmap=cmap,
    norm = norm,
    s = s
)

#plot outlines of temps

shape_list = [shape_a1, shape_a2, shape_a3, shape_a4]

for i, shape in enumerate(shape_list):
    color = colors[i % 2]  # Alternate between 'm' and 'g'
    if isinstance(shape, Polygon):
        x, y = shape.exterior.xy
        plt.plot(x, y, color=color, linewidth=2, label='Alpha Shape', transform=ccrs.PlateCarree())
    else:
        for polygon in shape.geoms:
            x, y = polygon.exterior.xy
            plt.plot(x, y, color=color, linewidth=2, label='Alpha Shape', transform=ccrs.PlateCarree())

# ax1.set_title("Skewness of temperature distribution", fontsize = fontsize)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
cb = plt.colorbar(sc,extend = "both")
cb.set_label("alpha",fontsize = fontsize)

plt.show()



# fig 5b
#Germany
skew_df_DE = pd.read_csv(f"{drive}:/outputs/Germany\\temp_skew.csv", dtype={"station":str})
df_parameters_DE = pd.read_csv('D:/outputs/Germany\\parameters.csv', dtype={'station': str}) 

#merging the dataframes to ensure station consistency
missing_rows = pd.merge(df_parameters_DE.station, skew_df_DE.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters_DE = df_parameters_DE.drop(missing_rows.index)
else:
    pass


lon_lims = [truncate_neg(np.min(df_parameters_DE.longitude),5),np.ceil(np.max(df_parameters_DE.longitude/5))*5]
lat_lims = [truncate_neg(np.min(df_parameters_DE.latitude),2.5),np.ceil(np.max(df_parameters_DE.latitude/2.5))*2.5]
fontsize = 15

cmap = "seismic"
s = 3
#norm = mcolors.Normalize(vmin=np.min(skew_df_DE.skewness)*0.4, vmax=np.min(skew_df_DE.skewness)*-0.4)
fig = plt.figure(figsize=(7, 3))

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters_DE.longitude,
    df_parameters_DE.latitude,
    c=skew_df_DE.skewness,
    cmap=cmap,
    norm = norm,
    s = s
)

#plot outlines of temps

ax1.set_title("Skewness of temperature distribution", fontsize = fontsize)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
gl.xlocator = mticker.FixedLocator(np.arange(6, 16, 3))
gl.ylocator = mticker.FixedLocator(np.arange(48, 56, 2))
cb = plt.colorbar(sc,extend = "both")
cb.set_label("alpha",fontsize = fontsize)

plt.show()


# fig 5c
#Japan



cmap = "seismic"
s = 3
#norm = mcolors.Normalize(vmin=np.min(skew_df_JP.skewness)*0.4, vmax=np.min(skew_df_JP.skewness)*-0.4)
fig = plt.figure(figsize=(7, 3))

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters_JP.longitude,
    df_parameters_JP.latitude,
    c=skew_df_JP.skewness,
    cmap=cmap,
    norm = norm,
    s = s
)

#plot outlines of temps

shape_list = [shape_c1, shape_c2, shape_c3]

for i, shape in enumerate(shape_list):
    color = colors[i % 2]  # Alternate between 'm' and 'g'
    if isinstance(shape, Polygon):
        x, y = shape.exterior.xy
        plt.plot(x, y, color=color, linewidth=2, label='Alpha Shape', transform=ccrs.PlateCarree())
    else:
        for polygon in shape.geoms:
            x, y = polygon.exterior.xy
            plt.plot(x, y, color=color, linewidth=2, label='Alpha Shape', transform=ccrs.PlateCarree())

# ax1.set_title("Skewness of temperature distribution", fontsize = fontsize)


gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
cb = plt.colorbar(sc,extend = "both")
cb.set_label("alpha",fontsize = fontsize)

plt.show()

#################################################################################

#fig 6a


country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9

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
    new_df = df_parameters[['station','latitude','longitude','b','kappa','lambda','a','mu','sigma','thr','n_events_per_yr']].copy()
    
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
    df_parameters = df_parameters.reindex(index = range(len(df_parameters)))
    new_df = new_df.drop(missing_rows.index)
    new_df = new_df.reindex(index = range(len(new_df)))
else:
    pass


norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
s = 1

#plot at 5% sig
fig = plt.figure(figsize=(4, 4))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 3)
ax1.add_feature(cfeature.BORDERS, linestyle=':')

# Choosing cmap


sc = ax1.scatter(
    df_parameters.longitude[df_parameters.b==0],
    df_parameters.latitude[df_parameters.b==0],transform=ccrs.PlateCarree(),
    s = s,
    color = 'darkgrey',zorder = 1
)

sc = ax1.scatter(
    df_parameters.longitude[df_parameters.b!=0],
    df_parameters.latitude[df_parameters.b!=0],transform=ccrs.PlateCarree(),
    c=df_parameters.b[df_parameters.b!=0],
    s = s,
    cmap='seismic',  
    norm=norm,zorder = 2
)



# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15,extend = "both")
cb.set_label('b', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
cb.ax.tick_params(labelsize=12)
arrow1 = patches.FancyArrowPatch((135, 39), [float(new_df.longitude[new_df.station == "12261"]),float(new_df.latitude[new_df.station == "12261"])], transform=ccrs.PlateCarree(), color='red', arrowstyle='->', mutation_scale=20,zorder = 6)
arrow2 = patches.FancyArrowPatch((145, 33), [float(new_df.longitude[new_df.station == "19376"]),float(new_df.latitude[new_df.station == "19376"])], transform=ccrs.PlateCarree(), color='green', arrowstyle='->', mutation_scale=20,zorder = 6)

ax1.add_patch(arrow1)
ax1.add_patch(arrow2)

ax1.scatter(134.5, 38.5,
            s = 400,
            c = "r", transform=ccrs.PlateCarree(),
            marker = ".",zorder = 3)
ax1.scatter(145, 31.5,
            s = 300,
            c = "g", transform=ccrs.PlateCarree(),
            marker = "*",zorder = 4)

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}

plt.title("Significant b", fontsize=16)
plt.show()

fig = plt.figure(figsize=(4, 4))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines(zorder = 2)
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = new_df.b,transform=ccrs.PlateCarree(),
    s = s,
    cmap = 'seismic',
    norm = norm,zorder = 1
)

arrow1 = patches.FancyArrowPatch((135, 39), [float(new_df.longitude[new_df.station == "12261"]),float(new_df.latitude[new_df.station == "12261"])], transform=ccrs.PlateCarree(), color='red', arrowstyle='->', mutation_scale=20,zorder = 6)
arrow2 = patches.FancyArrowPatch((145, 33), [float(new_df.longitude[new_df.station == "19376"]),float(new_df.latitude[new_df.station == "19376"])], transform=ccrs.PlateCarree(), color='green', arrowstyle='->', mutation_scale=20,zorder = 6)

ax1.add_patch(arrow1)
ax1.add_patch(arrow2)

ax1.scatter(134.5, 38.5,
            s = 400,
            c = "r", transform=ccrs.PlateCarree(),
            marker = ".",zorder = 3)
ax1.scatter(145, 31.5,
            s = 300,
            c = "g", transform=ccrs.PlateCarree(),
            marker = "*",zorder = 4)


# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15,extend = "both")
cb.set_label('b', fontsize=14)  
cb.ax.tick_params(labelsize=12)

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}

plt.title(f'All b', fontsize=16)
plt.show()

plt.scatter(new_df.longitude[new_df.station == "12261"],
            new_df.latitude[new_df.station == "12261"],
            s = 400,
            c = "r",
            marker = ".",zorder = 3)
plt.scatter(new_df.longitude[new_df.station == "19376"],
            new_df.latitude[new_df.station == "19376"],
            s = 300,
            c = "g",
            marker = "*",zorder = 4)
plt.show()
# fig 6b

fig = plt.figure(figsize=(4, 4))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter( #plot the negligable at 5% lvl points
    df_parameters_exp.longitude,
    df_parameters_exp.latitude,
    c = df_parameters_exp.b,
    s = s,
    cmap = 'seismic',
    norm = norm
)



# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.15,extend = "both")
cb.set_label('b', fontsize=14)  
cb.ax.tick_params(labelsize=12)

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}

plt.title(f'All b exp', fontsize=16)
plt.show()


# fig 6 0

RL_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\return_levels.csv", dtype={'station': str})
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



station_free = "12261"
station_0 = "19376"


T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station_0}.csv")
P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station_0}.csv")
eT = np.arange(np.min(T),np.max(T)+4,1)

kde  = gaussian_kde(T) #use kernel density to get probability




T_min = np.min(T)
T_max = np.max(T)
Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)




g_phat = [new_df[new_df.station == station_0].mu.to_numpy(),new_df[new_df.station == station_0].sigma.to_numpy()]

F_phat = [new_df[new_df.station == station_0].kappa.to_numpy(),
          new_df[new_df.station == station_0].b.to_numpy(),
          new_df[new_df.station == station_0]["lambda"].to_numpy(),
          new_df[new_df.station == station_0].a.to_numpy()]

F_phat_0 = [df_parameters_0[df_parameters_0.station == station_0].kappa.to_numpy(),
          df_parameters_0[df_parameters_0.station == station_0].b.to_numpy(),
          df_parameters_0[df_parameters_0.station == station_0]["lambda"].to_numpy(),
          df_parameters_0[df_parameters_0.station == station_0].a.to_numpy()]

thr = new_df[new_df.station == station_0].thr
n = new_df[new_df.station == station_0].n_events_per_yr

AMS = RL_df[new_df.station == station_0].obs_AMS.to_numpy()[0]

plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))

eRP = 1/(1-plot_pos)

S.return_period = eRP

pdf_values = kde(Ts)
df = np.vstack([pdf_values, Ts + 2]) # shifted by 2 degrees

T_mc = randdf(S.n_monte_carlo, df, 'pdf').T              


wbl_phat_0 = np.column_stack((
                            F_phat_0[2] * np.exp(F_phat_0[3] * T_mc),
                            F_phat_0[0] + F_phat_0[1] * T_mc
                            ))



vguess = 10 ** np.arange(np.log10(0.05), np.log10(5e2), 0.05)
RL_2 = SMEV_Mc_inversion(wbl_phat_0, n, S.return_period, vguess, method_root_scalar="brentq")



RL = RL_df[new_df.station == station_0].return_levels_kernal.to_numpy()[0]
RL_0 = RL_df[new_df.station == station_0].return_levels_kernal_0.to_numpy()[0]
RL_exp = RL_df[new_df.station == station_0].return_levels_kernal_exp.to_numpy()[0]

fontsize = 15



fig = plt.figure(figsize = (5,4))

percentile_lines = inverse_magnitude_model(F_phat,eT,qs,b_exp=False)
plt.scatter(T,P,s=1,color="r",label = 'observations')
plt.plot(eT,[thr]*np.size(eT),'--',alpha = 0.5,color = 'k',label = 'Left censoring threshold') #plot threshold

#first one outside loop so can be in legend
n=0
plt.plot(eT,percentile_lines[n],label = 'Magnitude model W(x,T), b = linear',color = "b")
plt.text(eT[-1], percentile_lines[n][-1], str(qs[n]*100)+'th', ha='left', va='center')
n=1
while n<np.size(qs):
    plt.plot(eT,percentile_lines[n],color = "b")
    plt.text(eT[-1], percentile_lines[n][-1], str(qs[n]*100)+'th', ha='left', va='center')
    n=n+1

percentile_lines = inverse_magnitude_model(F_phat_0,eT,qs,b_exp=False)
n=0
plt.plot(eT,percentile_lines[n],"--",label = 'Magnitude model W(x,T), b = 0',color = "g")
#plt.text(eT[-1], percentile_lines[n][-1], str(qs[n]*100)+'th', ha='left', va='center')
n=1
while n<np.size(qs):
    plt.plot(eT,percentile_lines[n],"--",color = "g")
    #plt.text(eT[-1], percentile_lines[n][-1], str(qs[n]*100)+'th', ha='left', va='center')
    n=n+1

plt.yscale('log')

plt.ylabel("Precipitation (mm/hr)",fontsize = fontsize)
plt.xlabel("T (°C)",fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.legend(fontsize = fontsize-3)
plt.xlim(np.min(eT),np.max(eT))
plt.show()

fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(1,1,1)
plt.plot(eRP,RL_0,label = f"b = 0")
plt.plot(eRP,RL,"--",label = "b = linear")  #plot TENAX return levels
plt.plot(eRP,RL_exp,"--",label = f"b = exp")
plt.plot(eRP,AMS,"k+",label = "annual maxima") #plot observed return levels

plt.ylim(0,45)
plt.xscale('log')
plt.xlabel('return period (years)')

plt.xticks([1,3,10,30])
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.title("with temperature kernel for temperature model")
plt.legend()
plt.show()



#################################################################
#with plus 2 deg
fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(1,1,1)
plt.plot(eRP,AMS,"k+",label = "annual maxima")
plt.plot(eRP,RL_0,label = f"fitted")
plt.plot(eRP,RL_2,"--",label = "+ 2 °C")  

plt.ylim(0,45)
plt.xscale('log')
plt.xlabel('return period (years)')
plt.ylabel("Precipitation (mm/hr)")
plt.xticks([1,3,10,30])
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.title("with temperature kernel for temperature model")
plt.legend()
plt.show()



station_0 = station_free

T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station_0}.csv")
P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station_0}.csv")
eT = np.arange(np.min(T),np.max(T)+4,1)

T_min = np.min(T)
T_max = np.max(T)
Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)


g_phat = [new_df[new_df.station == station_0].mu.to_numpy(),new_df[new_df.station == station_0].sigma.to_numpy()]

F_phat = [new_df[new_df.station == station_0].kappa.to_numpy(),
          new_df[new_df.station == station_0].b.to_numpy(),
          new_df[new_df.station == station_0]["lambda"].to_numpy(),
          new_df[new_df.station == station_0].a.to_numpy()]


thr = new_df[new_df.station == station_0].thr
n = new_df[new_df.station == station_0].n_events_per_yr

RL = RL_df[new_df.station == station_0].return_levels_kernal.to_numpy()[0]
AMS = RL_df[new_df.station == station_0].obs_AMS.to_numpy()[0]
RL_0 = RL_df[new_df.station == station_0].return_levels_kernal_0.to_numpy()[0]
RL_exp = RL_df[new_df.station == station_0].return_levels_kernal_exp.to_numpy()[0]

plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))

eRP = 1/(1-plot_pos)


fig = plt.figure(figsize = (4,4))
TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
plt.ylabel("Precipitation (mm/hr)",fontsize = fontsize)
plt.xlabel("T (°C)",fontsize = fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.legend(fontsize = fontsize)
plt.title(f"({station_0}. ")
plt.show()

fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(1,1,1)
plt.plot(eRP,RL_0,label = f"b = 0")
plt.plot(eRP,RL,"--",label = "b = linear")  #plot TENAX return levels
plt.plot(eRP,RL_exp,"--",label = f"b = exp")
plt.plot(eRP,AMS,"k+",label = "annual maxima") #plot observed return levels

plt.ylim(0,45)
plt.xscale('log')
plt.xlabel('return period (years)')

plt.xticks([1,3,10,30])
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.title("with temperature kernel for temperature model")
plt.legend()
plt.show()






###############################################################################
# slide 7: return levels
RL_future_10_savename = f"{drive}:/outputs/{country_save}\\RL_future_10_kernel.csv"
RL_future_10_kernel = pd.read_csv(RL_future_10_savename, dtype = {"station" : str})



RL10_df = pd.DataFrame({
    "return_levels":np.zeros(len(new_df)),
    "return_levels_0":np.zeros(len(new_df)),
    "return_levels_exp":np.zeros(len(new_df)),
    })

for i in np.arange(0,len(new_df)):
    plot_pos = np.arange(1,np.size(RL_df.obs_AMS.iloc[i])+1)/(1+np.size(RL_df.obs_AMS.iloc[i]))

    eRP = 1/(1-plot_pos)
    RL_free = RL_df.return_levels_kernal.iloc[i]
    RL_0 = RL_df.return_levels_kernal_0.iloc[i]
    RL_exp = RL_df.return_levels_kernal_exp.iloc[i]
    
    if np.size(RL_free) == 1:
        RL10_df.loc[i, 'return_levels'] = np.nan
    else:
        interp_func = interp1d(eRP, RL_free)
        RL10_df.loc[i, 'return_levels'] = interp_func(10)
        
    if np.size(RL_0) == 1:
        RL10_df.loc[i, 'return_levels_0'] = np.nan
    else:
        interp_func = interp1d(eRP, RL_0)
        RL10_df.loc[i, 'return_levels_0'] = interp_func(10)
    
    if np.size(RL_exp) == 1:
        RL10_df.loc[i, 'return_levels_exp'] = np.nan
    else:
        interp_func = interp1d(eRP, RL_exp)
        RL10_df.loc[i, 'return_levels_exp'] = interp_func(10)
    





# Define the boundaries and number of bins for the discrete colormap
num_bins = 12
cmap = plt.cm.rainbow  # You can use 'rainbow' or any other colormap
norm = mcolors.BoundaryNorm(boundaries=np.linspace(10, 70, num_bins + 1), ncolors=num_bins)

# Create a discrete colormap and add black for values above 70
colors = cmap(np.linspace(0, 1, num_bins))
colors = np.vstack([colors, [0, 0, 0, 1]])  # Add black as the last color
discrete_cmap = mcolors.ListedColormap(colors)
# Create the figure
fig = plt.figure(figsize=(10, 10))

proj = ccrs.PlateCarree()

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL10_df.return_levels,
    cmap=discrete_cmap,
    norm=norm,
    s=s,
)
gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
ax1.set_title("free")

# Second subplot
ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL10_df.return_levels_0,
    cmap=discrete_cmap,
    norm=norm,
    s=s,
)
gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
ax2.set_title("b = 0")

# Third subplot
ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL10_df.return_levels_exp,
    cmap=discrete_cmap,
    norm = norm,
    s=s,
)
gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
ax3.set_title("b exp")


norm2 = mcolors.Normalize(vmin=-20, vmax=20)
# Fourth subplot
ax4 = fig.add_subplot(2, 2, 4, projection=proj)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS, linestyle=':')
sc4 = ax4.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=RL10_df.return_levels - RL10_df.return_levels_0,
    cmap="bwr",  # You can use discrete colormap here if desired
    s=s,
    norm = norm2
)
gl = ax4.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
fig.subplots_adjust(right=0.85)


cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])
cb4 = plt.colorbar(sc4, cax=cbar_ax4,extend = "both")
cb4.set_label('b = linear - b = 0', fontsize=14)
cb4.ax.tick_params(labelsize=12)

# Colorbar for the first three subplots
cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
#cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
#cb = mcolors.colorbarbase(sc,cax=cbar_ax, cmap=discrete_cmap, norm=norm, orientation='horizontal')
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=discrete_cmap),
    cax=cbar_ax,
    orientation='horizontal',
    extend='max'
)

cb.set_label('10 year 1 hour return level (mm)', fontsize=14)
cb.ax.tick_params(labelsize=12)



fig.suptitle(f'{ERA_country} 10 year return levels. kernel density', fontsize=16)
plt.show()



cmap = "YlGnBu"
norm = mcolors.Normalize(vmin=0, vmax=70)


fig = plt.figure(figsize = (8,3))
ax1 = fig.add_subplot(1, 3, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL10_df.return_levels_0,
    cmap=cmap,
    norm=norm,
    s=s,
)
gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
gl.ylocator = MultipleLocator(5)
ax1.set_title("Observations 1976 - 2009")

ax2 = fig.add_subplot(1, 3, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL_future_10_kernel.RL_10,
    cmap=cmap,
    norm=norm,
    s=s,
)
gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
gl.ylocator = MultipleLocator(5)
ax2.set_title("+ 2°C")

cmap2 = "Blues"
norm2 = mcolors.Normalize(vmin=0, vmax=15)
ax3 = fig.add_subplot(1, 3, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL_future_10_kernel.RL_10 - RL10_df.return_levels_0,
    cmap=cmap2,
    norm=norm2,
    s=s,
)

gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': fontsize-2}
gl.ylabel_style = {'size': fontsize-2}
gl.ylocator = MultipleLocator(5)
ax3.set_title("Difference")

cbar_ax2 = fig.add_axes([0.735,  0.02, 0.235, 0.05])
cb2 = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm2, cmap=cmap2),
    cax=cbar_ax2,
    orientation='horizontal',
    extend='max'
)
cb2.set_label('Difference (mm/hr)', fontsize=14)
cb2.ax.tick_params(labelsize=12)
#([0.15, 0.02, 0.7, 0.03])

cbar_ax = fig.add_axes([0.08,  0.02, 0.56, 0.05])  # Position for the colorbar
#cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
#cb = mcolors.colorbarbase(sc,cax=cbar_ax, cmap=discrete_cmap, norm=norm, orientation='horizontal')
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cbar_ax,
    orientation='horizontal',
    extend='max'
)

cb.set_label('10 year return level (mm/hr)', fontsize=14)
cb.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()



























