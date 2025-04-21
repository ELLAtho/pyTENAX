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

qs = [.85,.95,.99,.999]
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
P_full,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
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

skew_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv",dtype = {"station":str})
temp_FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE.csv",dtype = {"station":str})

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
cb.set_label("skewness [°C]",fontsize = fontsize)

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
norm = mcolors.Normalize(vmin=np.min(skew_df_DE.skewness)*0.4, vmax=np.min(skew_df_DE.skewness)*-0.4)
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
cb.set_label("skewness [°C]",fontsize = fontsize)

plt.show()


# fig 5c
#Japan



cmap = "seismic"
s = 3
norm = mcolors.Normalize(vmin=np.min(skew_df_JP.skewness)*0.4, vmax=np.min(skew_df_JP.skewness)*-0.4)
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
cb.set_label("skewness [°C]",fontsize = fontsize)

plt.show()






















