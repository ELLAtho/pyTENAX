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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from matplotlib import cm
import alphashape
from shapely.geometry import Polygon


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
################################################################################
# 5b1 (Germany)

df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape.csv",dtype = {0:str})

eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv",dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()

interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df)
for i in np.arange(0,len(df)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        

temp_aves_proper = np.nanmean(interp_y,axis =0)



fig = plt.figure(figsize = (3,3))
ax = fig.add_subplot(1,1,1)

for i in np.arange(0,len(df)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax.plot(interp_x ,interp_y[i],alpha = 0.01,color = "b")

plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")
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
df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape.csv",dtype = {0:str})
df_parameters = pd.read_csv(drive + ':/outputs/'+country_save+'\\parameters.csv', dtype={'station': str}) 


eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv",dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()

interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df)
for i in np.arange(0,len(df)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        


peaks_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\peaks.csv")
skew_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_skew.csv", dtype={"station":str})


temp_aves_proper = np.nanmean(interp_y,axis =0)
fig = plt.figure(figsize = (3,3))

mask = ((skew_df.skewness > 0) &
    (peaks_df.n_peaks01 == 1) &
    (df_parameters.longitude < -110))

interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters[mask]

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

mask = ((skew_df.skewness > 0) &
    (peaks_df.n_peaks01 != 1) &
    (df_parameters.longitude < -90) &
    (FRMSE_skew_4_20 > 0)) #this is positive skew and 2 peaks and skew bad fit


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters[mask]

fig = plt.figure(figsize = (3,3))
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

mask = ((skew_df.skewness < 0) &
    (peaks_df.n_peaks01 == 1) &
    (df_parameters.longitude > -110)
    ) #negative skew, one peak


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters[mask]

fig = plt.figure(figsize = (3,3))
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

mask = ((skew_df.skewness < 0) &
    (peaks_df.n_peaks01 != 1) &
    (df_parameters.longitude > -100) &
    (df_parameters.latitude > 35)
    ) #negative skew, one peak


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters[mask]

fig = plt.figure(figsize = (3,3))
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
 
df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape.csv",dtype = {0:str})
df_parameters = pd.read_csv(drive + ':/outputs/'+country_save+'\\parameters.csv', dtype={'station': str}) 


eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
average_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv",dtype = {"station":str})
aves = average_df.aves.to_numpy()
sds = average_df.sds.to_numpy()
eTs = eTs_df.drop(columns = "station").to_numpy()

interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df)
for i in np.arange(0,len(df)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        


peaks_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\peaks.csv")
skew_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_skew.csv", dtype={"station":str})


temp_aves_proper = np.nanmean(interp_y,axis =0)
fig = plt.figure(figsize = (3,3))

mask = ((df_parameters.latitude < 30)) # The islands South

interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters[mask]

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


# 5a2 Japan

skew_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv",dtype = {"station":str})
temp_FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE.csv",dtype = {"station":str})

FRMSE_skew_4_20 =  skew_FRMSE_df.FRMSE_upper_perc - temp_FRMSE_df4.FRMSE_upper_perc

mask = ((skew_df.skewness < 0) &
    (peaks_df.n_peaks01 == 1) &
    (df_parameters.latitude > 31)) #1 peak, neg skew


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters[mask]

fig = plt.figure(figsize = (3,3))
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


# 5a3 Japan

skew_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv",dtype = {"station":str})
temp_FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE.csv",dtype = {"station":str})

FRMSE_skew_4_20 =  skew_FRMSE_df.FRMSE_upper_perc - temp_FRMSE_df4.FRMSE_upper_perc

mask = ((skew_df.skewness > 0) &
    (peaks_df.n_peaks01 != 1) &
    (df_parameters.latitude > 31)) #this is positive skew and 2 peaks and skew bad fit


interp_y_region = np.array(interp_y)[mask]
aves_region = np.array(aves)[mask]
loc_region = df_parameters[mask]

pdf4 = gen_norm_pdf(interp_x,0,2,4)
pdf6 = gen_norm_pdf(interp_x,0,2,6)

fig = plt.figure(figsize = (3,3))
ax = fig.add_subplot(1, 1, 1)
for i in range(len(interp_y_region)):
    if not np.isnan(aves_region[i]):
        ax.plot(interp_x, interp_y_region[i], alpha=0.1, color="b")
if interp_y_region.size > 0:
    ax.plot(interp_x, np.nanmean(interp_y_region, axis=0), color="r")
ax.plot(interp_x,pdf4,label = "temp model beta = 4")
ax.plot(interp_x,pdf6,label = "temp model beta = 6")
ax.set_title(u"Japan temperature distributions \n Western side")
ax.set_ylim(0, 0.5)
ax.set_xlabel("(T - μ)/σ")
ax.set_ylabel("Probability density")

points = np.column_stack((loc_region.longitude, loc_region.latitude))
alpha = 1.0  # Smaller alpha = tighter wrap. Try tuning this value.
shape = alphashape.alphashape(points, alpha)


fig = plt.figure()
ax_map = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.scatter(loc_region.longitude, loc_region.latitude,
               color = "r",
               transform=ccrs.PlateCarree()
               )

if isinstance(shape, Polygon):
    x, y = shape.exterior.xy
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
#





