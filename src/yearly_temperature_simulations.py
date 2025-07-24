# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 08:41:55 2025

@author: ellar
"""

from os.path import dirname, abspath, join
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
import time

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from scipy.stats import norm
from scipy.optimize import minimize

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob


#### a simulation of all the temperatures during the year

### define the easy functions
A = 13 #average yearly temperature
B = 5 # range of temperature going up and down
p = 365.25/(2*np.pi) #period (aka a year)
shift = 180 # shift of starting point


var = 2 #variance of the normal distribution around
delta = 0.5 #dependence on the day
ave_loc = 0 #equivalent to shift, where the changes start in the year
daysize = 3

x = np.arange(0,365)
xhour = np.arange(0,365,1/24)

# definition of the function for mu in the temperature distribution over the year
def yearly_mu(x, A, B, shift = 0, p = 365.25/(2*np.pi), daysize = 0, daylength = 1/(2*np.pi)):
    return A + B*np.sin(x/p + shift/p) + daysize*np.sin(x/daylength)

def yearly_sigma(x, var, delta, ave_loc, p = 365.25/(2*np.pi)):
    return (1 + delta * np.sin(x/p + ave_loc/p))*var

def gen_sine_temperature_pdf(eT, A, B, var, delta, ave_loc, x= np.arange(0,365), p = 365.25/(2*np.pi), daysize = 0, daylength = 1/(2*np.pi)):
    mu = yearly_mu(x, A, B, p = p, daysize = daysize, daylength = daylength)
    sigma = yearly_sigma(x, var, delta, ave_loc, p = p)
    
    norms = [gen_norm_pdf(eT, mu[i], sigma[i], 2)/len(x) for i in x]
    pdf = sum(norms) #TODO: here you can put weighting as a storm filter
    return pdf
    

def sine_temperature_loglik_day_incl(theta, T, p = 365.25/(2*np.pi), daylength = 1/(2*np.pi)):
   
    A, B, var, delta = theta[0], theta[1], theta[2], theta[3] 
    
    ave_loc, daysize =theta[4], theta[5]
    
    pdf = gen_sine_temperature_pdf(T, A, B, var, delta, ave_loc, x= np.arange(0,365), p = p, daysize = daysize, daylength = daylength)
    
    return sum(np.log(pdf + 1e-10))

def sine_temperature_loglik(theta, T, p = 365.25/(2*np.pi)):
   
    A, B, var, delta, ave_loc = theta[0], theta[1], theta[2], theta[3], theta[4]
    
    
    pdf = gen_sine_temperature_pdf(T, A, B, var, delta, ave_loc, x= np.arange(0,365), p = p)
    
    return sum(np.log(pdf + 1e-10))

def sine_temperature_model(x, init_params = [13, 4, 3, 0.5, 90, 0], day_incl = False):
    if day_incl:
        phat = minimize(lambda theta: -sine_temperature_loglik_day_incl(theta, x),
                        init_params,
                        method='Nelder-Mead')
    else:
        init_params = init_params[0:-1]
        phat = minimize(lambda theta: -sine_temperature_loglik(theta, x),
                        init_params,
                        method='Nelder-Mead')
    
    return phat.x


theta = [A, B, var, delta, ave_loc, 0]

plt.plot(xhour,yearly_mu(xhour, A, B, shift, daysize = daysize))
plt.fill_between(xhour, 
                 yearly_mu(xhour, A, B, shift, daysize = daysize) - yearly_sigma(xhour, var, delta, ave_loc),
                 yearly_mu(xhour, A, B, shift, daysize = daysize) + yearly_sigma(xhour, var, delta, ave_loc),
                 alpha = 0.5)
plt.title(f"mu = {A} + {B}sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + {ave_loc})*2pi/365.25))*{var}")

plt.show()

eT = np.arange(-12,40,0.2)
norms_day = [gen_norm_pdf(eT, yearly_mu(xhour[i], A, B, shift, daysize = daysize),
                      yearly_sigma(xhour[i], var, delta, ave_loc), 2) for i in range(365*24)]

norms = [gen_norm_pdf(eT, yearly_mu(x[i], A, B, shift),
                      yearly_sigma(x[i], var, delta, ave_loc), 2) for i in range(365)]



plt.plot(eT,sum(norms)/(365), label = "daily ave")
plt.plot(eT,sum(norms_day)/(365*24), label = "incl diurnal cycle")
plt.title(f"mu = {A} + {B}sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + {ave_loc})*2pi/365.25))*{var}")
plt.legend()
plt.show()

## make a grid with some options
Bs = [2,4,6]
ave_locs = [0,90,180]


fig = plt.figure(figsize = (12,12))
for i in range(3):
    B = Bs[i]
    for j in range(3):
        ave_loc = ave_locs[j]
        
        ax = fig.add_subplot(3,3,1+i+3*j)
        ax.plot(x,yearly_mu(x, A, B, shift))
        plt.fill_between(x, 
                         yearly_mu(x, A, B, shift) - yearly_sigma(x, var, delta, ave_loc),
                         yearly_mu(x, A, B, shift) + yearly_sigma(x, var, delta, ave_loc),
                         alpha = 0.5)
        plt.title(f"B = {B}, ave_loc = {ave_loc}")
        
        plt.ylim(6,22)

plt.suptitle(f"mu = {A} + B*sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + ave_loc)*2pi/365.25))*{var}")
plt.show()



# eT = np.arange(0,30,0.2)

fig = plt.figure(figsize = (12,12))
for i in range(3):
    B = Bs[i]
    for j in range(3):
        ave_loc = ave_locs[j]
        
        norms = [gen_norm_pdf(eT, yearly_mu(x[i], A, B, shift),
                              yearly_sigma(x[i], var, delta, ave_loc), 2) for i in range(365)]
        
        ax = fig.add_subplot(3,3,1+i+3*j)
        ax.plot(eT,sum(norms)/365)
        plt.title(f"B = {B}, ave_loc = {ave_loc}")

plt.suptitle(f"mu = {A} + B*sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + ave_loc)*2pi/365.25))*{var}")

plt.show()


################################################################################
## Looking at some examples


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
    s = s,
    label = "all stations"
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

station_names = []
station_lats = []
station_lons = []
for i in range(len(region_lats)-1):
    for j in range(len(region_lons)-1):
        minlat_now = region_lats[i]
        maxlat_now = region_lats[i+1]
        minlon_now = region_lons[j]
        maxlon_now = region_lons[j+1]
        
        
        mask = (val_info.latitude >= minlat_now)&(
            val_info.latitude < maxlat_now)&(
                val_info.longitude >= minlon_now)&(
                    val_info.longitude < maxlon_now)
                    
        info_now = val_info[mask].sort_values(by="cleaned_years",ascending=False).reset_index()
        station_names.append(info_now.station.iloc[0])
        station_lats.append(info_now.latitude.iloc[0])
        station_lons.append(info_now.longitude.iloc[0])





s = 3
fontsize = 15

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1,1,1, projection=proj)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')



sc = ax.scatter(
    val_info.longitude,
    val_info.latitude,
    c="g",
    s = s,
    label = "all stations"
)

sc = ax.scatter(
    station_lons,
    station_lats,
    c="r",
    s = s*3,
    label = "chosen stations"
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


for i in range(len(station_names)):
    station = station_names[i]
    
    T_ = np.genfromtxt(f"D:/ordinary_events/US_main/T_{station}.csv")
    P_ = np.genfromtxt(f"D:/ordinary_events/US_main/P_{station}.csv")
    times = pd.read_csv(f"D:/ordinary_events/US_main/time_{station}.csv",parse_dates = ["oe_time"])
    
    
    oe = pd.DataFrame({
        "oe_time": times.oe_time,
        "T" : T_,
        "P" : P_
        })
    
    oe["date"] = pd.to_datetime(oe.oe_time).dt.strftime('%Y%m%d')
    
    
    phat = sine_temperature_model(T_)
    pdf = gen_sine_temperature_pdf(eT,*phat)
    
    eT_hist = np.arange(-20,40)
    eT_edges = np.concatenate([np.array([eT_hist[0]-(eT_hist[1]-eT_hist[0])/2]),(eT_hist + (eT_hist[1]-eT_hist[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(T_, bins=eT_edges, density=True)
    plt.plot(eT_hist, hist, '--')
    
    plt.plot(eT,pdf)
    plt.title(f"{station}. ({station_lats[i]},{station_lons[i]})")
    plt.show()
    















