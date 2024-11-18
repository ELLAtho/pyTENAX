# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:59:50 2024

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

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

countries = ['ISD','Belgium','Finland','Germany','Ireland','Japan','Norway','Portugal','UK','US']

# latslons
# n=0
# while n<len(countries):
#     info[n]['startdate'] = info[n]['startdate'].apply(lambda x: dt.datetime.strptime("{:10.0f}".format(x), "%Y%m%d%H"))
#     info[n]['enddate'] = info[n]['enddate'].apply(lambda x: dt.datetime.strptime("{:10.0f}".format(x), "%Y%m%d%H"))
#     info[n].to_csv('D:/metadata/'+countries[n]+'_fulldata.csv')
#     n=n+1
    

info = []



for c in countries:
    dd = pd.read_csv('D:/metadata/'+c+'_fulldata.csv')
    info.append(dd)
    
info_full = pd.concat(info,axis=0)    

#yr bins
info_10 = info_full[info_full.cleaned_years<10]
info_20 = info_full[(info_full.cleaned_years<20)&(info_full.cleaned_years>=10)]
info_30 = info_full[(info_full.cleaned_years<30)&(info_full.cleaned_years>=20)]
info_30up = info_full[(info_full.cleaned_years>=30)]

#stupid version of truncating
def truncate_to_one_decimal(num):
    if num >= 0:
        return int(num * 10) / 10  # Truncate positive number
    else:
        return int(num * 10 - 1) / 10  # Truncate negative number to be more negative



loc_dic = {}

for n in np.arange(0,len(countries)):
    loc_dic[countries[n]] = {
        'minlat': truncate_to_one_decimal(np.min(info[n].latitude)),
        'maxlat': np.ceil(np.max(info[n].latitude)*10)/10,
        'minlon': truncate_to_one_decimal(np.min(info[n].longitude)),
        'maxlon': np.ceil(np.max(info[n].longitude)*10)/10,
        'startdate' : np.min(info[n].startdate),
        'enddate' : np.max(info[n].enddate)
    }


#BY DATASET
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
n=0
while n<len(countries):
    plt.scatter(info[n]['longitude'],info[n]['latitude'],s=1.5,label = countries[n],transform=ccrs.PlateCarree())
    n=n+1
plt.legend()
plt.show()




#BY YEARS
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()

ax1.scatter(info_10.longitude,info_10.latitude,s=1.5, color = 'red',label = '<10')
ax1.scatter(info_20.longitude,info_20.latitude,s=1.5, color = 'orange',label = '10-19')
ax1.scatter(info_30.longitude,info_30.latitude,s=1.5, color = 'gold',label = '20-29')
ax1.scatter(info_30.longitude,info_30.latitude,s=1.5, color = 'green',label = '30+')
plt.legend()
plt.show()


#EUROPE
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()

ax1.scatter(info_10.longitude,info_10.latitude, color = 'red',label = '<10')
ax1.scatter(info_20.longitude,info_20.latitude, color = 'orange',label = '10-19')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'gold',label = '20-29')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'green',label = '30+')
plt.legend()
plt.xlim(-10,45)
plt.ylim(20,75)

plt.show()


#JAPAN
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
sc = plt.scatter(info_full['longitude'],info_full['latitude'],s=1.5,c=info_full['cleaned_years'],transform=ccrs.PlateCarree())
plt.xlim(110,150)
plt.ylim(-10,48)
plt.show()


#ISLANDS
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
sc = plt.scatter(info_full['longitude'],info_full['latitude'],s=1.5,c=info_full['cleaned_years'],transform=ccrs.PlateCarree())
plt.xlim(130,170)
plt.ylim(-10,30)
plt.show()



#US
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
sc = plt.scatter(info_full['longitude'],info_full['latitude'],s=1.5,c=info_full['cleaned_years'],transform=ccrs.PlateCarree())
plt.xlim(-140,-50)
plt.ylim(20,75)
plt.show()


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
sc = plt.scatter(info_full['longitude'],info_full['latitude'],s=1.5,c=info_full['cleaned_years'],transform=ccrs.PlateCarree())
ax1.set_xticks(np.arange(-180,190,20), crs=proj)
ax1.set_yticks(np.arange(-90,100,20), crs=proj)
plt.show()
