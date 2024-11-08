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


year_bins = np.arange(info_full['cleaned_years'].min(), info_full['cleaned_years'].max() + 10, 10)

# Create a discrete colormap with one color per bin
cmap = plt.get_cmap("viridis", len(year_bins) - 1)  # Replace "viridis" with your preferred colormap
norm = mcolors.BoundaryNorm(year_bins, cmap.N)


#BY YEARS
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
sc = ax1.scatter(
    info_full['longitude'], 
    info_full['latitude'], 
    s=1.5, 
    c=info_full['cleaned_years'], 
    cmap=cmap, 
    norm=norm, 
    transform=ccrs.PlateCarree()
)

# Add color bar at the bottom of the plot with discrete labels
cbar = plt.colorbar(sc, ax=ax1, orientation='horizontal', pad=0.05, fraction=0.05)
cbar.set_label("Cleaned Years")
cbar.set_ticks(year_bins)
plt.show()


#EUROPE
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
sc = ax1.scatter(
    info_full['longitude'], 
    info_full['latitude'], 
    s=1.5, 
    c=info_full['cleaned_years'], 
    cmap=cmap, 
    norm=norm, 
    transform=ccrs.PlateCarree()
)

# Add color bar at the bottom of the plot with discrete labels
cbar = plt.colorbar(sc, ax=ax1, orientation='horizontal', pad=0.05, fraction=0.05)
cbar.set_label("Cleaned Years")
cbar.set_ticks(year_bins)
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


    
#FOCUS ON JAPAN
n=5

info_J = info[n]
info_J.station = info_J['station'].apply(lambda x: f'{int(x):05}') #convert station id into actual station id

info_J_sort = info_J.sort_values(by=['cleaned_years'],ascending=0)

for i in np.arange(0,10):
    plt.plot([info_J_sort.startdate[info_J_sort.index[i]],info_J_sort.enddate[info_J_sort.index[i]]],[i,i],label = info_J_sort.station[info_J_sort.index[i]])
   
plt.legend()
plt.show()




for i in np.arange(1000,1010):
    plt.plot([info_J_sort.startdate[info_J_sort.index[i]],info_J_sort.enddate[info_J_sort.index[i]]],[i,i],label = info_J_sort.station[info_J_sort.index[i]])
   
plt.legend()
plt.show()
