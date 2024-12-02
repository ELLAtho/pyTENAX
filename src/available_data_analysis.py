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

drive='F' #name of drive
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
    dd = pd.read_csv(drive+':/metadata/'+c+'_fulldata.csv')
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
plt.xlim(-10,50)
plt.ylim(20,80)

plt.show()


#JAPAN
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.scatter(info_10.longitude,info_10.latitude, color = 'red',label = '<10')
ax1.scatter(info_20.longitude,info_20.latitude, color = 'orange',label = '10-19')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'gold',label = '20-29')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'green',label = '30+')
plt.xlim(90,180)
plt.ylim(0,60)
plt.show()


#ISLANDS
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.scatter(info_10.longitude,info_10.latitude, color = 'red',label = '<10')
ax1.scatter(info_20.longitude,info_20.latitude, color = 'orange',label = '10-19')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'gold',label = '20-29')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'green',label = '30+')
plt.xlim(130,170)
plt.ylim(-10,30)
plt.show()



#US
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.scatter(info_10.longitude,info_10.latitude, color = 'red',label = '<10')
ax1.scatter(info_20.longitude,info_20.latitude, color = 'orange',label = '10-19')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'gold',label = '20-29')
ax1.scatter(info_30.longitude,info_30.latitude, color = 'green',label = '30+')
plt.xlim(-180,-50)
plt.ylim(0,75)
plt.show()



ISD_10 = info[0][info[0].cleaned_years>+10]


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()

ax1.scatter(ISD_10.longitude,ISD_10.latitude,s=1.5, color = 'red')
plt.title('ISD stations with more than 10 years data')
plt.show()


###grid

countries = ['ISD','Belgium','Finland','Germany','Ireland','Japan','Norway','Portugal','UK','US']
G_loc = np.array([47,3,55,15]) #DOWNLOADED
US_main_loc = np.array([24, -125, 56, -66]) #downloading
UK_loc = np.array([49.2, -8.0, 60.8, 1.8]) #DOWNLOADED
J_loc = np.array([24, 122.9, 45.6, 145.8]) #DOWNLOADED
P_loc = np.array([36.9,-9.5,42.1,-5])
N_loc = np.array([55,4.7,71.8,31.2])
H_loc = np.array([18.8,-160,22.3,-154.8])
PR_loc = np.array([17.6,-67.3,18.5,-64.7])
Al_loc = np.array([54.9,-171.9,71.4,-131.4])
I_loc = np.array([35,6,49,20])

#EUROPE
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


ax1.plot([G_loc[1], G_loc[1]],[G_loc[0], G_loc[2]],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([G_loc[3], G_loc[3]],[G_loc[2], G_loc[0]],  'r', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([G_loc[1], G_loc[3]],[G_loc[0], G_loc[0]],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([G_loc[3], G_loc[1]],[G_loc[2], G_loc[2]],  'r', linewidth=2, transform=ccrs.PlateCarree(),label = 'Germany')


ax1.plot([UK_loc[1], UK_loc[1]],[UK_loc[0], UK_loc[2]],  'b', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([UK_loc[3], UK_loc[3]],[UK_loc[2], UK_loc[0]],  'b', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([UK_loc[1], UK_loc[3]],[UK_loc[0], UK_loc[0]],  'b', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([UK_loc[3], UK_loc[1]],[UK_loc[2], UK_loc[2]],  'b', linewidth=2, transform=ccrs.PlateCarree(),label = 'UK')


ax1.plot([P_loc[1], P_loc[1]],[P_loc[0], P_loc[2]],  'g', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([P_loc[3], P_loc[3]],[P_loc[2], P_loc[0]],  'g', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([P_loc[1], P_loc[3]],[P_loc[0], P_loc[0]],  'g', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([P_loc[3], P_loc[1]],[P_loc[2], P_loc[2]],  'g', linewidth=2, transform=ccrs.PlateCarree(),label = 'Portugal')


ax1.plot([N_loc[1], N_loc[1]],[N_loc[0], N_loc[2]],  'c', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([N_loc[3], N_loc[3]],[N_loc[2], N_loc[0]],  'c', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([N_loc[1], N_loc[3]],[N_loc[0], N_loc[0]],  'c', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([N_loc[3], N_loc[1]],[N_loc[2], N_loc[2]],  'c', linewidth=2, transform=ccrs.PlateCarree(),label = 'Norway and Finland')

ax1.plot([I_loc[1], I_loc[1]],[I_loc[0], I_loc[2]],  'tab:orange', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([I_loc[3], I_loc[3]],[I_loc[2], I_loc[0]],  'tab:orange', linewidth=2, transform=ccrs.PlateCarree())


ax1.plot([I_loc[1], I_loc[3]],[I_loc[0], I_loc[0]],  'tab:orange', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([I_loc[3], I_loc[1]],[I_loc[2], I_loc[2]],  'tab:orange', linewidth=2, transform=ccrs.PlateCarree(),label = 'Italy')

n=0
while n<len(countries):
    plt.scatter(info[n]['longitude'],info[n]['latitude'],label = countries[n],transform=ccrs.PlateCarree())
    n=n+1
plt.legend()
plt.xlim(-10,50)
plt.ylim(20,80)

plt.show()


#ASIA
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


ax1.plot([J_loc[1], J_loc[1]],[J_loc[0], J_loc[2]],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([J_loc[3], J_loc[3]],[J_loc[2], J_loc[0]],  'r', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([J_loc[1], J_loc[3]],[J_loc[0], J_loc[0]],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([J_loc[3], J_loc[1]],[J_loc[2], J_loc[2]],  'r', linewidth=2, transform=ccrs.PlateCarree(),label = 'Japan')

n=0
while n<len(countries):
    plt.scatter(info[n]['longitude'],info[n]['latitude'],label = countries[n],transform=ccrs.PlateCarree())
    n=n+1
    
plt.legend()
plt.xlim(90,180)
plt.ylim(0,60)
plt.show()




#US
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

ax1.plot([US_main_loc[1], US_main_loc[1]],[US_main_loc[0], US_main_loc[2]],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([US_main_loc[3], US_main_loc[3]],[US_main_loc[2], US_main_loc[0]],  'r', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([US_main_loc[1], US_main_loc[3]],[US_main_loc[0], US_main_loc[0]],  'r', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([US_main_loc[3], US_main_loc[1]],[US_main_loc[2], US_main_loc[2]],  'r', linewidth=2, transform=ccrs.PlateCarree(),label = 'mainland')


ax1.plot([H_loc[1], H_loc[1]],[H_loc[0], H_loc[2]],  'b', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([H_loc[3], H_loc[3]],[H_loc[2], H_loc[0]],  'b', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([H_loc[1], H_loc[3]],[H_loc[0], H_loc[0]],  'b', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([H_loc[3], H_loc[1]],[H_loc[2], H_loc[2]],  'b', linewidth=2, transform=ccrs.PlateCarree(),label = 'Hawaii')


ax1.plot([PR_loc[1], PR_loc[1]],[PR_loc[0], PR_loc[2]],  'g', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([PR_loc[3], PR_loc[3]],[PR_loc[2], PR_loc[0]],  'g', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([PR_loc[1], PR_loc[3]],[PR_loc[0], PR_loc[0]],  'g', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([PR_loc[3], PR_loc[1]],[PR_loc[2], PR_loc[2]],  'g', linewidth=2, transform=ccrs.PlateCarree(),label = 'Puerto Rico')

ax1.plot([Al_loc[1], Al_loc[1]],[Al_loc[0], Al_loc[2]],  'c', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([Al_loc[3], Al_loc[3]],[Al_loc[2], Al_loc[0]],  'c', linewidth=2, transform=ccrs.PlateCarree())

ax1.plot([Al_loc[1], Al_loc[3]],[Al_loc[0], Al_loc[0]],  'c', linewidth=2, transform=ccrs.PlateCarree())
ax1.plot([Al_loc[3], Al_loc[1]],[Al_loc[2], Al_loc[2]],  'c', linewidth=2, transform=ccrs.PlateCarree(),label = 'Alaska')



for n in [0,9]:
    plt.scatter(info[n]['longitude'],info[n]['latitude'],s=2,label = countries[n],transform=ccrs.PlateCarree())
    n=n+1
    
plt.legend()
plt.xlim(-180,-50)
plt.ylim(0,75)
plt.show()
