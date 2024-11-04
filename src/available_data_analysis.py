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

countries = ['ISD','Belgium','Finland','Germany','Ireland','Japan','Norway','Portugal','UK','US']

# latslons

latslons = []
dates = []
info = []


for c in countries:
    dd = np.load('D:/'+c+'_latslons.npy')
    latslons.append(dd)
    dd = np.load('D:/'+c+'_dates.npy')
    dates.append(dd)
    dd = np.load('D:/'+c+'_data.npy')
    info.append(dd)
    
    

latslons_full = np.concatenate(latslons,axis = 1)
dates_full = np.concatenate(dates,axis = 1)
info_full = np.concatenate(info,axis = 1)

bounds = np.zeros(np.shape(dates_full)[1])
n=0
while n<len(bounds):
    if info_full[3][n]<=5:
        bounds[n] = 5
    elif info_full[3][n]<=10:
        bounds[n] = 10
    elif info_full[3][n]<=15:
        bounds[n] = 15
    elif info_full[3][n]<=20:
        bounds[n] = 20
    elif info_full[3][n]<=25:
        bounds[n] = 25
    elif info_full[3][n]<=30:
        bounds[n] = 30
    elif info_full[3][n]<=35:
        bounds[n] = 35
    else:
        bounds[n] = 40
    n=n+1

n=0
while n<len(countries):
    plt.scatter(latslons[n][1],latslons[n][0],s=1.5,label = countries[n])
    n=n+1
plt.legend()
plt.show()



sc = plt.scatter(latslons_full[1],latslons_full[0],s=1.5,c=bounds)
plt.colorbar(sc)
plt.show()


sc = plt.scatter(latslons_full[1],latslons_full[0],s=1.5,c=info_full[3])
plt.colorbar(sc,extend = 'max')
plt.show()