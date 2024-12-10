# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 07:34:54 2024

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

drive='D' #name of drive

maxlat = 30
maxlon = -79

US_main_loc = np.array([24, -125, 56, -66])
J_loc = np.array([24, 122.9, 45.6, 145.8]) 
H_loc = np.array([18.8,-160,22.3,-154.8])
PR_loc = np.array([17.6,-67.3,18.5,-64.7])
Is_loc = np.array([27, 34, 34, 36])
F_loc = np.array([25, -83, 31, -78])




countries = ['Japan','US'] #Puerto RIco
n_stations = 5

info = []

for c in countries:
    dd = pd.read_csv(drive+':/metadata/'+c+'_fulldata.csv')
    info.append(dd)
    
info_full = pd.concat(info,axis=0)    



info_tropics_J = info[0][info[0].latitude<maxlat]
info_tropics_J = info_tropics_J[info_tropics_J.cleaned_years>=20]

info_tropics_US = info[1][info[1].latitude<maxlat]
info_tropics_US = info_tropics_US[info_tropics_US.longitude<maxlon]
info_tropics_US = info_tropics_US[info_tropics_US.cleaned_years>=20]

info_hawaii = info_tropics_US[info_tropics_US.latitude>H_loc[0]]
info_hawaii = info_hawaii[info_hawaii.latitude<H_loc[2]]
info_hawaii = info_hawaii[info_hawaii.longitude>H_loc[1]]
info_hawaii = info_hawaii[info_hawaii.longitude<H_loc[3]]


info_florida = info_tropics_US[info_tropics_US.latitude>F_loc[0]]
info_florida = info_florida[info_florida.latitude<F_loc[2]]
info_florida = info_florida[info_florida.longitude>F_loc[1]]
info_florida = info_florida[info_florida.longitude<F_loc[3]]


J_sort = info_tropics_J.sort_values(by=['cleaned_years'],ascending=0) #sort by size so can choose top sizes

US_sort = info_tropics_US.sort_values(by=['cleaned_years'],ascending=0) #sort by size so can choose top sizes



J_selected = J_sort[0:n_stations]
US_selected = US_sort[0:n_stations]



















