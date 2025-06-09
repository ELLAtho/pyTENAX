# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 12:33:01 2025

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
from scipy.stats import chi2
from scipy import odr

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
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib.patches import Patch

drive = "D"


# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# station_chose = "18256"
# station_chose = "12261"
station_chose = "19376"

# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


country = 'UK' 
ERA_country = 'UK'
country_save = 'UK'
code_str = 'UK_'
minlat,minlon,maxlat,maxlon = 49, -9.0, 62, 3
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9

name_col = 'ppt' 
temp_name_col = "t2m"

all_files = glob.glob(f"{drive}:/{country}/*")
starttime = [0]*len(all_files)


for i in range(len(all_files)):
    starttime[i] = time.time()
    file = all_files[i]
    G,data_meta = read_GSDR_file(file,name_col)
    
    save_name = f"D:/georgia_data/{file[3:-3]}csv"
    
    
    G.to_csv(save_name)
    
    if i%50 == 0:
        time_taken = (time.time()-starttime[i-9])/10
        time_left = (len(all_files)-i)*time_taken/60
        print(f"{i}/{len(all_files)}. Approx time left: {time_left:.0f} mins")





















