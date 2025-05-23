# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:23:16 2025

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

from scipy.stats import norm

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob


country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9


min_years = 20

info = pd.read_csv(f"D:/metadata/{country}_fulldata.csv", dtype={'station': str})


# shouldn't need this anymore as changed files
# if name_len!=0:
#     info.station = info['station'].apply(lambda x: f'{int(x):0{name_len}}') #need to edit this according to file
# else:
#     pass

info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

#select stations


val_info = info[info['cleaned_years']>=min_years].reset_index()


i=0
saved_oes = glob.glob(f"D:/ordinary_events/{country_save}/*")
stat = val_info.iloc[i].station
oe_savename = f"D:/ordinary_events/{country_save}\\T_{stat}.csv"
if oe_savename in saved_oes:
    T = np.genfromtxt(f"D:/ordinary_events/{country_save}/T_{stat}.csv")
    P = np.genfromtxt(f"D:/ordinary_events/{country_save}/P_{stat}.csv")
    times = pd.read_csv(f"D:/ordinary_events/{country_save}/time_{stat}.csv",parse_dates = ["oe_time"])
    













