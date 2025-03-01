# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:13:59 2025

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
from matplotlib import cm
import kgcpy


drive='D' #name of drive
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
    dd = pd.read_csv(drive+':/metadata/'+c+'_fulldata.csv', dtype={'station': str})
    n_drop = len(dd.columns)-9
    dd = dd.drop(dd.columns[0:n_drop], axis=1)
    dd["kgb_zone"] = dd.apply(lambda row: kgcpy.lookupCZ(row["latitude"], row["longitude"]), axis=1)
    info.append(dd)
    









































