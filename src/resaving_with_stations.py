# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:21:56 2025

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
import matplotlib.patches as mpatches


drive='D' #name of drive
countries = ['ISD','Belgium','Finland','Germany','Ireland','Japan','Norway','Portugal','UK','US']


info = []

for c in countries:
    dd = pd.read_csv(drive+':/metadata/'+c+'_fulldata.csv', dtype={'station': str})
    station_names = [0]*len(dd)
    files = glob.glob('D:/'+c+'/*') #list of filenames in folder
    str_start = 7+len(c) # for getting filename number in loopstr_end = -4
    
    
    for i in range(len(dd)):
        station_names[i] = files[i][str_start:-4]
    
    dd["station"]=station_names
    dd.to_csv(drive+':/metadata/'+c+'_fulldata.csv', index = False)
    info.append(dd)
   
    
    
    
    
    