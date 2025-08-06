# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:34:01 2025

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
import xarray as xr
import time

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from scipy.stats import norm
from scipy.optimize import minimize
from scipy import fft


from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
from pyTENAX.sine_model import *


import glob


A = 13 #average yearly temperature
B = 5 # range of temperature going up and down
p = 365.25/(2*np.pi) #period (aka a year)
shift = 180 # shift of starting point


var = 2 #variance of the normal distribution around
delta = 0.5 #dependence on the day
ave_loc = 0 #equivalent to shift, where the changes start in the year
daysize = 3


Ts = np.arange(-20,40,0.1)
T_mc = generate_temperature_mc(A, B, var, delta, ave_loc, Ts)

###############################################################################
# plot hist of the T_mc
eT_hist = np.arange(-20,40,0.2)
eT_edges = np.concatenate([np.array([eT_hist[0]-(eT_hist[1]-eT_hist[0])/2]),(eT_hist + (eT_hist[1]-eT_hist[0])/2)]) #convert bin centres into bin edges
hist, bin_edges = np.histogram(T_mc, bins=eT_edges, density=True)

plt.plot(hist)
plt.show()


###############################################################################


phat = sine_temperature_model(T_mc)
















