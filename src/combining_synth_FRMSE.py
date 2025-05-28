# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:00:55 2025

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
import seaborn

from scipy.stats import norm

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob



synth_files = glob.glob("D:/outputs/synthetic\\RL_specific*")
use_files = glob.glob("D:/outputs/synthetic\\parameters_set*")


synth_RL = [pd.read_csv(file) for file in synth_files]
uses = [pd.read_csv(file) for file in use_files]

S = TENAX(
        return_period = [10,20,50,100],  
        durations = [60, 180],
        left_censoring = [0, 0.90],
        alpha = 0,
        n_monte_carlo = 20000, # total number of events (on average)
        
    )

RL_true = []
for i in range(len(uses)):
    F_phat_typical = [uses[i].kappa[0],uses[i].b[0],uses[i]["lambda"][0],uses[i].a[0]]
    g_phat_typical = [uses[i].mu[0],uses[i].sigma[0]]
    
    Ts = np.arange(g_phat_typical[0]-2*g_phat_typical[1] - S.temp_delta, g_phat_typical[0]+2*g_phat_typical[1] + S.temp_delta, S.temp_res_monte_carlo)
    
    RL_typical, _, _ = S.model_inversion(F_phat_typical, g_phat_typical, uses[i].n, Ts)
    RL_true.append(RL_typical)





















