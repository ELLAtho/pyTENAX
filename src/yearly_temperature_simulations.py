# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 08:41:55 2025

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


#### a simulation of all the temperatures during the year

### define the easy functions
alpha = 13 #average yearly temperature
beta = 3 # range of temperature going up and down
p = 365.25/(2*np.pi) #period (aka a year)
shift = 180 # shift of starting point


var = 2 #variance of the normal distribution around
delta = 0.5 #dependence on the day
ave_loc = 90 #equivalent to shift, where the changes start in the year

x = np.arange(0,365)

# definition of the function for mu in the temperature distribution over the year
def yearly_mu(x, alpha, beta, p, shift):
    return alpha + beta*np.sin(x/p + shift/p)

def yearly_sigma(x, var, delta, p, ave_loc):
    return (1 + delta * np.sin(x/p + ave_loc/p))*var



plt.plot(x,yearly_mu(x, alpha, beta, p, shift))
plt.fill_between(x, 
                 yearly_mu(x, alpha, beta, p, shift) - yearly_sigma(x, var, delta, p, ave_loc),
                 yearly_mu(x, alpha, beta, p, shift) + yearly_sigma(x, var, delta, p, ave_loc),
                 alpha = 0.5)
plt.title(f"mu = {alpha} + {beta}sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + {ave_loc})*2pi/365.25))*{var}")

plt.show()

eT = np.arange(-12,40,0.2)
norms = [gen_norm_pdf(eT, yearly_mu(x[i], alpha, beta, p, shift),
                      yearly_sigma(x[i], var, delta, p, ave_loc), 2) for i in range(365)]




plt.plot(eT,sum(norms)/365)
plt.title(f"mu = {alpha} + {beta}sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + {ave_loc})*2pi/365.25))*{var}")
plt.show()

## make a grid with some options
betas = [2,4,6]
ave_locs = [0,90,180]


fig = plt.figure(figsize = (12,12))
for i in range(3):
    beta = betas[i]
    for j in range(3):
        ave_loc = ave_locs[i]
        
        ax = fig.add_subplot(3,3,1+i+3*j)
        ax.plot(x,yearly_mu(x, alpha, beta, p, shift))
        plt.fill_between(x, 
                         yearly_mu(x, alpha, beta, p, shift) - yearly_sigma(x, var, delta, p, ave_loc),
                         yearly_mu(x, alpha, beta, p, shift) + yearly_sigma(x, var, delta, p, ave_loc),
                         alpha = 0.5)
        plt.title(f"beta = {beta}, ave_loc = {ave_loc}")
        
        plt.ylim(6,22)

plt.suptitle(f"mu = {alpha} + beta*sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + ave_loc)*2pi/365.25))*{var}")
plt.show()



eT = np.arange(0,30,0.2)

fig = plt.figure(figsize = (12,12))
for i in range(3):
    beta = betas[i]
    for j in range(3):
        ave_loc = ave_locs[j]
        
        norms = [gen_norm_pdf(eT, yearly_mu(x[i], alpha, beta, p, shift),
                              yearly_sigma(x[i], var, delta, p, ave_loc), 2) for i in range(365)]
        
        ax = fig.add_subplot(3,3,1+i+3*j)
        ax.plot(eT,sum(norms)/365)
        plt.title(f"beta = {beta}, ave_loc = {ave_loc}")

plt.suptitle(f"mu = {alpha} + beta*sin((day + {shift})*2pi/365.25), \n sigma = (1 + {delta} * np.sin((day + ave_loc)*2pi/365.25))*{var}")

plt.show()



##### now looking at a filter for the storms












##### skewed normal to fit?




















