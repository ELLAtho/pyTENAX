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
from matplotlib.patches import Patch

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
    
    RL_typical, _, _ = S.model_inversion(F_phat_typical, g_phat_typical, uses[i].n, Ts,b_exp = i%2)
    RL_true.append(RL_typical)



#for now only the first one because other one not ready yet but should be a loop

# diff_free = synth_RL[0].free_10 - RL_true[0][0]
# diff_0 = synth_RL[0].b0_10 - RL_true[0][0]
# diff_set = synth_RL[0].set_10 - RL_true[0][0]

# FRMSE_10 = pd.DataFrame({
#     "free" : np.sqrt(np.sum(diff_free**2))/RL_true[0][0],
#     "b0" : np.sqrt(np.sum(diff_0**2))/RL_true[0][0],
#     "set" : np.sqrt(np.sum(diff_set**2))/RL_true[0][0]
#     })


#violins of free
plt.violinplot([synth_RL[0].free_10,synth_RL[0].free_20,synth_RL[0].free_50,synth_RL[0].free_100],widths = [10*0.08,20*0.1,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True)
#violins of set
plt.violinplot([synth_RL[0].set_10,synth_RL[0].set_20,synth_RL[0].set_50,synth_RL[0].set_100],widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True)
#violins of 0
plt.violinplot([synth_RL[0].b0_10,synth_RL[0].b0_20,synth_RL[0].b0_50,synth_RL[0].b0_100],widths = [10*0.1/0.8,20/0.8*0.1,50/0.8*0.1,100/0.8*0.1],positions = [10/0.8,20/0.8,50/0.8,100/0.8],showmeans = True)

plt.plot([10*0.8,10,10/0.8],[RL_true[0][0]]*3,color = "r")
plt.plot([20*0.8,20,20/0.8],[RL_true[0][1]]*3,color = "r")
plt.plot([50*0.8,50,50/0.8],[RL_true[0][2]]*3,color = "r")
plt.plot([100*0.8,100,100/0.8],[RL_true[0][3]]*3,color = "r",label = "actual return levels")

legend_elements = [
    Patch(facecolor='C0', edgecolor='black', label='Free'),  # default matplotlib colors
    Patch(facecolor='C1', edgecolor='black', label='Set'),
    Patch(facecolor='C2', edgecolor='black', label='b = 0'),
    plt.Line2D([0], [0], color='r', label='actual return levels')
]

plt.legend(handles=legend_elements)

plt.xlabel("Return period (yr)")
plt.ylabel("precipitation (mm/hr)")
plt.xscale("log")
plt.title("linear b. 30 years")
plt.ylim(20,120)
plt.show()




#violins of free
plt.violinplot([synth_RL[0].free_10_exp,synth_RL[0].free_20_exp,synth_RL[0].free_50_exp,synth_RL[0].free_100_exp],widths = [10*0.08,20*0.1,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True)
#violins of set
plt.violinplot([synth_RL[0].set_10_exp,synth_RL[0].set_20_exp,synth_RL[0].set_50_exp,synth_RL[0].set_100_exp],widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True)
#violins of 0
plt.violinplot([synth_RL[0].b0_10_exp,synth_RL[0].b0_20_exp,synth_RL[0].b0_50_exp,synth_RL[0].b0_100_exp],widths = [10*0.1/0.8,20/0.8*0.1,50/0.8*0.1,100/0.8*0.1],positions = [10/0.8,20/0.8,50/0.8,100/0.8],showmeans = True)
plt.plot([10*0.8,10,10/0.8],[RL_true[1][0]]*3,color = "r")
plt.plot([20*0.8,20,20/0.8],[RL_true[1][1]]*3,color = "r")
plt.plot([50*0.8,50,50/0.8],[RL_true[1][2]]*3,color = "r")
plt.plot([100*0.8,100,100/0.8],[RL_true[1][3]]*3,color = "r",label = "actual return levels")

legend_elements = [
    Patch(facecolor='C0', edgecolor='black', label='Free'),  # default matplotlib colors
    Patch(facecolor='C1', edgecolor='black', label='Set'),
    Patch(facecolor='C2', edgecolor='black', label='b = 0'),
    plt.Line2D([0], [0], color='r', label='actual return levels')
]

plt.legend(handles=legend_elements)


plt.xlabel("Return period (yr)")
plt.ylabel("precipitation (mm/hr)")
plt.xscale("log")
plt.title("exponential b. 30 years")
plt.ylim(20,120)
plt.show()






#the next one now
#violins of free
plt.violinplot([synth_RL[1].free_10,synth_RL[1].free_20,synth_RL[1].free_50,synth_RL[1].free_100],widths = [10*0.08,20*0.1,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True)
#violins of set
plt.violinplot([synth_RL[1].set_10,synth_RL[1].set_20,synth_RL[1].set_50,synth_RL[1].set_100],widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True)
#violins of 0
plt.violinplot([synth_RL[1].b0_10,synth_RL[1].b0_20,synth_RL[1].b0_50,synth_RL[1].b0_100],widths = [10*0.1/0.8,20/0.8*0.1,50/0.8*0.1,100/0.8*0.1],positions = [10/0.8,20/0.8,50/0.8,100/0.8],showmeans = True)

plt.plot([10*0.8,10,10/0.8],[RL_true[0][0]]*3,color = "r")
plt.plot([20*0.8,20,20/0.8],[RL_true[0][1]]*3,color = "r")
plt.plot([50*0.8,50,50/0.8],[RL_true[0][2]]*3,color = "r")
plt.plot([100*0.8,100,100/0.8],[RL_true[0][3]]*3,color = "r",label = "actual return levels")

legend_elements = [
    Patch(facecolor='C0', edgecolor='black', label='Free'),  # default matplotlib colors
    Patch(facecolor='C1', edgecolor='black', label='Set'),
    Patch(facecolor='C2', edgecolor='black', label='b = 0'),
    plt.Line2D([0], [0], color='r', label='actual return levels')
]

plt.legend(handles=legend_elements)

plt.xlabel("Return period (yr)")
plt.ylabel("precipitation (mm/hr)")
plt.xscale("log")
plt.title("linear b. 10 years")
plt.ylim(20,120)
plt.show()




#violins of free
plt.violinplot([synth_RL[1].free_10_exp,synth_RL[1].free_20_exp,synth_RL[1].free_50_exp,synth_RL[1].free_100_exp],widths = [10*0.08,20*0.1,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True)
#violins of set
plt.violinplot([synth_RL[1].set_10_exp,synth_RL[1].set_20_exp,synth_RL[1].set_50_exp,synth_RL[1].set_100_exp],widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True)
#violins of 0
plt.violinplot([synth_RL[1].b0_10_exp,synth_RL[1].b0_20_exp,synth_RL[1].b0_50_exp,synth_RL[1].b0_100_exp],widths = [10*0.1/0.8,20/0.8*0.1,50/0.8*0.1,100/0.8*0.1],positions = [10/0.8,20/0.8,50/0.8,100/0.8],showmeans = True)
plt.plot([10*0.8,10,10/0.8],[RL_true[1][0]]*3,color = "r")
plt.plot([20*0.8,20,20/0.8],[RL_true[1][1]]*3,color = "r")
plt.plot([50*0.8,50,50/0.8],[RL_true[1][2]]*3,color = "r")
plt.plot([100*0.8,100,100/0.8],[RL_true[1][3]]*3,color = "r",label = "actual return levels")

legend_elements = [
    Patch(facecolor='C0', edgecolor='black', label='Free'),  # default matplotlib colors
    Patch(facecolor='C1', edgecolor='black', label='Set'),
    Patch(facecolor='C2', edgecolor='black', label='b = 0'),
    plt.Line2D([0], [0], color='r', label='actual return levels')
]

plt.legend(handles=legend_elements)


plt.xlabel("Return period (yr)")
plt.ylabel("precipitation (mm/hr)")
plt.xscale("log")
plt.title("exponential b. 10 years")
plt.ylim(20,120)
plt.show()










