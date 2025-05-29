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
RL_true_exp = []
for i in range(len(uses)):
    F_phat_typical = [uses[i].kappa[0],uses[i].b[0],uses[i]["lambda"][0],uses[i].a[0]]
    g_phat_typical = [uses[i].mu[0],uses[i].sigma[0]]
    
    Ts = np.arange(g_phat_typical[0]-2*g_phat_typical[1] - S.temp_delta, g_phat_typical[0]+2*g_phat_typical[1] + S.temp_delta, S.temp_res_monte_carlo)
    
    RL_typical_exp, _, _ = S.model_inversion(F_phat_typical, g_phat_typical, uses[i].n, Ts,b_exp = True)
    RL_typical, _, _ = S.model_inversion(F_phat_typical, g_phat_typical, uses[i].n, Ts)
    RL_true.append(RL_typical)
    RL_true_exp.append(RL_typical_exp)



#for now only the first one because other one not ready yet but should be a loop

FRMSE_free = [0]*len(synth_RL)
FRMSE_0 = [0]*len(synth_RL)
FRMSE_set = [0]*len(synth_RL)

FRMSE_free_exp = [0]*len(synth_RL)
FRMSE_0_exp = [0]*len(synth_RL)
FRMSE_set_exp = [0]*len(synth_RL)

for i in range(len(synth_RL)):
    
    temp_free = [0]*4
    temp_0  = [0]*4
    temp_set = [0]*4
    for j in range(4):
        diff_free = synth_RL[i][synth_RL[0].columns[0+3*j]] - RL_true[i][j] #column of df with 10, 20, 50, 100 RL
        diff_0 = synth_RL[i][synth_RL[0].columns[1+3*j]] - RL_true[i][j]
        diff_set = synth_RL[i][synth_RL[0].columns[2+3*j]] - RL_true[i][j]
        
        temp_free[j] = np.sqrt(np.sum(diff_free**2))/RL_true[i][j],
        temp_0[j] = np.sqrt(np.sum(diff_0**2))/RL_true[i][j],
        temp_set[j] = np.sqrt(np.sum(diff_set**2))/RL_true[i][j]
    
    FRMSE_free[i] = temp_free
    FRMSE_0[i] = temp_0
    FRMSE_set[i] = temp_set
    
    temp_free = [0]*4
    temp_0  = [0]*4
    temp_set = [0]*4
    for j in range(4):
        diff_free = synth_RL[i][synth_RL[0].columns[12+3*j]] - RL_true_exp[i][j] #column of df with 10, 20, 50, 100 RL
        diff_0 = synth_RL[i][synth_RL[0].columns[13+3*j]] - RL_true_exp[i][j] #shifted to the exponential ones
        diff_set = synth_RL[i][synth_RL[0].columns[14+3*j]] - RL_true_exp[i][j]
        
        temp_free[j] = np.sqrt(np.sum(diff_free**2))/RL_true_exp[i][j],
        temp_0[j] = np.sqrt(np.sum(diff_0**2))/RL_true_exp[i][j],
        temp_set[j] = np.sqrt(np.sum(diff_set**2))/RL_true_exp[i][j]
    
    FRMSE_free_exp[i] = temp_free
    FRMSE_0_exp[i] = temp_0
    FRMSE_set_exp[i] = temp_set



def stretch_violin_lines(violinplot, scale=1.5):
    for key in ['cmeans', 'cmedians']:
        if key in violinplot:
            lines = violinplot[key]
            segments = lines.get_segments()
            new_segments = []
            for seg in segments:
                x0, y0 = seg[0]
                x1, y1 = seg[1]
                center = (x0 + x1) / 2
                half_length = (x1 - x0) / 2 * scale
                new_seg = [[center - half_length, y0], [center + half_length, y1]]
                new_segments.append(new_seg)
            lines.set_segments(new_segments)


for i in range(len(synth_RL)):
    
    #plot the actual return levels
    plt.plot([10*0.75,10,10/0.75],[RL_true[i][0]]*3,color = "r")
    plt.plot([20*0.75,20,20/0.75],[RL_true[i][1]]*3,color = "r")
    plt.plot([50*0.75,50,50/0.75],[RL_true[i][2]]*3,color = "r")
    plt.plot([100*0.75,100,100/0.75],[RL_true[i][3]]*3,color = "r",label = "actual return levels")
    
    
    #FRMSE
    
    
    
    #violins of free
    vln = [0]*3
    vln[0] = plt.violinplot([synth_RL[i].free_10.dropna(),synth_RL[i].free_20.dropna(),synth_RL[i].free_50.dropna(),synth_RL[i].free_100.dropna()]
                            ,widths = [10*0.08,20*0.08,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True,showmedians = True)
    #violins of set
    vln[1] = plt.violinplot([synth_RL[i].set_10.dropna(),synth_RL[i].set_20.dropna(),synth_RL[i].set_50.dropna(),synth_RL[i].set_100.dropna()],
                            widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True,showmedians = True)
    #violins of 0
    vln[2] = plt.violinplot([synth_RL[i].b0_10.dropna(),synth_RL[i].b0_20.dropna(),synth_RL[i].b0_50.dropna(),synth_RL[i].b0_100.dropna()],
                            widths = [10*0.1/0.8,20/0.8*0.1,50/0.8*0.1,100/0.8*0.1],positions = [10/0.8,20/0.8,50/0.8,100/0.8],showmeans = True,showmedians = True)
    
    for violin in vln:
        violin["cmeans"].set_color('k')
        violin["cmedians"].set_color('k')
        violin["cmeans"].set_zorder(10)
        violin["cmeans"].set_linestyle(":")
        stretch_violin_lines(violin, scale=3.0)
    
    
    legend_elements = [
        Patch(facecolor='C0', label='Free'),  # default matplotlib colors
        Patch(facecolor='C1', label='Set'),
        Patch(facecolor='C2', label='b = 0'),
        plt.Line2D([0], [0], color='r', label='actual return levels'),
        plt.Line2D([0], [0], color='k', label='median'),
        plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
    ]
    
    plt.legend(handles=legend_elements)
    
    plt.xlabel("Return period (yr)")
    plt.ylabel("precipitation (mm/hr)")
    plt.xscale("log")
    plt.title(f"linear b. {uses[i].n_years[0]} years \
              \n g_phat = [{uses[i].mu[0]},{uses[i].sigma[0]}] \
              \n F_phat = [{uses[i].kappa[0]},{uses[i].b[0]},{uses[i]["lambda"][0]},{uses[i].a[0]}]")
    plt.ylim(20,120)
    plt.show()
    

    #exponential
    
    #plot the actual return levels
    plt.plot([10*0.75,10,10/0.75],[RL_true_exp[i][0]]*3,color = "r") #TODO: uncertainty on the RL from MC
    plt.plot([20*0.75,20,20/0.75],[RL_true_exp[i][1]]*3,color = "r")
    plt.plot([50*0.75,50,50/0.75],[RL_true_exp[i][2]]*3,color = "r")
    plt.plot([100*0.75,100,100/0.75],[RL_true_exp[i][3]]*3,color = "r",label = "actual return levels")
    
    
    #violins of free
    vln[0] = plt.violinplot([synth_RL[i].free_10_exp.dropna(),synth_RL[i].free_20_exp.dropna(),synth_RL[i].free_50_exp.dropna(),synth_RL[i].free_100_exp.dropna()],widths = [10*0.08,20*0.1,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True,showmedians = True)
    #violins of set
    vln[1] = plt.violinplot([synth_RL[i].set_10_exp.dropna(),synth_RL[i].set_20_exp.dropna(),synth_RL[i].set_50_exp.dropna(),synth_RL[i].set_100_exp.dropna()],widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True,showmedians = True)
    #violins of 0
    vln[2] = plt.violinplot([synth_RL[i].b0_10_exp,synth_RL[i].b0_20_exp,synth_RL[i].b0_50_exp,synth_RL[i].b0_100_exp],widths = [10*0.1/0.8,20/0.8*0.1,50/0.8*0.1,100/0.8*0.1],positions = [10/0.8,20/0.8,50/0.8,100/0.8],showmeans = True,showmedians = True)
    
    
    for violin in vln:
        violin["cmeans"].set_color('k')
        violin["cmedians"].set_color('k')
        violin["cmeans"].set_zorder(10)
        violin["cmeans"].set_linestyle(":")
        stretch_violin_lines(violin, scale=3.0)
    
    
    legend_elements = [
        Patch(facecolor='C0', label='Free'),  # default matplotlib colors
        Patch(facecolor='C1', label='Set'),
        Patch(facecolor='C2', label='b = 0'),
        plt.Line2D([0], [0], color='r', label='actual return levels'),
        plt.Line2D([0], [0], color='k', label='median'),
        plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
    ]
    
    plt.legend(handles=legend_elements)
    
    
    plt.xlabel("Return period (yr)")
    plt.ylabel("precipitation (mm/hr)")
    plt.xscale("log")
    plt.title(f"exponential b. {uses[i].n_years[0]} years \
              \n g_phat = [{uses[i].mu[0]},{uses[i].sigma[0]}] \
              \n F_phat = [{uses[i].kappa[0]},{uses[i].b[0]},{uses[i]["lambda"][0]},{uses[i].a[0]}]")
    plt.ylim(20,120)
    plt.show()




