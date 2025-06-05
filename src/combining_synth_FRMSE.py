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
        
        temp_free[j] = np.sqrt(np.sum(diff_free**2)/len(diff_free))/RL_true[i][j]
        temp_0[j] = np.sqrt(np.sum(diff_0**2)/len(diff_free))/RL_true[i][j]
        temp_set[j] = np.sqrt(np.sum(diff_set**2)/len(diff_free))/RL_true[i][j]
    
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
        
        temp_free[j] = np.sqrt(np.sum(diff_free**2)/len(diff_free))/RL_true_exp[i][j]
        temp_0[j] = np.sqrt(np.sum(diff_0**2)/len(diff_free))/RL_true_exp[i][j]
        temp_set[j] = np.sqrt(np.sum(diff_set**2)/len(diff_free))/RL_true_exp[i][j]
    
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
    
    x_mult = [0.75,0.93,1.15]
    FRMSEs = [FRMSE_free,FRMSE_set,FRMSE_0]
    
    for k in range(3):
        plt.text(10*x_mult[k], np.min(synth_RL[i].free_10)-5, f"{FRMSEs[k][i][0]:.2}") #last 0 in string is to get float from array
        plt.text(20*x_mult[k], np.min(synth_RL[i].free_20)-5, f"{FRMSEs[k][i][1]:.2}") 
        plt.text(50*x_mult[k], np.min(synth_RL[i].free_50)-5, f"{FRMSEs[k][i][2]:.2}") 
        plt.text(100*x_mult[k], np.min(synth_RL[i].free_100)-5, f"{FRMSEs[k][i][3]:.2}") 
    
    
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
    plt.ylim(0,120)
    plt.show()
    

    #exponential
    
    #plot the actual return levels
    plt.plot([10*0.75,10,10/0.75],[RL_true_exp[i][0]]*3,color = "r") #TODO: uncertainty on the RL from MC
    plt.plot([20*0.75,20,20/0.75],[RL_true_exp[i][1]]*3,color = "r")
    plt.plot([50*0.75,50,50/0.75],[RL_true_exp[i][2]]*3,color = "r")
    plt.plot([100*0.75,100,100/0.75],[RL_true_exp[i][3]]*3,color = "r",label = "actual return levels")
    
    
    #FRMSE
    FRMSEs = [FRMSE_free_exp,FRMSE_set_exp,FRMSE_0_exp]
    
    for k in range(3):
        plt.text(10*x_mult[k], np.min(synth_RL[i].free_10)-5, f"{FRMSEs[k][i][0]:.2}") #last 0 in string is to get float from array
        plt.text(20*x_mult[k], np.min(synth_RL[i].free_20)-5, f"{FRMSEs[k][i][1]:.2}") 
        plt.text(50*x_mult[k], np.min(synth_RL[i].free_50)-5, f"{FRMSEs[k][i][2]:.2}") 
        plt.text(100*x_mult[k], np.min(synth_RL[i].free_100)-5, f"{FRMSEs[k][i][3]:.2}") 
    
    
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
    plt.ylim(0,120)
    plt.show()


#plot fractional violins (i think)
for i in range(len(synth_RL)):
    
    #plot the actual return levels
    plt.plot([10*0.75,10,10/0.75],[1]*3,color = "r")
    plt.plot([20*0.75,20,20/0.75],[1]*3,color = "r")
    plt.plot([50*0.75,50,50/0.75],[1]*3,color = "r")
    plt.plot([100*0.75,100,100/0.75],[1]*3,color = "r",label = "actual return levels")
    
    
    
    #violins of free
    vln = [0]*3
    vln[0] = plt.violinplot([synth_RL[i].free_10.dropna()/RL_true[i][0],synth_RL[i].free_20.dropna()/RL_true[i][1],synth_RL[i].free_50.dropna()/RL_true[i][2],synth_RL[i].free_100.dropna()/RL_true[i][3]]
                            ,widths = [10*0.08,20*0.08,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True,showmedians = True)
    #violins of set
    vln[1] = plt.violinplot([synth_RL[i].set_10.dropna()/RL_true[i][0],synth_RL[i].set_20.dropna()/RL_true[i][1],synth_RL[i].set_50.dropna()/RL_true[i][2],synth_RL[i].set_100.dropna()/RL_true[i][3]],
                            widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True,showmedians = True)
    #violins of 0
    vln[2] = plt.violinplot([synth_RL[i].b0_10.dropna()/RL_true[i][0],synth_RL[i].b0_20.dropna()/RL_true[i][1],synth_RL[i].b0_50.dropna()/RL_true[i][2],synth_RL[i].b0_100.dropna()/RL_true[i][3]],
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
    plt.ylabel("gen_RL/RL")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"linear b. {uses[i].n_years[0]} years \
              \n g_phat = [{uses[i].mu[0]},{uses[i].sigma[0]}] \
              \n F_phat = [{uses[i].kappa[0]},{uses[i].b[0]},{uses[i]["lambda"][0]},{uses[i].a[0]}]")
    plt.ylim(0.6,3)
    plt.show()
    

    #exponential
    
    #plot the actual return levels
    plt.plot([10*0.75,10,10/0.75],[1]*3,color = "r")
    plt.plot([20*0.75,20,20/0.75],[1]*3,color = "r")
    plt.plot([50*0.75,50,50/0.75],[1]*3,color = "r")
    plt.plot([100*0.75,100,100/0.75],[1]*3,color = "r",label = "actual return levels")
    
    
    
    #violins of free
    vln[0] = plt.violinplot([synth_RL[i].free_10_exp.dropna()/RL_true_exp[i][0],synth_RL[i].free_20_exp.dropna()/RL_true_exp[i][1],synth_RL[i].free_50_exp.dropna()/RL_true_exp[i][2],synth_RL[i].free_100_exp.dropna()/RL_true_exp[i][3]],widths = [10*0.08,20*0.1,50*0.08,100*0.08], positions = [10*0.8,20*0.8,50*0.8,100*0.8],showmeans = True,showmedians = True)
    #violins of set
    vln[1] = plt.violinplot([synth_RL[i].set_10_exp.dropna()/RL_true_exp[i][0],synth_RL[i].set_20_exp.dropna()/RL_true_exp[i][1],synth_RL[i].set_50_exp.dropna()/RL_true_exp[i][2],synth_RL[i].set_100_exp.dropna()/RL_true_exp[i][3]],widths = [10*0.1,20*0.1,50*0.1,100*0.1],positions = [10,20,50,100],showmeans = True,showmedians = True)
    #violins of 0
    vln[2] = plt.violinplot([synth_RL[i].b0_10_exp.dropna()/RL_true_exp[i][0],synth_RL[i].b0_20_exp/RL_true_exp[i][1],synth_RL[i].b0_50_exp/RL_true_exp[i][2],synth_RL[i].b0_100_exp/RL_true_exp[i][3]],widths = [10*0.1/0.8,20/0.8*0.1,50/0.8*0.1,100/0.8*0.1],positions = [10/0.8,20/0.8,50/0.8,100/0.8],showmeans = True,showmedians = True)
    
    
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
    plt.ylabel("gen_RL/RL")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"exponential b. {uses[i].n_years[0]} years \
              \n g_phat = [{uses[i].mu[0]},{uses[i].sigma[0]}] \
              \n F_phat = [{uses[i].kappa[0]},{uses[i].b[0]},{uses[i]["lambda"][0]},{uses[i].a[0]}]")
    plt.ylim(0.6,3)
    plt.show()
    
    
ret_lvls = ["10","20","50","100"]
# plot the fractionals all together in a different layout
fig = plt.figure(figsize=(12,12))
for i in range(3):
    ax = fig.add_subplot(2,3,i+1)
    
    #plot boxes
    boxplot_list = [synth_RL[years][f"{bstyle}_{ret_lvls[ret_n]}"].dropna()/RL_true[years][ret_n] for ret_n in range(4) for bstyle in ["free","set","b0"] for years in [i*3,i*3+2,i*3+1]]
       
    box_plot = ax.boxplot(boxplot_list,
                          positions = np.concat([np.arange(0,4.5,0.5),np.arange(5,9.5,0.5),np.arange(10,14.5,0.5),np.arange(15,19.5,0.5)]),
                          showmeans = True, meanline = True, patch_artist=True,
                          meanprops=dict(marker=None, linestyle=':', linewidth=1,color = 'k'),
                          sym = "",
                          whis = [5,95])
    ax.grid(axis = "y")
    
    #set colors of boxes
    colors = ["r","g","b","r","g","b","r","g","b","r","g","b",]
    alpha = [1,0.6,0.3]
    for n_patch in range(len(box_plot['boxes'])):
        patch = box_plot['boxes'][n_patch]
        patch.set_facecolor(colors[int(np.trunc(n_patch/3))])
        patch.set_alpha(alpha[int(n_patch%3)])
                
    for median_line in box_plot["medians"]:
        median_line.set_color('k')
    
    ax.set_xticks(np.arange(2,18,5),ret_lvls)
    
    ax.set_yscale("log")
    ax.set_ylim(10**-0.15,10**0.15)
    custom_ticks = [1/1.1, 1, 1.1]
    custom_ticklabels = ["1/1.1", "1", "1.1"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    ax.set_title(f"linear b = {uses[i*3].b[0]}")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("gen_RL/RL")

legend_elements = [
    Patch(facecolor='r', label='Free'),  # default matplotlib colors
    Patch(facecolor='g', label='Set'),
    Patch(facecolor='b', label='b = 0, 30 years'),
    Patch(facecolor='b', alpha = 0.6, label='b = 0, 20 years'),
    Patch(facecolor='b', alpha = 0.3, label='b = 0, 10 years'),
    plt.Line2D([0], [0], color='k', label='median'),
    plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
]    

plt.legend(handles=legend_elements)


for i in range(3):
    ax = fig.add_subplot(2,3,i+4)
    
    #plot boxes
    boxplot_list = [synth_RL[years][f"{bstyle}_{ret_lvls[ret_n]}_exp"].dropna()/RL_true_exp[years][ret_n] for ret_n in range(4) for bstyle in ["free","set","b0"] for years in [i*3,i*3+2,i*3+1]]
       
    box_plot = ax.boxplot(boxplot_list,
                          positions = np.concat([np.arange(0,4.5,0.5),np.arange(5,9.5,0.5),np.arange(10,14.5,0.5),np.arange(15,19.5,0.5)]),
                          showmeans = True, meanline = True, patch_artist=True,
                          meanprops=dict(marker=None, linestyle=':', linewidth=1,color = 'k'),
                          sym = "",
                          whis = [5,95])
    plt.grid(axis = "y")
    
    #set colors of boxes
    colors = ["r","g","b","r","g","b","r","g","b","r","g","b",]
    alpha = [1,0.6,0.3]
    for n_patch in range(len(box_plot['boxes'])):
        patch = box_plot['boxes'][n_patch]
        patch.set_facecolor(colors[int(np.trunc(n_patch/3))])
        patch.set_alpha(alpha[int(n_patch%3)])
                
    for median_line in box_plot["medians"]:
        median_line.set_color('k')
    
    plt.xticks(np.arange(2,18,5),ret_lvls)
    
    plt.yscale("log")
    plt.ylim(10**-0.15,10**0.15)
    
    custom_ticks = [1/1.1, 1, 1.1]
    custom_ticklabels = ["1/1.1", "1", "1.1"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("gen_RL/RL")
    
    
    plt.title(f"exponential b = {uses[i*3].b[0]}")
    plt.xlabel("Return period (years)")

legend_elements = [
    Patch(facecolor='r', label='Free'),  # default matplotlib colors
    Patch(facecolor='g', label='Set'),
    Patch(facecolor='b', label='b = 0, 30 years'),
    Patch(facecolor='b', alpha = 0.6, label='b = 0, 20 years'),
    Patch(facecolor='b', alpha = 0.3, label='b = 0, 10 years'),
    plt.Line2D([0], [0], color='k', label='median'),
    plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
]    

plt.legend(handles=legend_elements)

plt.tight_layout()
plt.show()




# same as above but with less stuff

    
ret_lvls = ["10","100"]
# plot the fractionals all together in a different layout
fig = plt.figure(figsize=(12,12))
for i in range(3):
    ax = fig.add_subplot(2,3,i+1)
    
    #plot boxes
    boxplot_list = [synth_RL[years][f"{bstyle}_{ret_lvls[ret_n]}"].dropna()/RL_true[years][ret_n*3] for ret_n in range(2) for bstyle in ["free","set","b0"] for years in [i*3,i*3+2,i*3+1]]
       
    box_plot = ax.boxplot(boxplot_list,
                          positions = np.concat([np.arange(0,4.5,0.5),np.arange(5,9.5,0.5)]),
                          showmeans = True, meanline = True, patch_artist=True,
                          meanprops=dict(marker=None, linestyle=':', linewidth=1,color = 'k'),
                          sym = "",
                          whis = [5,95])
    ax.grid(axis = "y")
    
    #set colors of boxes
    colors = ["r","g","b","r","g","b","r","g","b","r","g","b",]
    alpha = [1,0.6,0.3]
    for n_patch in range(len(box_plot['boxes'])):
        patch = box_plot['boxes'][n_patch]
        patch.set_facecolor(colors[int(np.trunc(n_patch/3))])
        patch.set_alpha(alpha[int(n_patch%3)])
                
    for median_line in box_plot["medians"]:
        median_line.set_color('k')
    
    ax.set_xticks(np.arange(2,9,5),ret_lvls)
    
    ax.set_yscale("log")
    ax.set_ylim(10**-0.15,10**0.15)
    custom_ticks = [1/1.1, 1, 1.1]
    custom_ticklabels = ["1/1.1", "1", "1.1"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    ax.set_title(f"linear b = {uses[i*3].b[0]}")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("gen_RL/RL")

legend_elements = [
    Patch(facecolor='r', label='Free'),  # default matplotlib colors
    Patch(facecolor='g', label='Set'),
    Patch(facecolor='b', label='b = 0, 30 years'),
    Patch(facecolor='b', alpha = 0.6, label='b = 0, 20 years'),
    Patch(facecolor='b', alpha = 0.3, label='b = 0, 10 years'),
    plt.Line2D([0], [0], color='k', label='median'),
    plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
]    

plt.legend(handles=legend_elements)


for i in range(3):
    ax = fig.add_subplot(2,3,i+4)
    
    #plot boxes
    boxplot_list = [synth_RL[years][f"{bstyle}_{ret_lvls[ret_n]}_exp"].dropna()/RL_true_exp[years][ret_n*3] for ret_n in range(2) for bstyle in ["free","set","b0"] for years in [i*3,i*3+2,i*3+1]]
       
    box_plot = ax.boxplot(boxplot_list,
                          positions = np.concat([np.arange(0,4.5,0.5),np.arange(5,9.5,0.5)]),
                          showmeans = True, meanline = True, patch_artist=True,
                          meanprops=dict(marker=None, linestyle=':', linewidth=1,color = 'k'),
                          sym = "",
                          whis = [5,95])
    plt.grid(axis = "y")
    
    #set colors of boxes
    colors = ["r","g","b","r","g","b","r","g","b","r","g","b",]
    alpha = [1,0.6,0.3]
    for n_patch in range(len(box_plot['boxes'])):
        patch = box_plot['boxes'][n_patch]
        patch.set_facecolor(colors[int(np.trunc(n_patch/3))])
        patch.set_alpha(alpha[int(n_patch%3)])
                
    for median_line in box_plot["medians"]:
        median_line.set_color('k')
    
    plt.xticks(np.arange(2,9,5),ret_lvls)
    
    plt.yscale("log")
    plt.ylim(10**-0.15,10**0.15)
    
    custom_ticks = [1/1.1, 1, 1.1]
    custom_ticklabels = ["1/1.1", "1", "1.1"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("gen_RL/RL")
    
    
    plt.title(f"exponential b = {uses[i*3].b[0]}")
    plt.xlabel("Return period (years)")

legend_elements = [
    Patch(facecolor='r', label='Free'),  # default matplotlib colors
    Patch(facecolor='g', label='Set'),
    Patch(facecolor='b', label='b = 0, 30 years'),
    Patch(facecolor='b', alpha = 0.6, label='b = 0, 20 years'),
    Patch(facecolor='b', alpha = 0.3, label='b = 0, 10 years'),
    plt.Line2D([0], [0], color='k', label='median'),
    plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
]    

plt.legend(handles=legend_elements)

plt.tight_layout()
plt.show()





