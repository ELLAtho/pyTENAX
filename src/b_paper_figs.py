# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:53:56 2025

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
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from scipy.stats import lmoment

import datetime as dt
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

import xarray as xr
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, FormatStrFormatter


import cartopy.crs as ccrs
import matplotlib.dates as mdates
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
import matplotlib.ticker 
import seaborn


from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from matplotlib import cm
import alphashape
from shapely.geometry import Polygon
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

drive = 'D'


countries = ["germany","Japan","UK","US"]
country_saves = ["germany","Japan","UK","US_main"]
code_strs = ["DP_","JP_","UK_","US_"]
min_startdates = [dt.datetime(1900,1,1),dt.datetime(1900,1,1),dt.datetime(1950,1,1),dt.datetime(1950,1,1)] #this is for if havent read all ERA5 data yet

lons_lats = [[47, 3, 55, 15],[24, 122.9, 45.6, 145.8],[49, -9.0, 62, 3] ,[24, -125, 56, -66]]


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
min_yrs = 10
# censor_thr = 0.9



df_parameters = [0]*4
TENAX_use = [0]*4
df_parameters_0 = [0]*4
df_parameters_exp = [0]*4
df_parameters_neg = [0]*4
new_df = [0]*4
df_generated_parameters = [0]*4
df_generated_parameters_0 = [0]*4
df_generated_parameters_exp = [0]*4


df_generated_parameters_one2one = [0]*4
df_generated_parameters_0_one2one = [0]*4
df_generated_parameters_exp_one2one = [0]*4


comb_df_gen_params = [0]*4
comb_df_gen_params_0 = [0]*4
comb_df_gen_params_exp = [0]*4


info = [0]*4

for country_i in range(4):
    country_save = country_saves[country_i]
    country = countries[country_i]
    code_str = code_strs[country_i]
    
    minlat, minlon, maxlat, maxlon = lons_lats[country_i]
    min_startdate = min_startdates[country_i]
    
    
    save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
    df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'
    df_gen_savename = drive + ':/outputs/'+country_save+'\\synth_generated_parameters.csv'
    
    df_parameters[country_i] = pd.read_csv(df_savename, dtype={'station': str}) 
    TENAX_use[country_i] = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters
    
    df_parameters_0[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/parameters.csv", dtype={'station': str})
    df_parameters_exp[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/parameters_exp.csv", dtype={'station': str})
    df_generated_parameters[country_i] = pd.read_csv(df_gen_savename)
    df_generated_parameters_0[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/synth_generated_parameters.csv")
    df_generated_parameters_exp[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/synth_generated_parameters_exp.csv")
    
    df_generated_parameters_one2one[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/synth_generated_parameters_one2one.csv")
    df_generated_parameters_0_one2one[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/synth_generated_parameters_one2one.csv")
    df_generated_parameters_exp_one2one[country_i] = pd.read_csv(f"{drive}:/outputs/{country_save}/synth_generated_parameters_one2one_exp.csv")
    
    
    # puts the redos all together into one dataframe
    comb_df_gen_params[country_i] = pd.DataFrame()
    comb_df_gen_params_0[country_i] = pd.DataFrame()
    comb_df_gen_params_exp[country_i] = pd.DataFrame()

    params = ["b", "kappa", "lambda", "a"]
    for param in params:
        comb_df_gen_params[country_i][param] = pd.concat([df_generated_parameters_one2one[country_i][f"{param}{j}"].copy().dropna() for j in range(5)])
        comb_df_gen_params_0[country_i][param] = pd.concat([df_generated_parameters_0_one2one[country_i][f"{param}{j}"].copy().dropna() for j in range(5)])
        comb_df_gen_params_exp[country_i][param] = pd.concat([df_generated_parameters_exp_one2one[country_i][f"{param}{j}"].copy().dropna() for j in range(5)])
        
    comb_df_gen_params[country_i].reset_index(inplace = True)
    comb_df_gen_params_0[country_i].reset_index(inplace = True)
    comb_df_gen_params_exp[country_i].reset_index(inplace = True)
    
    
    info1 = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
    
    info1.startdate = pd.to_datetime(info1.startdate)
    info1.enddate = pd.to_datetime(info1.enddate)
    val_info = info1[info1['cleaned_years']>=min_yrs] #filter out stations that are less than min
    val_info = val_info[val_info['startdate']>=min_startdate]
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
    
    info[country_i] = val_info.reset_index()
    
    if np.size(glob.glob(save_path_neg)) != 0:
        df_parameters_neg[country_i] = pd.read_csv(save_path_neg, dtype={'station': str})
    
        #dataframe with all values
        new_df[country_i] = df_parameters[country_i][['station','latitude','longitude','b','kappa','lambda','a','mu','sigma','thr','n_events_per_yr']].copy()
        
        mask = new_df[country_i]['b'] == 0
        
        new_df[country_i].loc[mask, 'b'] = df_parameters_neg[country_i]['b2'].to_numpy()
        new_df[country_i].loc[mask, 'kappa'] = df_parameters_neg[country_i]['kappa2'].to_numpy()
        new_df[country_i].loc[mask, 'lambda'] = df_parameters_neg[country_i]['lambda2'].to_numpy()
        new_df[country_i].loc[mask, 'a'] = df_parameters_neg[country_i]['a2'].to_numpy()
    
    else:
        new_df[country_i] = df_parameters[country_i].copy()
    
    missing_rows = pd.merge(df_parameters[country_i].station, df_parameters_0[country_i].station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    if len(missing_rows) != 0:
        print("miss-match, dropping")
        df_parameters[country_i] = df_parameters[country_i].drop(missing_rows.index)
        df_parameters[country_i] = df_parameters[country_i].reindex(index = range(len(df_parameters[country_i])))
        new_df[country_i] = new_df[country_i].drop(missing_rows.index)
        new_df[country_i] = new_df[country_i].reindex(index = range(len(new_df[country_i])))
    else:
        pass


# load hindcast data


hindcasts = [0]*4
hindcasts_exp = [0]*4

for country_i in range(4):
    country_save = country_saves[country_i]
    hindcast_savename = f"{drive}:/outputs/{country_save}/hindcasts\\F_phat.csv"
    hindcast_savename_exp = f"{drive}:/outputs/{country_save}/hindcasts\\F_phat_exp.csv"
    hindcasts[country_i] = pd.read_csv(hindcast_savename, dtype = {"station" : str})
    hindcasts_exp[country_i] = pd.read_csv(hindcast_savename_exp, dtype = {"station" : str})


# load synthetic FRMSE data


synth_files = [f"D:/outputs/synthetic\\RL_specific{num}.csv" for num in np.arange(1,10)]
#glob.glob("D:/outputs/synthetic\\RL_specific*")
use_files = [f"D:/outputs/synthetic\\parameters_set{num}.csv" for num in np.arange(1,10)]
#glob.glob("D:/outputs/synthetic\\parameters_set*")


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


# t test for average of b
for country_i in range(4):
    t_test = ttest_1samp(new_df[country_i].b,0,nan_policy = "omit")
    print(f"p value for {countries[country_i]} is {t_test[1]}")

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


# L moments and measure of spatial spread
# l_moments = [0]*4
# l_moments_synth = [0]*4
# for country_i in range(4):
#     l_moments[country_i] = lmoment(new_df[country_i].b)
#     l_moments_synth[country_i] = lmoment(df_generated_parameters[country_i].b) # these are nan
    
#     n_events = np.ceil(new_df[country_i].n_events_per_yr * info[country_i].cleaned_years) # this is wrong because the events per year are wrong
#     weights = n_events/np.sum(n_events)
#     mean_l1,std_l1 = weighted_avg_and_std(l_moments[country_i][1], weights)



# FIG 1
# maps of spatial distributions

s = 3
sig_mod = 8
norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)



# fig = plt.figure(figsize=(12, 10))
# proj = ccrs.PlateCarree()
# for country_i in range(4):
#     ax = fig.add_subplot(2, 2, country_i+1, projection=proj)
    
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')

    

#     norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
    
#     sc = ax.scatter(
#         df_parameters[country_i].longitude[df_parameters[country_i].b==0],
#         df_parameters[country_i].latitude[df_parameters[country_i].b==0],
#         s = s,
#         color = 'darkgrey',  
#     )

#     sc = ax.scatter(
#         df_parameters[country_i].longitude[df_parameters[country_i].b!=0],
#         df_parameters[country_i].latitude[df_parameters[country_i].b!=0],
#         c=df_parameters[country_i].b[df_parameters[country_i].b!=0],
#         s = s,
#         cmap='seismic',  
#         norm=norm
#     )
    
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False
#     gl.xlabel_style = {'size': fontsize}
#     gl.ylabel_style = {'size': fontsize}
#     gl.xformatter = LongitudeFormatter(degree_symbol="° ")
#     gl.yformatter = LatitudeFormatter(degree_symbol="° ")


# # Add a colorbar at the bottom
# cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
# cb.set_label('b', fontsize=fontsize)  
# cb.ax.tick_params(labelsize=fontsize)

    
# plt.show()

letter = ["(c)","(a)","(b)","(d)"]
fontsize = 20

fig = plt.figure(figsize=(12, 17))
gs = GridSpec(3, 2, figure=fig,
              width_ratios = [2.3,1],height_ratios = [1.2,1,1.7],
              hspace = 0, wspace = 0.25)

axes = [fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0:2, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree()),
        ]

legend_elements = [
    plt.Line2D([0], [0], marker = "o",markersize = np.sqrt(s*sig_mod), linestyle = " ", color='k', label=r'$b$ significantly'+' \ndifferent from 0'),
    plt.Line2D([0], [0], marker = "o",markersize  = np.sqrt(s), linestyle = " ", color='k', label=r'not significant'),
]

#loop to go through the countries
for country_i in range(4): 

    axes[country_i].coastlines()
    axes[country_i].add_feature(cfeature.BORDERS, linestyle=':')
    
    axes[country_i].set_title(letter[country_i],fontsize = fontsize+2,loc = "left")
    
    
    
    sc = axes[country_i].scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b==0],
        df_parameters[country_i].latitude[df_parameters[country_i].b==0],
        c=new_df[country_i].b[df_parameters[country_i].b==0],
        s = s,
        cmap='seismic',  
        norm=norm,
    )

    sc = axes[country_i].scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b!=0],
        df_parameters[country_i].latitude[df_parameters[country_i].b!=0],
        c=new_df[country_i].b[df_parameters[country_i].b!=0],
        s = s*sig_mod,
        # edgecolors = "grey",
        cmap='seismic',  
        norm=norm,
    )
    
    if country_i == 1:        
        axes[country_i].legend(handles = legend_elements,fontsize = fontsize)
    else: 
        pass
    
    
    # Set x and y ticks
    gl = axes[country_i].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    if country_i == 0:
        gl.xlocator = FixedLocator([6,9,12,15])
        gl.ylocator = FixedLocator([48,50,52,54])
    elif country_i == 2:
        gl.xlocator = FixedLocator([-7,-4,-1,2])
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    gl.xformatter = LongitudeFormatter(degree_symbol="° ")
    gl.yformatter = LatitudeFormatter(degree_symbol="° ")
    







cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))


cb.set_label(r'$b$ [°C$^{-1}$]', fontsize=fontsize)  
cb.ax.tick_params(labelsize=fontsize)


plt.show()


# exponential b
fig = plt.figure(figsize=(12, 17))
gs = GridSpec(3, 2, figure=fig,
              width_ratios = [2.3,1],height_ratios = [1.2,1,1.7],
              hspace = 0, wspace = 0.25)

axes = [fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0:2, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree()),
        ]

legend_elements = [
    plt.Line2D([0], [0], marker = "o",markersize = np.sqrt(s*sig_mod), linestyle = " ", color='k', label=r'$b_{\mathrm{exp}}$ significantly'+' \ndifferent from 0'),
    plt.Line2D([0], [0], marker = "o",markersize  = np.sqrt(s), linestyle = " ", color='k', label=r'not significant'),
]
#loop to go through the countries
for country_i in range(4): 

    axes[country_i].coastlines()
    axes[country_i].add_feature(cfeature.BORDERS, linestyle=':')
    
    axes[country_i].set_title(letter[country_i],fontsize = fontsize+2,loc = "left")
    
    
    
    sc = axes[country_i].scatter(
        df_parameters_exp[country_i].longitude[df_parameters[country_i].b==0],
        df_parameters_exp[country_i].latitude[df_parameters[country_i].b==0],
        c=df_parameters_exp[country_i].b[df_parameters[country_i].b==0]*df_parameters_exp[country_i].kappa[df_parameters[country_i].b==0],
        s = s,
        cmap='seismic',  
        norm=norm,
    )

    sc = axes[country_i].scatter(
        df_parameters_exp[country_i].longitude[df_parameters[country_i].b!=0],
        df_parameters_exp[country_i].latitude[df_parameters[country_i].b!=0],
        c=df_parameters_exp[country_i].b[df_parameters[country_i].b!=0]*df_parameters_exp[country_i].kappa[df_parameters[country_i].b!=0],
        s = s*sig_mod,
        # edgecolors = "grey",
        cmap='seismic',  
        norm=norm,
    )
    
    if country_i == 1:        
        axes[country_i].legend(handles = legend_elements,fontsize = fontsize)
    else: 
        pass
    
    
    # Set x and y ticks
    gl = axes[country_i].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    if country_i == 0:
        gl.xlocator = FixedLocator([6,9,12,15])
        gl.ylocator = FixedLocator([48,50,52,54])
    elif country_i == 2:
        gl.xlocator = FixedLocator([-7,-4,-1,2])
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    gl.xformatter = LongitudeFormatter(degree_symbol="° ")
    gl.yformatter = LatitudeFormatter(degree_symbol="° ")
    






cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))


cb.set_label(r'$b_{\mathrm{exp}}$ [°C$^{-1}$]', fontsize=fontsize)  
cb.ax.tick_params(labelsize=fontsize)



plt.show()

###############################################################################

# FIG 2
# Synthetic spreads

letter = ["(a)","(b)","(c)","(d)"]
darken = 2

# linear b
params = ["lambda","a","kappa","b"]
params_titles =  [r"$\lambda_0$ [mm h${^{-1}}$]",r"$a$ [°C$^{-1}$]",r"$\kappa_0$ [-]",r"$b$ [°C$^{-1}$]"]
xticks_list = [[0.3,1,3,9],np.arange(-0.07,0.15,0.07),[0.5,1,2,4],np.arange(-0.18,0.1,0.06)]
lims = [[0.11,16],[-0.07,0.15],[0.3,6],[-0.19,0.1]]


fig = plt.figure(figsize=[12,12])
for param_num in range(4):
    ax = fig.add_subplot(1,4,param_num+1)
    
    vln_list = [item 
            for country_num in range(4) 
            for item in [ 
            df_generated_parameters_0[country_num][params[param_num]],
            df_parameters_0[country_num][params[param_num]].copy().dropna(),
                         df_generated_parameters[country_num][params[param_num]],
                         new_df[country_num][params[param_num]].copy().dropna(),]
            ]
    
    violin = plt.violinplot(vln_list,vert=False,showmeans = True)
    
    #stupid workaround to darken the observed plots
    for drk in range(darken):
        violin2 = plt.violinplot([vln_list[l] for l in np.arange(1,17,2)],vert=False,showmeans = True,positions = np.arange(2,17,2))
    
    if param_num == 0:
        plt.yticks(list(np.arange(2,17,2)),
                   
                    [r"$b$ = 0", r"$b$ = free"]*4,
                    
                   rotation = 50,
                   size = fontsize
                   )
        plt.xscale("log")
        
        
    elif param_num == 3:
        plt.yticks(list(np.arange(2.5,17,4)),
                   
                    [
                    'Germany',
                    'Japan',
                    'UK',
                    'USA',],
                    
                   rotation = -90,
                   size = fontsize,
                   verticalalignment = "center",
                   )
        
        
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        
        
        
    else:
        ax.get_yaxis().set_visible(False)
        
        
    if param_num == 2:
        plt.xscale("log")
        
    
    for n in np.arange(0,4):
        violin['bodies'][n].set_facecolor('y')
    for n in np.arange(4,8):
        violin['bodies'][n].set_facecolor('r')   
    for n in np.arange(8,12):
        violin['bodies'][n].set_facecolor('b')  
    for n in np.arange(12,16):
        violin['bodies'][n].set_facecolor('g')   
        
    for n in np.arange(0,2):
        violin2['bodies'][n].set_facecolor('y')
    for n in np.arange(2,4):
        violin2['bodies'][n].set_facecolor('r')   
    for n in np.arange(4,6):
        violin2['bodies'][n].set_facecolor('b')  
    for n in np.arange(6,8):
        violin2['bodies'][n].set_facecolor('g')   


        
    for partname in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
        violin[partname].set_color('k')
        violin2[partname].set_color('k')
     
    plt.grid(axis = 'x')
    plt.plot(lims[param_num],[4.5,4.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[8.5,8.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[12.5,12.5],color = "k",alpha = 0.6)
    
    plt.plot(lims[param_num],[2.5,2.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[6.5,6.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[10.5,10.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[14.5,14.5],color = "k",alpha = 0.3)
    
    plt.xlim(lims[param_num])
    
    ax.text(0.12, 1.03, letter[param_num], transform=ax.transAxes,
      fontsize=fontsize, va='top', ha='right')
    
    
    
    plt.xticks(xticks_list[param_num],fontsize = fontsize,rotation = 45 if param_num%2 == 1 else 0)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    plt.title(params_titles[param_num],fontsize = fontsize)


dark_patch = mpatches.Patch(color='k', alpha = 0.7, label='Observed')
light_patch = mpatches.Patch(color='k', alpha = 0.3, label='MC generated')

plt.legend(handles=[dark_patch,light_patch], loc='lower right', fontsize=fontsize)

plt.subplots_adjust(wspace=0, hspace=0)
#plt.suptitle("Linear", fontsize = fontsize+2)

plt.show()


################################################################################
# exp b
params_titles =  [r"$\lambda_0$ [mm h${^{-1}}$]",r"$a$ [°C$^{-1}$]",r"$\kappa_0$ [-]",r"$b_{\mathrm{exp}}$ [°C$^{-1}$]"]


fig = plt.figure(figsize=[12,12])
for param_num in range(4):
    
    
    ax = fig.add_subplot(1,4,param_num+1)
    
    vln_list = [item 
            for country_num in range(4) 
            for item in [ 
            df_generated_parameters_0[country_num][params[param_num]],
            df_parameters_0[country_num][params[param_num]].copy().dropna(),
                         df_generated_parameters_exp[country_num][params[param_num]],
                         df_parameters_exp[country_num][params[param_num]].copy().dropna(),]
            ]
    
    violin = plt.violinplot(vln_list,vert=False,showmeans = True)
    
    #stupid workaround to darken the observed plots
    for drk in range(darken):
        violin2 = plt.violinplot([vln_list[l] for l in np.arange(1,17,2)],vert=False,showmeans = True,positions = np.arange(2,17,2))
    
    if param_num == 0:
        plt.yticks(list(np.arange(2,17,2)),
                   
                    ["$b$ = 0", "$b$ = free"]*4,
                    
                   rotation = 50,
                   size = fontsize
                   )
        plt.xscale("log")
        
        
    elif param_num == 3:
        plt.yticks(list(np.arange(2.5,17,4)),
                   
                    [
                    'Germany',
                    'Japan',
                    'UK',
                    'USA',],
                    
                   rotation = -90,
                   size = fontsize,
                   verticalalignment = "center",
                   )
        
        
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        
        
        
    else:
        ax.get_yaxis().set_visible(False)
        
        
    if param_num == 2:
        plt.xscale("log")
        
    
    for n in np.arange(0,4):
        violin['bodies'][n].set_facecolor('y')
    for n in np.arange(4,8):
        violin['bodies'][n].set_facecolor('r')   
    for n in np.arange(8,12):
        violin['bodies'][n].set_facecolor('b')  
    for n in np.arange(12,16):
        violin['bodies'][n].set_facecolor('g')   
        
    for n in np.arange(0,2):
        violin2['bodies'][n].set_facecolor('y')
    for n in np.arange(2,4):
        violin2['bodies'][n].set_facecolor('r')   
    for n in np.arange(4,6):
        violin2['bodies'][n].set_facecolor('b')  
    for n in np.arange(6,8):
        violin2['bodies'][n].set_facecolor('g')   


        
    for partname in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
        violin[partname].set_color('k')
        violin2[partname].set_color('k')
     
    plt.grid(axis = 'x')
    plt.plot(lims[param_num],[4.5,4.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[8.5,8.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[12.5,12.5],color = "k",alpha = 0.6)
    
    plt.plot(lims[param_num],[2.5,2.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[6.5,6.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[10.5,10.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[14.5,14.5],color = "k",alpha = 0.3)
    
    plt.xlim(lims[param_num])
    
    ax.text(0.12, 1.03, letter[param_num], transform=ax.transAxes,
      fontsize=fontsize, va='top', ha='right')
    
    
    
    plt.xticks(xticks_list[param_num],fontsize = fontsize,rotation = 45 if param_num%2 == 1 else 0)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    plt.title(params_titles[param_num],fontsize = fontsize)


dark_patch = mpatches.Patch(color='k', alpha = 0.7, label='Observed')
light_patch = mpatches.Patch(color='k', alpha = 0.3, label='MC generated')

plt.legend(handles=[dark_patch,light_patch], loc='lower right', fontsize=fontsize)

plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle("Exponential", fontsize = fontsize+2)
plt.show()

###############################################################################
# the newer version with changing n events
params_titles =  [r"$\lambda_0$ [mm h${^{-1}}$]",r"$a$ [°C$^{-1}$]",r"$\kappa_0$ [-] ",r"$b$ [°C$^{-1}$]"]

fig = plt.figure(figsize=[12,12])
for param_num in range(4):
    ax = fig.add_subplot(1,4,param_num+1)
    
    vln_list = [item 
            for country_num in range(4) 
            for item in [ 
            comb_df_gen_params_0[country_num][params[param_num]],
            df_parameters_0[country_num][params[param_num]].copy().dropna(),
                         comb_df_gen_params[country_num][params[param_num]],
                         new_df[country_num][params[param_num]].copy().dropna(),]
            ]
    
    violin = plt.violinplot(vln_list,vert=False,showmeans = True)
    
    #stupid workaround to darken the observed plots
    for drk in range(darken):
        violin2 = plt.violinplot([vln_list[l] for l in np.arange(1,17,2)],vert=False,showmeans = True,positions = np.arange(2,17,2))
    
    if param_num == 0:
        plt.yticks(list(np.arange(2,17,2)),
                   
                    [r"$b$ = 0", r"$b$ = free"]*4,
                    
                   rotation = 50,
                   size = fontsize
                   )
        plt.xscale("log")
        
        
    elif param_num == 3:
        plt.yticks(list(np.arange(2.5,17,4)),
                   
                    [
                    'Germany',
                    'Japan',
                    'UK',
                    'USA',],
                    
                   rotation = -90,
                   size = fontsize,
                   verticalalignment = "center",
                   )
        
        
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        
        
        
    else:
        ax.get_yaxis().set_visible(False)
        
        
    if param_num == 2:
        plt.xscale("log")
        
    
    for n in np.arange(0,4):
        violin['bodies'][n].set_facecolor('y')
    for n in np.arange(4,8):
        violin['bodies'][n].set_facecolor('r')   
    for n in np.arange(8,12):
        violin['bodies'][n].set_facecolor('b')  
    for n in np.arange(12,16):
        violin['bodies'][n].set_facecolor('g')   
        
    for n in np.arange(0,2):
        violin2['bodies'][n].set_facecolor('y')
    for n in np.arange(2,4):
        violin2['bodies'][n].set_facecolor('r')   
    for n in np.arange(4,6):
        violin2['bodies'][n].set_facecolor('b')  
    for n in np.arange(6,8):
        violin2['bodies'][n].set_facecolor('g')   


        
    for partname in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
        violin[partname].set_color('k')
        violin2[partname].set_color('k')
     
    plt.grid(axis = 'x')
    plt.plot(lims[param_num],[4.5,4.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[8.5,8.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[12.5,12.5],color = "k",alpha = 0.6)
    
    plt.plot(lims[param_num],[2.5,2.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[6.5,6.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[10.5,10.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[14.5,14.5],color = "k",alpha = 0.3)
    
    plt.xlim(lims[param_num])
    
    ax.text(0.12, 1.03, letter[param_num], transform=ax.transAxes,
      fontsize=fontsize, va='top', ha='right')
    
    
    
    plt.xticks(xticks_list[param_num],fontsize = fontsize,rotation = 45 if param_num%2 == 1 else 0)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    plt.title(params_titles[param_num],fontsize = fontsize)


dark_patch = mpatches.Patch(color='k', alpha = 0.7, label='Observed')
light_patch = mpatches.Patch(color='k', alpha = 0.3, label='MC generated')

plt.legend(handles=[dark_patch,light_patch], loc='lower right', fontsize=fontsize)

plt.subplots_adjust(wspace=0, hspace=0)
#plt.suptitle("Linear", fontsize = fontsize+2)

plt.show()

################################################################################
# exp b
params_titles =  [r"$\lambda_0$ [mm h${^{-1}}$]",r"$a$ [°C$^{-1}$]",r"$\kappa_0$ [-]",r"$b_{\mathrm{exp}}$ [°C$^{-1}$]"]


fig = plt.figure(figsize=[12,12])
for param_num in range(4):
    
    
    ax = fig.add_subplot(1,4,param_num+1)
    
    vln_list = [item 
            for country_num in range(4) 
            for item in [ 
            comb_df_gen_params_0[country_num][params[param_num]],
            df_parameters_0[country_num][params[param_num]].copy().dropna(),
                         comb_df_gen_params_exp[country_num][params[param_num]],
                         df_parameters_exp[country_num][params[param_num]].copy().dropna(),]
            ]
    
    violin = plt.violinplot(vln_list,vert=False,showmeans = True)
    
    #stupid workaround to darken the observed plots
    for drk in range(darken):
        violin2 = plt.violinplot([vln_list[l] for l in np.arange(1,17,2)],vert=False,showmeans = True,positions = np.arange(2,17,2))
    
    if param_num == 0:
        plt.yticks(list(np.arange(2,17,2)),
                   
                    ["$b$ = 0", "$b$ = free"]*4,
                    
                   rotation = 50,
                   size = fontsize
                   )
        plt.xscale("log")
        
        
    elif param_num == 3:
        plt.yticks(list(np.arange(2.5,17,4)),
                   
                    [
                    'Germany',
                    'Japan',
                    'UK',
                    'USA',],
                    
                   rotation = -90,
                   size = fontsize,
                   verticalalignment = "center",
                   )
        
        
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        
        
        
    else:
        ax.get_yaxis().set_visible(False)
        
        
    if param_num == 2:
        plt.xscale("log")
        
    
    for n in np.arange(0,4):
        violin['bodies'][n].set_facecolor('y')
    for n in np.arange(4,8):
        violin['bodies'][n].set_facecolor('r')   
    for n in np.arange(8,12):
        violin['bodies'][n].set_facecolor('b')  
    for n in np.arange(12,16):
        violin['bodies'][n].set_facecolor('g')   
        
    for n in np.arange(0,2):
        violin2['bodies'][n].set_facecolor('y')
    for n in np.arange(2,4):
        violin2['bodies'][n].set_facecolor('r')   
    for n in np.arange(4,6):
        violin2['bodies'][n].set_facecolor('b')  
    for n in np.arange(6,8):
        violin2['bodies'][n].set_facecolor('g')   


        
    for partname in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
        violin[partname].set_color('k')
        violin2[partname].set_color('k')
     
    plt.grid(axis = 'x')
    plt.plot(lims[param_num],[4.5,4.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[8.5,8.5],color = "k",alpha = 0.6)
    plt.plot(lims[param_num],[12.5,12.5],color = "k",alpha = 0.6)
    
    plt.plot(lims[param_num],[2.5,2.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[6.5,6.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[10.5,10.5],color = "k",alpha = 0.3)
    plt.plot(lims[param_num],[14.5,14.5],color = "k",alpha = 0.3)
    
    plt.xlim(lims[param_num])
    
    ax.text(0.12, 1.03, letter[param_num], transform=ax.transAxes,
      fontsize=fontsize, va='top', ha='right')
    
    
    
    plt.xticks(xticks_list[param_num],fontsize = fontsize,rotation = 45 if param_num%2 == 1 else 0)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    plt.title(params_titles[param_num],fontsize = fontsize)


dark_patch = mpatches.Patch(color='k', alpha = 0.7, label='Observed')
light_patch = mpatches.Patch(color='k', alpha = 0.3, label='MC generated')

plt.legend(handles=[dark_patch,light_patch], loc='lower right', fontsize=fontsize)

plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle("Exponential", fontsize = fontsize+2)
plt.show()



###############################################################################
#calculating the numbers
# for country_i in range(len(countries)):
#     print(countries[country_i])
#     for param_num in range(4):
#         param = params[param_num]
        
#         gen = df_generated_parameters[country_i][param]
#         obs = new_df[country_i][param]
#         gen_0 = df_generated_parameters_0[country_i][param]
#         obs_0 = df_parameters_0[country_i][param]
        
#         SD_ratio = gen.std()/obs.std()
#         IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)
#                      )/(obs.quantile(0.75) - obs.quantile(0.25))
        
#         SD_ratio_0 = gen_0.std()/obs_0.std()
#         IQR_ratio_0 = (gen_0.quantile(0.75) - gen_0.quantile(0.25)
#                      )/(obs_0.quantile(0.75) - obs_0.quantile(0.25))
        
#         print(f"The ratio of the standard deviations for {param} is {SD_ratio}. When b = 0, it is {SD_ratio_0}.")
#         print(f"The ratio of the IQR for {param} is {IQR_ratio}. When b = 0, it is {IQR_ratio_0}.")
        


# print("\\begin{tabular}{l l c c c c}")
# print("\\toprule")
# print("Country & Parameter & SD Ratio & SD Ratio ($b=0$) & IQR Ratio & IQR Ratio ($b=0$) \\\\")
# print("\\midrule")

# for country_i in range(len(countries)):
#     country_name = countries[country_i]

#     for param_num, param in enumerate(params):
#         gen = df_generated_parameters[country_i][param]
#         obs = new_df[country_i][param]
#         gen_0 = df_generated_parameters_0[country_i][param]
#         obs_0 = df_parameters_0[country_i][param]

#         SD_ratio = gen.std() / obs.std()
#         IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)) / (obs.quantile(0.75) - obs.quantile(0.25))
#         SD_ratio_0 = gen_0.std() / obs_0.std()
#         IQR_ratio_0 = (gen_0.quantile(0.75) - gen_0.quantile(0.25)) / (obs_0.quantile(0.75) - obs_0.quantile(0.25))

#         if param_num == 0:
#             print(f"\\multirow{{{len(params)}}}{{*}}{{{country_name}}} & {param} & {SD_ratio:.2f} & {SD_ratio_0:.2f} & {IQR_ratio:.2f} & {IQR_ratio_0:.2f} \\\\")
#         else:
#             print(f" & {param} & {SD_ratio:.2f} & {SD_ratio_0:.2f} & {IQR_ratio:.2f} & {IQR_ratio_0:.2f} \\\\")

# print("\\bottomrule")
# print("\\end{tabular}")

for country_i in range(len(countries)):
    country_name = countries[country_i]

    for param_num, param in enumerate(params):
        # gen = comb_df_gen_params_exp[country_i][param]
        # obs = df_parameters_exp[country_i][param]
        
        # # SD_ratio = gen.std() / obs.std()
        # IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)) / (obs.quantile(0.75) - obs.quantile(0.25))
        # print(country_name)
        # print(param)
        # print(f" exp IQR ratio: {IQR_ratio}")
        
        
        gen = comb_df_gen_params[country_i][param]
        obs = new_df[country_i][param]
        
        # SD_ratio = gen.std() / obs.std()
        IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)) / (obs.quantile(0.75) - obs.quantile(0.25))
        print(country_name)
        print(param)
        print(f"IQR ratio: {IQR_ratio}")

print("b = 0")
for country_i in range(len(countries)):
    country_name = countries[country_i]

    for param_num, param in enumerate(params):
        # gen = comb_df_gen_params_exp[country_i][param]
        # obs = df_parameters_exp[country_i][param]
        
        # # SD_ratio = gen.std() / obs.std()
        # IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)) / (obs.quantile(0.75) - obs.quantile(0.25))
        # print(country_name)
        # print(param)
        # print(f" exp IQR ratio: {IQR_ratio}")
        
        
        gen = comb_df_gen_params_0[country_i][param]
        obs = df_parameters_0[country_i][param]
        
        # SD_ratio = gen.std() / obs.std()
        IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)) / (obs.quantile(0.75) - obs.quantile(0.25))
        print(country_name)
        print(param)
        print(f"IQR ratio: {IQR_ratio}")


print("exp")
for country_i in range(len(countries)):
    country_name = countries[country_i]

    for param_num, param in enumerate(params):
        # gen = comb_df_gen_params_exp[country_i][param]
        # obs = df_parameters_exp[country_i][param]
        
        # # SD_ratio = gen.std() / obs.std()
        # IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)) / (obs.quantile(0.75) - obs.quantile(0.25))
        # print(country_name)
        # print(param)
        # print(f" exp IQR ratio: {IQR_ratio}")
        
        
        gen = comb_df_gen_params_exp[country_i][param]
        obs = df_parameters_exp[country_i][param]
        
        # SD_ratio = gen.std() / obs.std()
        IQR_ratio = (gen.quantile(0.75) - gen.quantile(0.25)) / (obs.quantile(0.75) - obs.quantile(0.25))
        print(country_name)
        print(param)
        print(f"IQR ratio: {IQR_ratio}")

#################################################################################
# FIG 3
# synthetic FRMSE


gap1 = 0.3 #gaps between the bar plots
gap2 = 1.2
fontsize = 20

colors = ['#377eb8', '#ff7f00', '#4daf4a']*4
    
labels = ["(a)","(b)","(c)"]

ret_lvls = ["10","100"]
# plot the fractionals all together in a different layout
fig = plt.figure(figsize=(12,12))
for i in range(3):
    ax = fig.add_subplot(2,3,i+1)
    
    #plot boxes
    boxplot_list = [synth_RL[years][f"{bstyle}_{ret_lvls[ret_n]}"].dropna()/RL_true[years][ret_n*3] for ret_n in range(2) for bstyle in ["free","set","b0"] for years in [i*3,i*3+2,i*3+1]]
    # positions = np.concat([np.arange(0,4.5,0.5),np.arange(5,9.5,0.5)]) 
    
    base_positions = np.concat([np.arange(0+pos*(1.5+gap1),1.5+pos*(1.5+gap1),0.5) for pos in range(3)])
    positions = np.concat([base_positions,base_positions+(1.5+2*(1.5+gap1))+gap2])
                          
    
    box_plot = ax.boxplot(boxplot_list,
                          positions = positions,
                          showmeans = True, meanline = True, patch_artist=True,
                          meanprops=dict(marker=None, linestyle=':', linewidth=1,color = 'k'),
                          sym = "",
                          whis = [5,95])
    
    strip_list = [box[(box<np.quantile(box,0.05)) | (box>np.quantile(box,0.95))] for box in boxplot_list]
    
    for l in range(len(strip_list)):
        strip_list[l][strip_list[l]>2] = 2
        strip_list[l][strip_list[l]<1/2] = 1/2
    
    strip_data = pd.DataFrame({
        
        "x": np.concatenate([[pos]*len(vals) for pos, vals in zip(positions, strip_list)]),
        "y": np.concatenate(strip_list)
        })
    
    
    seaborn.stripplot(x="x", y="y", data=strip_data, color='black',alpha = 0.5,size = 3,native_scale=True)
    
    
    ax.grid(axis = "y")
    
    
    alpha = [1,0.6,0.3]
    for n_patch in range(len(box_plot['boxes'])):
        patch = box_plot['boxes'][n_patch]
        patch.set_facecolor(colors[int(np.trunc(n_patch/3))])
        patch.set_alpha(alpha[int(n_patch%3)])
                
    for median_line in box_plot["medians"]:
        median_line.set_color('k')
    
    ax.set_xticks([positions[4],positions[13]],ret_lvls,fontsize = fontsize)
    
    ax.set_yscale("log")
    ax.set_ylim(1/2.1,2.1)
    custom_ticks = [1/2,1/1.5,1/1.1, 1, 1.1,1.5,2]
    custom_ticklabels = ["≤ 1/2","1/1.5","1/1.1", "1", "1.1","1.5","≥ 2"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels,fontsize = fontsize)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    ax.text(-0.03, 1.06, labels[i], transform=ax.transAxes,
      fontsize=fontsize+2, va='top', ha='right')
    
    ax.set_title(f"$b$ = {uses[i*3].b[0]}"+r" °C$^{-1}$",fontsize = fontsize+2)
    ax.set_xlabel("Return period (years)",fontsize = fontsize)
    ax.set_ylabel("gen_RL/RL",fontsize = fontsize)
    
    if i == 0:
        legend_elements = [
        plt.Line2D([0], [0], color='k', label='median'),
        plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
        ]
        plt.legend(handles=legend_elements,fontsize = fontsize)

    
    if i == 1:
        legend_elements = [
            Patch(facecolor="k", label='30 years'),
            Patch(facecolor="k", alpha = 0.6, label='20 years'),
            Patch(facecolor="k", alpha = 0.3, label='10 years'),
            ]    
        
        plt.legend(handles=legend_elements,fontsize = fontsize)
    else:
        pass

legend_elements = [
    Patch(facecolor=colors[0], label='Free'),  # default matplotlib colors
    Patch(facecolor=colors[1], label='Set'),
    Patch(facecolor=colors[2], label=r'$b$ = 0'),
]

plt.legend(handles=legend_elements,fontsize = fontsize)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(12,12))
for i in range(3):
    ax = fig.add_subplot(2,3,i+1)
    
    #plot boxes
    boxplot_list = [synth_RL[years][f"{bstyle}_{ret_lvls[ret_n]}_exp"].dropna()/RL_true_exp[years][ret_n*3] for ret_n in range(2) for bstyle in ["free","set","b0"] for years in [i*3,i*3+2,i*3+1]]
    
    base_positions = np.concat([np.arange(0+pos*(1.5+gap1),1.5+pos*(1.5+gap1),0.5) for pos in range(3)])
    positions = np.concat([base_positions,base_positions+(1.5+2*(1.5+gap1))+gap2])
       
    
    box_plot = ax.boxplot(boxplot_list,
                          positions = positions,
                          showmeans = True, meanline = True, patch_artist=True,
                          meanprops=dict(marker=None, linestyle=':', linewidth=1,color = 'k'),
                          sym = "",
                          whis = [5,95])
    
    strip_list = [box[(box<np.quantile(box,0.05)) | (box>np.quantile(box,0.95))] for box in boxplot_list]
    for l in range(len(strip_list)):
        strip_list[l][strip_list[l]>2] = 2
        strip_list[l][strip_list[l]<1/2] = 1/2
    
    strip_data = pd.DataFrame({
        
        "x": np.concatenate([[pos]*len(vals) for pos, vals in zip(positions, strip_list)]),
        "y": np.concatenate(strip_list)
        })
    
    
    seaborn.stripplot(x="x", y="y", data=strip_data, color='black',alpha = 0.5,size = 3,native_scale=True)
    
    plt.grid(axis = "y")
    
    alpha = [1,0.6,0.3]
    for n_patch in range(len(box_plot['boxes'])):
        patch = box_plot['boxes'][n_patch]
        patch.set_facecolor(colors[int(np.trunc(n_patch/3))])
        patch.set_alpha(alpha[int(n_patch%3)])
                
    for median_line in box_plot["medians"]:
        median_line.set_color('k')
    
    plt.xticks([positions[4],positions[13]],ret_lvls,fontsize = fontsize)
    
    plt.yscale("log")
    plt.ylim(1/2.1,2.1)
    
    custom_ticks = [1/2,1/1.5,1/1.1, 1, 1.1,1.5,2]
    custom_ticklabels = ["≤ 1/2","1/1.5","1/1.1", "1", "1.1","1.5","≥ 2"]
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticklabels,fontsize = fontsize)
    ax.yaxis.set_major_locator(plt.FixedLocator(custom_ticks)) 
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("gen_RL/RL",fontsize = fontsize)
    
    ax.text(-0.03, 1.06, labels[i], transform=ax.transAxes,
      fontsize=fontsize+2, va='top', ha='right')
    
    ax.set_title(r"$b_{\mathrm{exp}}$"+f" = {uses[i*3].b[0]}"+r" °C$^{-1}$",fontsize = fontsize+2)
    plt.xlabel("Return period (years)",fontsize = fontsize)
    
    if i == 0:
        legend_elements = [
            plt.Line2D([0], [0], color='k', label='median'),
            plt.Line2D([0], [0], color='k', linestyle = ":", label='mean') 
        ]
        plt.legend(handles=legend_elements,fontsize = fontsize)

    
    
    if i == 1:
        legend_elements = [
            Patch(facecolor="k", label='30 years'),
            Patch(facecolor="k", alpha = 0.6, label='20 years'),
            Patch(facecolor="k", alpha = 0.3, label='10 years'),
            ]    
        
        plt.legend(handles=legend_elements,fontsize = fontsize)
    else:
        pass

legend_elements = [  
    Patch(facecolor=colors[0], label='Free'), 
    Patch(facecolor=colors[1], label='Set'),
    Patch(facecolor=colors[2], label=r'$b_{\mathrm{exp}}$ = 0'),
]

plt.legend(handles=legend_elements,fontsize = fontsize)
plt.suptitle("exponential",fontsize = fontsize+2)

plt.tight_layout()
plt.show()


#################################################################################
# FIG 4
# hindcasts
labels = ["(a)","(b)","(c)","(d)","(e)","(f)","(g)"]
colors = ["y","r","b","g"]
s = 5

norm = mcolors.Normalize(vmin=0, vmax=1)
threshold = 2.5
min_years_strong = 20

lims = [
        [0,13],
        [-0.1,0.15],
        [0.5,4],
        [-0.2,0.1]
        ]

ticks = [
    [0,4,8,12],
    [-0.07,0,0.07,0.14],
    [1,2,3,4],
    [-0.2,-0.1,0,0.1]
    ]

variables = ["kappa","b","lambda","a"]
variables = ["lambda","a","kappa","b"]
params_titles =  [r"$\lambda_0$",r"$a$",r"$\kappa_0$ ",r"$b$"]
param_units = [r"[mm h${^{-1}}$]",r"[°C$^{-1}$]",r"[-]",r"[°C$^{-1}$]"]

hindcasts_comb = pd.concat([hindcasts[country_i][info[country_i].cleaned_years>=min_years_strong] for country_i in range(4)])
hindcasts_comb_exp = pd.concat([hindcasts_exp[country_i][info[country_i].cleaned_years>=min_years_strong] for country_i in range(4)])


fig = plt.figure(figsize = (12,18))


for i in range(4):
    vari = variables[i]
    
    df_small = hindcasts_comb[[f"{vari}1",f"{vari}2",f"{vari}1_0",f"{vari}2_0"]]
    corr_table = df_small.corr()
    
    ax1 = fig.add_subplot(4,2,1+2*i)
    ax2 = fig.add_subplot(4,2,2+2*i)
    
    if i == 0:
        ax1.set_title(f"$b$ = free \n ρ = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}",fontsize = fontsize)
        ax2.set_title(f"$b$ = 0 \n ρ = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}",fontsize = fontsize)
    else:
        ax1.set_title(f"$ρ$ = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}", 
          fontsize=fontsize+2)
        if vari != "b":
            ax2.set_title(f"$ρ$ = {corr_table[f"{vari}1_0"][f"{vari}2_0"]:.2f}", 
              fontsize=fontsize+2)
        
    ax1.set_xlim(lims[i])
    ax1.set_ylim(lims[i])
    
    ax1.set_xlabel(f"{params_titles[i]} first period {param_units[i]}",fontsize = fontsize)
    ax1.set_ylabel(f"{params_titles[i]} second period {param_units[i]}",fontsize = fontsize)
    ax1.tick_params(labelsize=fontsize)
    
    ax1.set_xticks(ticks[i])
    ax2.set_xticks(ticks[i])
    ax1.set_yticks(ticks[i])
    ax2.set_yticks(ticks[i])
    
    ax1.text(0.12, 1.1, labels[i], transform=ax1.transAxes,
      fontsize=fontsize+2, va='top', ha='right')
    
    
    if vari != "b":
        ax2.set_xlim(lims[i])
        ax2.set_ylim(lims[i])
        
        ax2.set_xlabel(f"{params_titles[i]} first period {param_units[i]}",fontsize = fontsize)
        ax2.set_ylabel(f"{params_titles[i]} second period {param_units[i]}",fontsize = fontsize)
        ax2.tick_params(labelsize=fontsize)
        ax2.text(0.12, 1.1, labels[i+4], transform=ax2.transAxes,
          fontsize=fontsize+2, va='top', ha='right')
    if vari == "b":
        ax2.set_axis_off()
    
    for country_i in [3,1,2,0]:
    
        hindcast_Fphat_short = hindcasts[country_i][info[country_i].cleaned_years>=min_years_strong]
        new_df_short = new_df[country_i][info[country_i].cleaned_years>=min_years_strong]
        
        
        if country_i == 3:
            ax1.plot(lims[i],lims[i],label = "line of equality",linestyle = "--")
            if vari != "b":
                ax2.plot(lims[i],lims[i],label = "line of equality",linestyle = "--")
            
        sc = ax1.scatter(hindcast_Fphat_short[f"{vari}1"],hindcast_Fphat_short[f"{vari}2"],
                    s=s,label = countries[country_i],color = colors[country_i],alpha = 0.5)
        
        
        
        # ax1.plot(x_fit,poly_y,label = "best fit, outliers removed",color = "r")
        # ax1.plot(hindcast_Fphat_short[f"{vari}1"].dropna(),poly_y2,label = "best fit")
        
        # ax1.set_title(f"free b. corr = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}")
        
        
        if vari != "b": 
            sc = ax2.scatter(hindcast_Fphat_short[f"{vari}1_0"],hindcast_Fphat_short[f"{vari}2_0"],
                        s=s,label = countries[country_i],color = colors[country_i],alpha = 0.5)
        
        
        
legend_elements = [
    Patch(facecolor=colors[0], alpha = 0.5, label='Germany          '),  # default matplotlib colors
    Patch(facecolor=colors[1], alpha = 0.5, label='UK'),
    Patch(facecolor=colors[2], alpha = 0.5, label='Japan'),
    Patch(facecolor=colors[3], alpha = 0.5, label='USA'),
]

plt.legend(handles = legend_elements, fontsize = fontsize, loc = "upper left")   
plt.tight_layout(w_pad = 6)
plt.show()

    
 

hindcasts_comb_exp.b1 = hindcasts_comb_exp.b1 * hindcasts_comb_exp.kappa1
hindcasts_comb_exp.b2 = hindcasts_comb_exp.b2 * hindcasts_comb_exp.kappa2

for country_i in range(4):
    hindcasts_exp[country_i].b1 = hindcasts_exp[country_i].b1 * hindcasts_exp[country_i].kappa1
    hindcasts_exp[country_i].b2 = hindcasts_exp[country_i].b2 * hindcasts_exp[country_i].kappa2
    

fig = plt.figure(figsize = (8.3,20))

for i in range(4):
    vari = variables[i]
    
    df_small = hindcasts_comb_exp[[f"{vari}1",f"{vari}2"]]
    corr_table = df_small.corr()
    
    
    ax1 = fig.add_subplot(4,1,1+i)
    
    if i == 0:
        ax1.set_title(r"$b_{\mathrm{exp}}$ = free,"+ f"\n $ρ$ = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}",fontsize = fontsize)
    
    else:
        ax1.set_title(f"$ρ$ = {corr_table[f"{vari}1"][f"{vari}2"]:.2f}", 
          fontsize=fontsize+2)
        
    ax1.set_xlim(lims[i])
    ax1.set_ylim(lims[i])
    ax1.tick_params(labelsize=fontsize)
    
    
    ax1.set_xlabel(f"{params_titles[i]} first period {param_units[i]}",fontsize = fontsize)
    ax1.set_ylabel(f"{params_titles[i]} second period {param_units[i]}",fontsize = fontsize)
    
    
    for country_i in [3,1,2,0]:
    
        hindcast_Fphat_exp_short = hindcasts_exp[country_i][info[country_i].cleaned_years>=min_years_strong]
        new_df_short = new_df[country_i][info[country_i].cleaned_years>=min_years_strong]
        
        
        if country_i == 3:
            ax1.plot(lims[i],lims[i],label = "line of equality",linestyle = "--")
            
            
        sc = ax1.scatter(hindcast_Fphat_exp_short[f"{vari}1"],hindcast_Fphat_exp_short[f"{vari}2"],
                    s=s,label = countries[country_i],color = colors[country_i],alpha = 0.5)
        ax1.text(0.12, 1.1, labels[i], transform=ax1.transAxes,
                     fontsize=fontsize+2, va='top', ha='right')

        
        
        # ax1.plot(x_fit,poly_y,label = "best fit, outliers removed",color = "r")
        # ax1.plot(hindcast_Fphat_short[f"{vari}1"].dropna(),poly_y2,label = "best fit")
        
        
        
    
legend_elements = [
    Patch(facecolor=colors[0], alpha = 0.5, label='Germany          '),  # default matplotlib colors
    Patch(facecolor=colors[1], alpha = 0.5, label='UK'),
    Patch(facecolor=colors[2], alpha = 0.5, label='Japan'),
    Patch(facecolor=colors[3], alpha = 0.5, label='USA'),
]

plt.legend(handles = legend_elements, fontsize = fontsize, loc = "upper left")     
plt.tight_layout()
plt.show()

    
###############################################################################
# FIG 5
# hindcast maps
significance = 0.05

changes_pvals = [0]*4
sig_list = [0]*4
sig_list_0 = [0]*4

for country_i in range(4):
    hindcast_Fphat = hindcasts[country_i][info[country_i].cleaned_years>=min_years_strong]
    
    sig_list[country_i] = hindcast_Fphat.pvals > significance #True/1 = insignificant
    sig_list_0[country_i] = hindcast_Fphat.pvals_0 > significance
    
    sig_list[country_i] = sig_list[country_i].astype(int)
    sig_list_0[country_i] = sig_list_0[country_i].astype(int)
    
    changes_pvals[country_i] = sig_list[country_i] + sig_list_0[country_i]*2 # 0 means both sig, 1 means free insig but 0 sig, 2 means free sig then 0 insig, 3 means both insig
    
    frac_0 = len(changes_pvals[country_i][changes_pvals[country_i]==0])/len(changes_pvals[country_i])
    frac_1 = len(changes_pvals[country_i][changes_pvals[country_i]==1])/len(changes_pvals[country_i])
    frac_2 = len(changes_pvals[country_i][changes_pvals[country_i]==2])/len(changes_pvals[country_i])
    frac_3 = len(changes_pvals[country_i][changes_pvals[country_i]==3])/len(changes_pvals[country_i])
    
    
    print(countries[country_i])
    print(f"both sig: {frac_0}")
    print(f"both insig: {frac_3}")
    print(f"b=0 sig, else insig: {frac_1}")
    print(f"b=0 insig, else sig: {frac_2}")
    
    
    



colors = ["r","b","y"]
letter = ["(c)","(a)","(b)","(d)"]
fontsize = 20

fig = plt.figure(figsize=(12, 15.5))
gs = GridSpec(3, 2, figure=fig,
              width_ratios = [2.3,1],height_ratios = [1.2,1,1.7],
              hspace = 0, wspace = 0.25)

axes = [fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0:2, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree()),
        ]



#loop to go through the countries
for country_i in range(4): 

    axes[country_i].coastlines()
    axes[country_i].add_feature(cfeature.BORDERS, linestyle=':')
    
    axes[country_i].set_title(letter[country_i],fontsize = fontsize+2,loc = "left")
    
    new_df_short = new_df[country_i][info[country_i].cleaned_years>=min_years_strong]
    
    
    sc = axes[country_i].scatter(
        new_df_short.longitude[changes_pvals[country_i]==0],
        new_df_short.latitude[changes_pvals[country_i]==0],
        color = colors[0],
        s = s,
        
    )

    sc = axes[country_i].scatter(
        new_df_short.longitude[(changes_pvals[country_i]==1)|(changes_pvals[country_i]==2)],
        new_df_short.latitude[(changes_pvals[country_i]==1)|(changes_pvals[country_i]==2)],
        color = colors[1],
        s = s
    )
    
    sc = axes[country_i].scatter(
        new_df_short.longitude[(changes_pvals[country_i]==3)],
        new_df_short.latitude[(changes_pvals[country_i]==3)],
        color = colors[2],
        s = s
    )
    
    
    
    
    
    # Set x and y ticks
    gl = axes[country_i].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    if country_i == 0:
        gl.xlocator = FixedLocator([6,9,12,15])
        gl.ylocator = FixedLocator([48,50,52,54])
    elif country_i == 2:
        gl.xlocator = FixedLocator([-7,-4,-1,2])
        
    
    if country_i == 1:      
        legend_elements = [
            Patch(facecolor=colors[0], label='significantly different'),  # default matplotlib colors
            Patch(facecolor=colors[2], label='not significantly \ndifferent')]

        first_legend = axes[country_i].legend(handles = legend_elements,fontsize = fontsize,loc = "upper left")
        legend_elements = [
            Patch(facecolor=colors[1], label='significantly different \nin one case only'),
            ]
        axes[country_i].legend(handles = legend_elements,fontsize = fontsize,loc = "lower right")
        axes[country_i].add_artist(first_legend)

        
        
    else: 
        pass
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    gl.xformatter = LongitudeFormatter(degree_symbol="° ")
    gl.yformatter = LatitudeFormatter(degree_symbol="° ")
    
    
            
            
            
    
plt.show()







