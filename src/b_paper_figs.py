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
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


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
l_moments = [0]*4
l_moments_synth = [0]*4
for country_i in range(4):
    l_moments[country_i] = lmoment(new_df[country_i].b)
    l_moments_synth[country_i] = lmoment(df_generated_parameters[country_i].b)
    
    n_events = np.ceil(new_df[country_i].n_events_per_yr * info[country_i].cleaned_years) # this is wrong because the events per year are wrong
    weights = n_events/np.sum(n_events)
    mean_l1,std_l1 = weighted_avg_and_std(l_moments[country_i][1], weights)



# FIG 1


# FIG 2
# maps of spatial distributions

s = 3
fontsize = 12




fig = plt.figure(figsize=(12, 10))
proj = ccrs.PlateCarree()
for country_i in range(4):
    ax = fig.add_subplot(2, 2, country_i+1, projection=proj)
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    

    norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
    
    sc = ax.scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b==0],
        df_parameters[country_i].latitude[df_parameters[country_i].b==0],
        s = s,
        color = 'darkgrey',  
    )

    sc = ax.scatter(
        df_parameters[country_i].longitude[df_parameters[country_i].b!=0],
        df_parameters[country_i].latitude[df_parameters[country_i].b!=0],
        c=df_parameters[country_i].b[df_parameters[country_i].b!=0],
        s = s,
        cmap='seismic',  
        norm=norm
    )
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    gl.xformatter = LongitudeFormatter(degree_symbol="° ")
    gl.yformatter = LatitudeFormatter(degree_symbol="° ")


# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
cb.set_label('b', fontsize=fontsize)  
cb.ax.tick_params(labelsize=fontsize)

    
plt.show()



fig = plt.figure(figsize=(12, 17))

(topfig, bottomfig) = fig.subfigures(2, 1, height_ratios=(1,1))

(topleft, topright) = topfig.subfigures(1, 2, width_ratios=(2.5,1))
topleft_axs = topleft.add_subplot(1, 1, 1, projection=proj)

topright_axs = topright.subfigures(2, 1, height_ratios=(1.5,1))
topright_ax1 = topright_axs[0].add_subplot(1, 1, 1, projection=proj)
topright_ax2 = topright_axs[1].add_subplot(1, 1, 1, projection=proj)
bottom_axs = bottomfig.add_subplot(1, 1, 1, projection=proj)


axes = [topright_ax2,topleft_axs,topright_ax1,bottom_axs]

#loop to go through the countries
for country_i in range(4): 

    axes[country_i].coastlines()
    axes[country_i].add_feature(cfeature.BORDERS, linestyle=':')
    
    # # Choosing cmap
    # if df_parameters.b.min() == 0:
    #     norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
    # else:
    #     norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())
    
    sc = axes[country_i].scatter( #plot the negligable at 5% lvl points
        new_df[country_i].longitude,
        new_df[country_i].latitude,
        c = new_df[country_i].b,
        s = s,
        cmap = 'seismic',
        norm = norm
    )
    
    
    # Set x and y ticks
    gl = axes[country_i].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}


topfig.subplots_adjust(bottom = 0.3,left=.1, right=.9, wspace=0.2, hspace=.4)


cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
cb.set_label('b', fontsize=fontsize)  
cb.ax.tick_params(labelsize=fontsize)



plt.show()


# exponential b

fig = plt.figure(figsize=(12, 17))

(topfig, bottomfig) = fig.subfigures(2, 1, height_ratios=(1,1))

(topleft, topright) = topfig.subfigures(1, 2, width_ratios=(2.5,1))
topleft_axs = topleft.add_subplot(1, 1, 1, projection=proj)

topright_axs = topright.subfigures(2, 1, height_ratios=(1.5,1))
topright_ax1 = topright_axs[0].add_subplot(1, 1, 1, projection=proj)
topright_ax2 = topright_axs[1].add_subplot(1, 1, 1, projection=proj)
bottom_axs = bottomfig.add_subplot(1, 1, 1, projection=proj)


axes = [topright_ax2,topleft_axs,topright_ax1,bottom_axs]

#loop to go through the countries
for country_i in range(4): 

    axes[country_i].coastlines()
    axes[country_i].add_feature(cfeature.BORDERS, linestyle=':')
    
    # # Choosing cmap
    # if df_parameters.b.min() == 0:
    #     norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
    # else:
    #     norm = mcolors.TwoSlopeNorm(vmin=df_parameters.b.min(), vcenter=0, vmax=-1*df_parameters.b.min())
    
    sc = axes[country_i].scatter( #plot the negligable at 5% lvl points
        df_parameters_exp[country_i].longitude,
        df_parameters_exp[country_i].latitude,
        c = df_parameters_exp[country_i].b,
        s = s,
        cmap = 'seismic',
        norm = norm
    )
    
    
    # Set x and y ticks
    gl = axes[country_i].gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}


topfig.subplots_adjust(bottom = 0.3,left=.1, right=.9, wspace=0.2, hspace=.4)


cb = plt.colorbar(sc, orientation='horizontal',  extend = "both")
cb.set_label('b (exp)', fontsize=fontsize)  
cb.ax.tick_params(labelsize=fontsize)



plt.show()



# FIG 3
# Synthetic spreads

# linear b
params = ["lambda","a","kappa","b"]
params_titles =  [r"$\lambda_0$",r"$a$",r"$\kappa_0$",r"$b$"]
xticks_list = [np.arange(0,15,3),np.arange(-0.1,0.2,0.06),np.arange(0,5),np.arange(-0.18,0.1,0.06)]
letter = ["(a)","(b)","(c)","(d)"]

fig = plt.figure(figsize=[12,12])
for param_num in range(4):
    ax = fig.add_subplot(1,4,param_num+1)
    
    vln_list = [item 
            for country_num in range(4) 
            for item in [new_df[country_num][params[param_num]].copy().dropna(), 
                         df_generated_parameters[country_num][params[param_num]],
                         df_parameters_0[country_num][params[param_num]].copy().dropna(),
                         df_generated_parameters_0[country_num][params[param_num]]]]
    
    violin = plt.violinplot(vln_list,vert=False,showmeans = True)
    
    if param_num == 0:
        plt.yticks(list(np.arange(1,17)),
                   
                    ['Free b',
                    'MC gen',
                    'b=0',
                    'MC gen, b=0']*4,
                    
                   rotation = 50,
                   size = fontsize
                   )
    elif param_num == 3:
        plt.yticks(list(np.arange(1,17)),
                   
                    ['Free b',
                    'MC gen',
                    'b=0',
                    'MC gen, b=0']*4,
                    
                   rotation = -50,
                   size = fontsize,
                   )
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
    else:
        ax.get_yaxis().set_visible(False)
    plt.xticks(size = fontsize)
    
    
    for n in np.arange(0,4):
        violin['bodies'][n].set_facecolor('y')
    for n in np.arange(4,8):
        violin['bodies'][n].set_facecolor('r')   
    for n in np.arange(8,12):
        violin['bodies'][n].set_facecolor('g')   
    for n in np.arange(12,16):
        violin['bodies'][n].set_facecolor('b')   


        
    for partname in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
        violin[partname].set_color('k')
     
    plt.grid(axis = 'x')
    
    ax.text(0.12, 1.03, letter[param_num], transform=ax.transAxes,
      fontsize=fontsize, va='top', ha='right')
    
    
    
    plt.xticks(xticks_list[param_num])
    plt.title(params_titles[param_num])


yellow_patch = mpatches.Patch(color='y', label='Germany')
red_patch = mpatches.Patch(color='r', label='Japan')
green_patch = mpatches.Patch(color='g', label='UK')
blue_patch = mpatches.Patch(color='b', label='USA')

plt.legend(handles=[blue_patch, green_patch, red_patch, yellow_patch], loc='upper right', fontsize=fontsize)

plt.subplots_adjust(wspace=0, hspace=0)

plt.show()
