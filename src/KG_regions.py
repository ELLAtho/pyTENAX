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
import matplotlib.patches as mpatches


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

grouped_kgb = dict({"alpine":["ET","Dfc","Dsc","Dwc"], #Dwc added by me.. might not be here
                    "arid_cold":["BSk","BWk"],
                    "arid_hot": ["BSh","BWh"],
                    "humid_continental": ["Dfa","Dfb","Dwa","Dwb"],
                    "humid_subtropical": ["Cfa"],
                    "mediterranean":["Csa","Csb","Csc","Dsa","Dsb"],
                    "oceanic": ["Cfb","Cfc"],
                    "ocean": ["Ocean"],
                    "tropical": ["Af","Am","Aw","As","Cwa",]
                    }) #Cfc, Csc, Dwc, As added by me... could be wrong


inverse_mapping = {item: key for key, values in grouped_kgb.items() for item in values}

for c in countries:
    dd = pd.read_csv(drive+':/metadata/'+c+'_fulldata.csv', dtype={'station': str})
    n_drop = len(dd.columns)-9
    dd = dd.drop(dd.columns[0:n_drop], axis=1)
    dd["kgb_zone"] = dd.apply(lambda row: kgcpy.lookupCZ(row["latitude"], row["longitude"]), axis=1)
    dd['kgb_group'] = dd['kgb_zone'].map(inverse_mapping)
    #dd.to_csv(drive+':/metadata/'+c+'_fulldata.csv', index = False)
    info.append(dd)
    

colors_map = dict({"alpine":"grey", #Dwc added by me.. might not be here
                    "arid_cold":"pink",
                    "arid_hot": "r",
                    "humid_continental": "c",
                    "humid_subtropical": "greenyellow",
                    "mediterranean":"yellow",
                    "oceanic": "g",
                    "ocean": "k",
                    "tropical": "b"
                    })


for i in np.arange(1,len(info)):
    
    curr_info = info[i]
    curr_info["color"] = curr_info['kgb_group'].map(colors_map)
    
    
    lon_lims = [truncate_neg(np.min(curr_info.longitude),2.5),np.ceil(np.max(curr_info.longitude/2.5))*2.5]
    lat_lims = [truncate_neg(np.min(curr_info.latitude),2.5),np.ceil(np.max(curr_info.latitude/2.5))*2.5]


    fig = plt.figure(figsize=(20, 20))
    norm = mcolors.Normalize(vmin=0, vmax=1)


    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax1.scatter(
        curr_info.longitude,
        curr_info.latitude,
        c=curr_info.color.to_list(),
        norm = norm,
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax1.set_title("kgb zones")
    
    unique_groups = curr_info[['kgb_group', 'color']].drop_duplicates()
    legend_handles = [
        mpatches.Patch(color=row['color'], label=row['kgb_group']) 
        for _, row in unique_groups.iterrows()
        ]

    # Add legend
    plt.legend(handles=legend_handles, title='KGB Groups')
    
    plt.show()


































