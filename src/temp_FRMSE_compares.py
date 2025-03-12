# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:50:36 2025

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
from scipy.stats import anderson
from scipy.stats import gaussian_kde

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import xarray as xr
import time

drive = 'D'

country = 'Japan'
country_save = 'Japan'
code_str = 'JP' 
n_stations = 2 #number of stations to sample
min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
max_yrs = 1000 #if no max, set to very high
name_col = 'ppt'
temp_name_col = "t2m"


temp_output_files = glob.glob(f"{drive}:/outputs/{country_save}/temp_FRMSE*")
df = [0]*len(temp_output_files)
label = [0]*len(temp_output_files)


for i in range(len(temp_output_files)):
    label[i] = temp_output_files[i][len(country_save)+12:-4]
    df[i] = pd.read_csv(temp_output_files[i], dtype={'station': str})


all_temp_FRMSE = pd.DataFrame({
    "station": df[0].station
    })

for i in range(len(temp_output_files)):
    all_temp_FRMSE[f"{label[i]}_upper_perc"] = df[i].FRMSE_upper_perc
    all_temp_FRMSE[label[i]] = df[i].FRMSE
    
    
number_betas = len(temp_output_files)

box_list = [all_temp_FRMSE[lab].copy().dropna() for lab in label]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks(range(1,number_betas+1),label)
plt.title(f'{country} FRMSE')
plt.show()
    
    
#upper percent
box_list = [all_temp_FRMSE[lab].copy().dropna() 
            for lab in [f"{label[i]}_upper_perc" for i in range(len(temp_output_files))]]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks(range(1,number_betas+1),label)
plt.title(f'{country} FRMSE upper 20%')
plt.show()  
    
    
#differences
box_list = [all_temp_FRMSE.temp_FRMSE6.copy().dropna() - all_temp_FRMSE.temp_FRMSE.copy().dropna()]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1],["beta = 6 - beta = 4"])
plt.title(f'{country} FRMSE')
plt.show()  
    
    
#differences 20%
box_list = [all_temp_FRMSE.temp_FRMSE6_upper_perc.copy().dropna() - all_temp_FRMSE.temp_FRMSE_upper_perc.copy().dropna()]

plt.boxplot(box_list,vert=False)
plt.xlabel('FRMSE')
#plt.xlim(-0.1,0.2)
plt.yticks([1],["beta = 6 - beta = 4"])
plt.title(f'{country} FRMSE upper 20%')
plt.show()  
    
