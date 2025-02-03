# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:31:50 2025

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
import matplotlib.patches as mpatches

from scipy.stats import norm

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob

drive='D' #name of drive

read_countries = ['Germany' ,'Germany_b0','Japan','Japan_b0','US_main','US_main_b0']
df_generated_parameters = []

for country_name in read_countries:
    df_gen_savename = drive + ':/outputs/'+country_name+'\\synth_generated_parameters.csv'
    df_generated_parameters.append(pd.read_csv(df_gen_savename))
    
## READ IN FILES
TENAX_use = []
df_parameters = []
new_df = []

for country_save in read_countries:
    save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
    df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'
    
    
    df_parameters_temp = pd.read_csv(df_savename) 
    TENAX_use.append(pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv')) #save calculated parameters
    
    if np.size(glob.glob(save_path_neg)) != 0:
        df_parameters_neg = pd.read_csv(save_path_neg)
    
        #dataframe with all values
        new_df_temp = df_parameters_temp[['latitude','longitude','b']].copy()
        mask = new_df_temp['b'] == 0
        
        new_df_temp.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
        new_df_temp.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
        new_df_temp.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
        new_df_temp.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()
        
        new_df.append(new_df_temp)
        
    else:
        new_df.append(df_parameters.copy()) 
    
    df_parameters.append(df_parameters_temp)


#kappa on all 3

fig = plt.figure(figsize=[10,15])
violin = plt.violinplot(
    [
        new_df[0].kappa.copy().dropna(), #GERMANY
        df_parameters[0].kappa.copy().dropna(),
        df_parameters[1].kappa.copy().dropna(),
        df_generated_parameters[0].kappa,
        df_generated_parameters[1].kappa,
        
        new_df[2].kappa.copy().dropna(), #JAPAN
        df_parameters[2].kappa.copy().dropna(),
        df_parameters[3].kappa.copy().dropna(),
        df_generated_parameters[2].kappa,
        df_generated_parameters[3].kappa,
        
        new_df[4].kappa.copy().dropna(), #US
        df_parameters[4].kappa.copy().dropna(),
        df_parameters[5].kappa.copy().dropna(),
        df_generated_parameters[4].kappa,
        df_generated_parameters[5].kappa,
    ],
    vert=False,
    showmeans = True)
plt.yticks(list(np.arange(1,16)),
           
            ['Free b',
            '5% sig b',
            'b=0',
            'MC gen',
            'MC gen, b=0']*3,
            
           rotation = 50,
           size = 20
           )
for n in np.arange(0,5):
    violin['bodies'][n].set_facecolor('y')
for n in np.arange(5,10):
    violin['bodies'][n].set_facecolor('r')   
for n in np.arange(10,15):
    violin['bodies'][n].set_facecolor('b')   


    
for partname in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
    violin[partname].set_color('k')
 
    
yellow_patch = mpatches.Patch(color='y', label='Germany')
red_patch = mpatches.Patch(color='r', label='Japan')
blue_patch = mpatches.Patch(color='b', label='USA')

plt.legend(handles=[blue_patch, red_patch, yellow_patch], loc='upper right', fontsize=15)

plt.xticks(size = 20)
    
plt.title(r'$κ_0$',size = 25)
plt.show()






#a on all 3

fig = plt.figure(figsize=[10,15])
violin = plt.violinplot(
    [
        new_df[0].a.copy().dropna(), #GERMANY
        df_parameters[0].a.copy().dropna(),
        df_parameters[1].a.copy().dropna(),
        df_generated_parameters[0].a,
        df_generated_parameters[1].a,
        
        new_df[2].a.copy().dropna(), #JAPAN
        df_parameters[2].a.copy().dropna(),
        df_parameters[3].a.copy().dropna(),
        df_generated_parameters[2].a,
        df_generated_parameters[3].a,
        
        new_df[4].a.copy().dropna(), #US
        df_parameters[4].a.copy().dropna(),
        df_parameters[5].a.copy().dropna(),
        df_generated_parameters[4].a,
        df_generated_parameters[5].a,
    ],
    vert=False,
    showmeans = True)
plt.yticks(list(np.arange(1,16)),
           
            ['Free b',
            '5% sig b',
            'b=0',
            'MC gen',
            'MC gen, b=0']*3,
            
           rotation = 50,
           size = 20
           )
for n in np.arange(0,5):
    violin['bodies'][n].set_facecolor('y')
for n in np.arange(5,10):
    violin['bodies'][n].set_facecolor('r')   
for n in np.arange(10,15):
    violin['bodies'][n].set_facecolor('b')   


    
for partname in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
    violin[partname].set_color('k')
 
    
yellow_patch = mpatches.Patch(color='y', label='Germany')
red_patch = mpatches.Patch(color='r', label='Japan')
blue_patch = mpatches.Patch(color='b', label='USA')

plt.legend(handles=[blue_patch, red_patch, yellow_patch], loc='upper right', fontsize=15)

plt.xticks(size = 20)
plt.title(r'a',size = 25)
plt.show()





    
    
    
    
    
    