# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:45:31 2025

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

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

from scipy.stats import norm

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

drive='D' #name of drive

country = 'Germany' 
ERA_country = 'Germany'
country_save = 'Germany'
code_str = 'DE_'
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9

save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename) 
TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters
df_parameters_neg = pd.read_csv(save_path_neg)


#dataframe with all values
new_df = df_parameters[['latitude','longitude','b']].copy()
mask = new_df['b'] == 0

new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()

#Fit observed F_hat values to normal distribution

kappa_mu_sigma = norm.fit(new_df.kappa.copy().dropna())
b_mu_sigma = norm.fit(new_df.b.copy().dropna())
lambda_mu_sigma = norm.fit(new_df['lambda'].copy().dropna())
a_mu_sigma = norm.fit(new_df.a.copy().dropna())


#now with beta = 4
kappa_mu_sigma_4 = normal_model(new_df.kappa.copy().dropna(), 4)
b_mu_sigma_4 = normal_model(new_df.b.copy().dropna(), 4)
lambda_mu_sigma_4 = normal_model(new_df['lambda'].copy().dropna(), 4)
a_mu_sigma_4 = normal_model(new_df.a.copy().dropna(), 4)

#now with skew
kappa_skew = normal_model(new_df.kappa.copy().dropna(), method = 'skewnorm')
b_skew = normal_model(new_df.b.copy().dropna(), method = 'skewnorm')
lambda_skew = normal_model(new_df['lambda'].copy().dropna(), method = 'skewnorm')
a__skew = normal_model(new_df.a.copy().dropna(), method = 'skewnorm')



#plot distribution of values b
bins = np.arange(np.min(new_df.b),np.max(new_df.b)+0.005,0.005)
bin_edge = np.concatenate([np.array([bins[0]-(bins[1]-bins[0])/2]),(bins + (bins[1]-bins[0])/2)]) #convert bin centres into bin edges
hist, bin_edges = np.histogram(new_df.b, bins=bin_edge, density=True)
plt.plot(bins, hist, '--', color='b',label = 'observed values of b')

# Plot analytical PDF of b (validation)
plt.plot(bins, gen_norm_pdf(bins, b_mu, b_sigma, 2), '-', color='r', label='fitted b')

# Set plot parameters
#ax.set_xlim(Tlims)
plt.xlabel('b',fontsize=14)
plt.ylabel('pdf',fontsize=14)
# plt.ylim()
# plt.xlim()
plt.legend(fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()

plt.plot(bins, hist, '--', color='b',label = 'observed values of b')

plt.plot(bins, gen_norm_pdf(bins, b_mu_4, b_sigma_4, 4), '-', color='r', label='fitted b, beta = 4')

# Set plot parameters
#ax.set_xlim(Tlims)
plt.xlabel('b',fontsize=14)
plt.ylabel('pdf',fontsize=14)
# plt.ylim()
# plt.xlim()
plt.legend(fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()


TNX_FIG_temp_model(new_df.b, b_skew, 2, bins, obscol='r',valcol='b',
                       obslabel = 'observaed b values',
                       vallabel = f'skewnorm_fit, mu = {b_skew[0]} \n sigma = {b_skew[1]}, skew = {b_skew[2]}',
                       xlimits = [-0.08,0.03],
                       ylimits = [0,30],
                       method = "skewnorm") 
plt.show()