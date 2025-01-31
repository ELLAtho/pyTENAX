# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:33:48 2025

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
import xarray as xr

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import glob
import time

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning
warnings.simplefilter("ignore", IterationLimitWarning)



drive = 'D'
name_col = 'ppt'
temp_name_col = 't2m'

code = '083186' #identifier of station we are looking at
ppt_path = f"{drive}:/US/US_{code}.txt"
temp_path = f"{drive}:/US_temp\\US_{code}.nc"
F_loc = np.array([25, -83, 31, -78])

G,data_meta = read_GSDR_file(ppt_path,name_col) #read ppt data
T_ERA = xr.load_dataarray(temp_path) # read temp data


fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()

ax1.scatter(data_meta.longitude,data_meta.latitude,1000,'r','x')

plt.xlim(F_loc[1]-2,F_loc[3]+2)
plt.ylim(F_loc[0]-2,F_loc[2]+2)
plt.show()

start_time = time.time()

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.9],
        separation = 6,
        alpha = 0.05,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )


data = G 
data = S.remove_incomplete_years(data, name_col)
t_data = (T_ERA.squeeze()-273.15).to_dataframe()

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

#extract indexes of ordinary events
#these are time-wise indexes =>returns list of np arrays with np.timeindex
idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
    

#get ordinary events by removing too short events
#returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)

#assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)


df_arr_t_data = np.array(t_data[temp_name_col])
df_dates_t_data = np.array(t_data.index)

dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)


# Your data (P, T arrays) and threshold thr=3.8
P = dict_ordinary["60"]["ordinary"].to_numpy() 
T = dict_ordinary["60"]["T"].to_numpy() 
blocks_id = dict_ordinary["60"]["year"].to_numpy()  


# Number of threshold 
thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])

# Sampling intervals for the Montecarlo
Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)

#TENAX MODEL HERE
#magnitude model
F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
#temperature model
g_phat = S.temperature_model(T)
# M is mean n of ordinary events
n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
#estimates return levels using MC samples

#redefining the return period so it is calculated the same as the observations

AMS = dict_AMS['60']
AMS_sort = AMS.sort_values(by=['AMS'])['AMS']
plot_pos = np.arange(1,np.size(AMS_sort)+1)/(1+np.size(AMS_sort))
eRP = 1/(1-plot_pos)
S.return_period = eRP


RL, _, P_check = S.model_inversion(F_phat, g_phat, n, Ts) 
print(RL)
elapsed_time = time.time() - start_time
# Print the elapsed time
print(f"Elapsed time TENAX : {elapsed_time:.4f} seconds")



start_time = time.time()

S.n_monte_carlo = 20000

# # tENAX uncertainty
# start_time = time.time()
# F_phat_unc, g_phat_unc, RL_unc, n_unc, n_err = S.TNX_tenax_bootstrap_uncertainty(P, T, blocks_id, Ts)

# elapsed_time = time.time() - start_time
# print(f"Time to do TENAX uncertainty: {elapsed_time:.4f} seconds")

# # SMEV and its uncertainty
# start_time = time.time()
# #TODO: clean this part cause it is a bit messy with namings
# S_SMEV = SMEV(threshold=0.1,
#               separation = 24,
#               return_period = S.return_period,
#               durations = S.durations,
#               time_resolution = 5, #time resolution in minutes
#               min_duration = 30 ,
#               left_censoring = [S.left_censoring[1],1])      

# #estimate shape and  scale parameters of weibull distribution
# smev_shape,smev_scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)
# #estimate return period (quantiles) with SMEV
# smev_RL = S_SMEV.smev_return_values(S_SMEV.return_period, smev_shape, smev_scale, n.item())

# smev_RL_unc = S_SMEV.SMEV_bootstrap_uncertainty(P, blocks_id, S.niter_smev, n.item())


# elapsed_time = time.time() - start_time
# print(f"Time to do SMEV and uncertainty: {elapsed_time:.4f} seconds")



#PLOTTING THE GRAPHS

eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end

# fig 2a
qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
plt.ylabel('10-minute precipitation (mm)')
plt.title('fig 2a')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.show()

#fig 2b
TNX_FIG_temp_model(T, g_phat,S.beta,eT,xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,5/(np.max(T)-np.min(T))])
plt.title('fig 2b')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.show()

#fig 4 
AMS = dict_AMS['60'] # yet the annual maxima
TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+10])
plt.title('fig 4')
plt.ylabel('60-minute precipitation (mm)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.show()


#fig 5 
iTs = np.arange(-2.5,37.5,1.5) #idk why we need a different T range here 
S.n_monte_carlo = np.size(P)*S.niter_smev
_, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,gen_P_mc = True,gen_RL=False) 
elapsed_time = time.time() - start_time
# Print the elapsed time
print(f"Elapsed time model_inversion all: {elapsed_time:.4f} seconds")
scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phat,S.niter_smev,eT,iTs,xlimits = [np.min(T)-3,np.max(T)+3])
plt.title('fig 5')
plt.ylabel('60-minute precipitation (mm)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.show()



#TENAX MODEL VALIDATION
S.n_monte_carlo = 20000 # set number of MC for getting RL
yrs = dict_ordinary["60"]["oe_time"].dt.year
yrs_unique = np.unique(yrs)
midway = yrs_unique[int(np.ceil(np.size(yrs_unique)/2))-1] # -1 to adjust indexing because this returns a sort of length



#DEFINE FIRST PERIOD
P1 = P[yrs<=midway]
T1 = T[yrs<=midway]
AMS1 = AMS[AMS['year']<=midway]
n_ordinary_per_year1 = n_ordinary_per_year[n_ordinary_per_year.index<=midway]
n1 = n_ordinary_per_year1.sum() / len(n_ordinary_per_year1)

#DEFINE SECOND PERIOD
P2 = P[yrs>midway]
T2 = T[yrs>midway]
AMS2 = AMS[AMS['year']>midway]
n_ordinary_per_year2 = n_ordinary_per_year[n_ordinary_per_year.index>midway]
n2 = n_ordinary_per_year2.sum() / len(n_ordinary_per_year2)


g_phat1 = S.temperature_model(T1)
g_phat2 = S.temperature_model(T2)


F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr)
RL1, _, _ = S.model_inversion(F_phat1, g_phat1, n1, Ts)
   

F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr)
RL2, _, _ = S.model_inversion(F_phat2, g_phat2, n2, Ts)   

if F_phat[1]==0: #check if b parameter is 0 (shape=shape_0*b
    dof=3
    alpha1=1; # b parameter is not significantly different from 0; 3 degrees of freedom for the LR test
else: 
    dof=4
    alpha1=0  # b parameter is significantly different from 0; 4 degrees of freedom for the LR test


###############################################################################

month_names = ['jan','feb','mar', 'apr', 'may', 'jun','jul','aug','sep','oct','nov','dec']
#SPLITTING INTO SUMMER/WINTER


S.beta = 2
season_separations = [6, 9] #THIS IS COUNTING FROM 1. Summer includes both
#DOING THIS PROPERLY MIGHT NEED A TOTAL REDO OF ORDINARY EVENTS
#t_data_summer = t_data[(t_data.index.month<=season_separations[1])&(t_data.index.month>=season_separations[0])]


months = dict_ordinary["60"]["oe_time"].dt.month
winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
summer_inds = months.index[(months<=season_separations[1])&(months>=season_separations[0])]
T_winter = T[winter_inds]
T_summer = T[summer_inds]
P_summer = P[summer_inds]
n_summer = n/2 #TODO: this is a guess and wrong, need to do properly


g_phat_winter = S.temperature_model(T_winter,beta = 2)
g_phat_summer = S.temperature_model(T_summer,beta = 2)


thr_summer = np.quantile(P_summer,S.left_censoring[1])

#TENAX MODEL HERE
#magnitude model
F_phat_summer, loglik_summer, _, _ = S.magnitude_model(P_summer, T_summer, thr_summer)

winter_pdf = gen_norm_pdf(eT, g_phat_winter[0], g_phat_winter[1], 2)
summer_pdf = gen_norm_pdf(eT, g_phat_summer[0], g_phat_summer[1], 2)

combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))

RL_summer, _, _ = S.model_inversion(F_phat_summer, g_phat_summer, n_summer, Ts) 


#fig 3

TNX_FIG_temp_model(T=T_summer, g_phat=g_phat_summer,beta=2,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Summer')
TNX_FIG_temp_model(T=T_winter, g_phat=g_phat_winter,beta=2,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Winter')
TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,obscol='k',valcol='k',obslabel = None,vallabel = 'Annual',xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,7/(np.max(T)-np.min(T))])
plt.plot(eT,combined_pdf,'m',label = 'Combined summer and winter')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.show()


# fig 2a
qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P_summer,T_summer,F_phat_summer,thr_summer,eT,qs,xlimits = [np.min(T)-3,np.max(T)+3])
plt.ylabel('60-minute precipitation (mm)')
plt.title(f'Using only summer data ({month_names[season_separations[0]-1]} to {month_names[season_separations[1]-1]})')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.show()

#fig 2b
TNX_FIG_temp_model(T_summer, g_phat_summer,S.beta,eT,xlimits = [np.min(T)-3,np.max(T)+3],ylimits = [0,10/(np.max(T)-np.min(T))])
plt.title(f'Using only summer data ({month_names[season_separations[0]-1]} to {month_names[season_separations[1]-1]})')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.show()


#fig 4 
AMS = dict_AMS['60'] # yet the annual maxima
TNX_FIG_valid(AMS,S.return_period,RL_summer,ylimits = [0,np.max(AMS.AMS)+10])
plt.title(f'Using only summer data ({month_names[season_separations[0]-1]} to {month_names[season_separations[1]-1]})')
plt.ylabel('60-minute precipitation (mm)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.show()



#fig 5 
scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phat,S.niter_smev,eT,iTs,xlimits = [np.min(T)-3,np.max(T)+3])
plt.scatter(T_summer,P_summer,color = 'b',s=1.5,alpha = 0.3, label = 'Summer values')
plt.title('fig 5')
plt.ylabel('60-minute precipitation (mm)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.show()

# # SENSITIVITY ANALYSIS

# # changes in T mean, std, and n to chekc sensitivity
# delta_Ts = [-1, 1, 2, 3]
# delta_as = [.9, .95, 1.05, 1.1, 1.2]
# delta_ns = [.5, .75, 1.3, 2]

# # T mean sensitivity
# T_sens = np.zeros([np.size(delta_Ts),np.size(S.return_period)])
# i=0
# while i<np.size(delta_Ts):
    
#     T_sens[i,:],_,_ = S.model_inversion(F_phat, [g_phat[0]+delta_Ts[i],g_phat[1]], n, Ts) 
#     i=i+1

# # T std sensitivity

# as_sens = np.zeros([np.size(delta_as),np.size(S.return_period)])
# i=0
# while i<np.size(delta_as):
    
#     as_sens[i,:],_,_ = S.model_inversion(F_phat, [g_phat[0],g_phat[1]*delta_as[i]], n, Ts) 
#     i=i+1

# # n sensitivity
# n_sens = np.zeros([np.size(delta_ns),np.size(S.return_period)])
# i=0
# while i<np.size(delta_ns):
    
#     n_sens[i,:],_,_ = S.model_inversion(F_phat, g_phat, n*delta_ns[i], Ts) 
#     i=i+1


# #fig 6
# fig = plt.figure(figsize = (17,5))
# ax1 = fig.add_subplot(1,3,1)
# i = 0
# while i< np.size(delta_Ts)-1:  
#     ax1.plot(S.return_period,T_sens[i],'k',alpha = 0.7)
#     plt.text(S.return_period[-1]+10, T_sens[i][-1], '{0:+}'.format(delta_Ts[i])+'°C', ha='left', va='center')
#     i=i+1
# #plot last one differently
# ax1.plot(S.return_period,T_sens[i],'k',alpha = 0.7)
# plt.text(S.return_period[-3], T_sens[i][-1], 'μ\'=μ'+'{0:+}'.format(delta_Ts[i])+'°C', ha='left', va='center')   
    
# plt.xscale('log')
# ax1.plot(S.return_period,RL,'b')
# ax1.set_title('Sensitivity to changes in mean temp')
# plt.xscale('log')
# plt.xlim(1,200)
# plt.ylim(0,60)

# ax2 = fig.add_subplot(1,3,2)
# i = 0
# while i< np.size(delta_as)-1:  
#     ax2.plot(S.return_period,as_sens[i],'k',alpha = 0.7)
#     plt.text(S.return_period[-1]+10, as_sens[i][-1], str(delta_as[i])+'σ', ha='left', va='center')
#     i=i+1
# ax2.plot(S.return_period,as_sens[i],'k',alpha = 0.7)
# plt.text(S.return_period[-3]+20, as_sens[i][-1], 'σ\'='+str(delta_as[i])+'σ', ha='left', va='center')

    
# plt.xscale('log')
# ax2.plot(S.return_period,RL,'b',label = 'The TENAX MODEL')
# ax2.set_title('Sensitivity to changes in temp std')
# plt.legend()
# plt.xscale('log')
# plt.xlim(1,200)
# plt.ylim(0,60)

# ax3 = fig.add_subplot(1,3,3)
# i = 0
# while i< np.size(delta_ns)-1:  
#     ax3.plot(S.return_period,n_sens[i],'k',alpha = 0.7,label = str(delta_ns[i]))
#     plt.text(S.return_period[-1]+10, n_sens[i][-1], str(delta_ns[i])+'n', ha='left', va='center')
#     i=i+1
    
# ax3.plot(S.return_period,n_sens[i],'k',alpha = 0.7)
# plt.text(S.return_period[-3]+20, n_sens[i][-1], 'n\'='+str(delta_ns[i])+'n', ha='left', va='center')

# plt.xscale('log')
# ax3.plot(S.return_period,RL,'b')
# ax3.set_title('Sensitivity to changes in mean events per year (n)')
# plt.xscale('log')
# plt.xlim(1,200)
# plt.ylim(0,60)


# plt.show()

# #TODO: n looks a litte different from in paper





