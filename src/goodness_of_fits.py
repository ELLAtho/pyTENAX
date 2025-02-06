# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:58:04 2025

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

#TODO: set subregions of e.g. Japan
###############################################################################
#read in saved bs
save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename) 
df_parameters_neg = pd.read_csv(save_path_neg)

#dataframe with all values
new_df = df_parameters[['latitude','longitude','b']].copy()
mask = new_df['b'] == 0

new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()
###############################################################################


b_set = np.nanmean(new_df.b) #set b_set as average calculated b



#READ IN META INFO FOR COUNTRY
comb = pd.read_csv('D:/metadata/'+country+'_fulldata.csv', dtype={'station': str})

comb.startdate = pd.to_datetime(comb.startdate)
comb.enddate = pd.to_datetime(comb.enddate)


#select stations
val_comb = comb[comb['cleaned_years']>=min_yrs] #filter out stations that are less than min
val_comb = val_comb[val_comb['cleaned_years']<=max_yrs] #filter out stations that are more than max 


#TODO: add in lat and lon conditions to choose from regions. also add plots of region

comb_sort = val_comb.sort_values(by=['cleaned_years'],ascending=0) #sort by size so can choose top sizes


selected = comb_sort[0:n_stations] #choose top n_stations stations

#PLOT SELECTED STATIONS LOCATIONS

fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS)
plt.scatter(selected['longitude'],selected['latitude'])
plt.xlim(np.min(comb.longitude)-5,np.max(comb.longitude)+5)
plt.ylim(np.min(comb.latitude)-5,np.max(comb.latitude)+5)
plt.show()

#READ IN RAIN DATA
files = glob.glob('D:/'+country+'/*') #list of files in country folder
files_sel = [files[i] for i in selected.index]

print(files_sel) #check files and stations the same
print(selected['station'])

#make empty lists to read into
G = [0]*n_stations
data_meta = [0]*n_stations

start_time = time.time()
T_ERA = [0]*n_stations

for i in np.arange(0, n_stations):
    G[i], data_meta[i] =  read_GSDR_file(files_sel[i],name_col)
    T_path = drive + ':/'+country+'_temp\\'+code_str+'_'+str(selected.station[selected.index[i]]) + '.nc'
    T_ERA[i] = xr.load_dataarray(T_path)




print('time to read era5 and ppt '+str(time.time()-start_time))



#get selected lats and lons
lats_sel = [selected.latitude[i] for i in selected.index]
lons_sel = [selected.longitude[i] for i in selected.index]



S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
    )


data_full = [0]*n_stations
for i in np.arange(0,n_stations):
    data_full[i] = S.remove_incomplete_years(G[i], name_col)


RL = [0]*n_stations #TODO: sort this crap out
RL1 = [0]*n_stations
RL_b_set = [0]*n_stations
dicts = [0]*n_stations
thr = [0]*n_stations
F_phats = [0]*n_stations
F_phats1 = [0]*n_stations
F_phats_b_set = [0]*n_stations
g_phats = [0]*n_stations
ns = [0]*n_stations
dict_AMS = [0]*n_stations
eRP = [0]*n_stations
AMS = [0]*n_stations
RMSE = [0]*n_stations
RMSE1 = [0]*n_stations
RMSE_b_set = [0]*n_stations

for i in np.arange(0,n_stations):
    S.alpha = 0
    
    data = data_full[i]
    t_data = (T_ERA[i]-273.15).to_dataframe()

    df_arr = np.array(data[name_col])
    df_dates = np.array(data.index)
    
    #extract indexes of ordinary events
    #these are time-wise indexes =>returns list of np arrays with np.timeindex
    idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
        
    
    #get ordinary events by removing too short events
    #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
    arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
    
    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
    dict_ordinary, dict_AMS[i] = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
    
    
    
    df_arr_t_data = np.array(t_data[temp_name_col])
    df_dates_t_data = np.array(t_data.index)
    
    dicts[i], _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
    
    
    
    start_time = time.time()
    # Your data (P, T arrays) and threshold thr=3.8
    P = dicts[i]["60"]["ordinary"].to_numpy() 
    T = dicts[i]["60"]["T"].to_numpy()  
    
    
    # Number of threshold 
    thr[i] = dicts[i]["60"]["ordinary"].quantile(S.left_censoring[1])
    
    # Sampling intervals for the Montecarlo
    Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
    
    AMS[i] = dict_AMS[i]['60']
    AMS_sort = AMS[i].sort_values(by=['AMS'])['AMS']
    plot_pos = np.arange(1,np.size(AMS_sort)+1)/(1+np.size(AMS_sort))
    eRP[i] = 1/(1-plot_pos)
    S.return_period = eRP[i]
    
    #TENAX MODEL HERE
    #magnitude model
    F_phats[i], loglik, _, _ = S.magnitude_model(P, T, thr[i])
    F_phats_b_set[i],loglik_b_set, _, _ = S.magnitude_model(P, T, thr[i],b_set=b_set)
    #temperature model
    g_phats[i] = S.temperature_model(T)
    # M is mean n of ordinary events
    ns[i] = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    #estimates return levels using MC samples
    RL[i], __, __ = S.model_inversion(F_phats[i], g_phats[i], ns[i], Ts)
    
    S.n_monte_carlo = np.size(P)*S.niter_smev
    _, T_mc, P_mc = S.model_inversion(F_phats[i], g_phats[i], ns[i], Ts,gen_P_mc = True,gen_RL=False) 
    S.n_monte_carlo = 20000
    
       
    
   #PLOTTING THE GRAPHS
    titles = str(i)+': Latitude: '+str(lats_sel[i])+'. Longitude: '+str(lons_sel[i])
    
    
    eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    ylim_perc = [-100,100]
       
    #fig 2b
    TNX_FIG_temp_model(T=T, g_phat=g_phats[i],beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
    plt.title(titles)
    plt.show()
    
    #fig 4 
    
    diffs = RL[i] - AMS_sort
    diffs_frac = diffs/AMS_sort
    
    RMSE[i] = np.sqrt(np.sum(diffs**2)/len(diffs))
    
    fig, ax = plt.subplots()
    
    TNX_FIG_valid(AMS[i],S.return_period,RL[i],xlimits = [1,np.max(S.return_period)+10],ylimits = [0,np.max(np.hstack([RL[i],AMS[i].AMS.to_numpy()]))+3])
    plt.title(titles+'. alpha = 0'+f'. RMSE: {RMSE[i]:.2f}')
    
    ax2 = ax.twinx()
    plt.plot(S.return_period,diffs_frac*100,alpha = 0.5,label = 'percentage difference')
    plt.plot(S.return_period,[0]*len(S.return_period),'k--',alpha = 0.4)
    plt.ylim(ylim_perc)
    plt.legend()
    
    plt.show()
    
    # fig 2a
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phats[i],thr[i],eT,qs)
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(titles+'. alpha = 0')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.show()
    
    
    
    S.alpha = 1
    F_phats1[i], loglik1, _, _ = S.magnitude_model(P, T, thr[i])
    RL1[i], __, __ = S.model_inversion(F_phats1[i], g_phats[i], ns[i], Ts)
    
    #fig 4 (without SMEV and uncertainty)
    diffs1 = RL1[i] - AMS_sort
    diffs_frac1 = diffs1/AMS_sort
    
    RMSE1[i] = np.sqrt(np.sum(diffs1**2)/len(diffs1))
    
    fig, ax = plt.subplots()
    TNX_FIG_valid(AMS[i],S.return_period,RL1[i],xlimits = [1,np.max(S.return_period)+10],ylimits = [0,np.max(np.hstack([RL[i],AMS[i].AMS.to_numpy()]))+3])
    plt.title(titles+'. alpha = 1' + f'. RMSE: {RMSE1[i]:.2f}')
    
    
    ax2 = ax.twinx()
    plt.plot(S.return_period,diffs_frac1*100,alpha = 0.5,label = 'percentage difference')
    plt.plot(S.return_period,[0]*len(S.return_period),'k--',alpha = 0.4)
    plt.ylim(ylim_perc)
    plt.legend()
    
    plt.show()
    
    # fig 2a
    TNX_FIG_magn_model(P,T,F_phats1[i],thr[i],eT,qs)
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(titles+'. alpha = 1')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.show()
    
    
    
    RL_b_set[i], __, __ = S.model_inversion(F_phats_b_set[i], g_phats[i], ns[i], Ts)
    
    #fig 4 (without SMEV and uncertainty)
    diffs_b_set = RL_b_set[i] - AMS_sort
    diffs_frac_b_set = diffs_b_set/AMS_sort
    
    RMSE_b_set[i] = np.sqrt(np.sum(diffs_b_set**2)/len(diffs_b_set))
    
    fig, ax = plt.subplots()
    TNX_FIG_valid(AMS[i],S.return_period,RL_b_set[i],xlimits = [1,np.max(S.return_period)+10],ylimits = [0,np.max(np.hstack([RL[i],AMS[i].AMS.to_numpy()]))+3])
    plt.title(titles+'. b set' + f'. RMSE: {RMSE_b_set[i]:.2f}')
    
    
    ax2 = ax.twinx()
    
    plt.plot(S.return_period,diffs_frac_b_set*100,alpha = 0.5,label = 'fractional difference')
    plt.plot(S.return_period,[0]*len(S.return_period),'k--',alpha = 0.4)
    plt.legend()
    plt.ylim(ylim_perc)
    plt.show()
    
    # fig 2a
    TNX_FIG_magn_model(P,T,F_phats_b_set[i],thr[i],eT,qs)
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(titles+'. b set')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.show()
    
    print(f'alpha = 0 {F_phats[i]}')
    print(f'b set to mean {F_phats_b_set[i]}')
    print(f'alpha = 1 {F_phats1[i]}')
      
    # #TENAX MODEL VALIDATION
    # yrs = dicts[i]["60"]["oe_time"].dt.year
    # yrs_unique = np.unique(yrs)
    # midway = yrs_unique[int(np.ceil(np.size(yrs_unique)/2))]
    
    # #DEFINE FIRST PERIOD
    # P1 = P[yrs<=midway]
    # T1 = T[yrs<=midway]
    # AMS1 = AMS[AMS['year']<=midway]
    # n_ordinary_per_year1 = n_ordinary_per_year[n_ordinary_per_year.index<=midway]
    # n1 = n_ordinary_per_year1.sum() / len(n_ordinary_per_year1)
    
    # #DEFINE SECOND PERIOD
    # P2 = P[yrs>midway]
    # T2 = T[yrs>midway]
    # AMS2 = AMS[AMS['year']>midway]
    # n_ordinary_per_year2 = n_ordinary_per_year[n_ordinary_per_year.index>midway]
    # n2 = n_ordinary_per_year2.sum() / len(n_ordinary_per_year2)
    
    
    # g_phat1 = S.temperature_model(T1)
    # g_phat2 = S.temperature_model(T2)
    
    
    # F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr[i])
    # F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr[i])
    
    # S.n_monte_carlo = 20000  #for RL, set to lower to be safe
    # RL1, __, __ = S.model_inversion(F_phat1, g_phat1, n1, Ts)
    # RL2, __, __ = S.model_inversion(F_phat2, g_phat2, n2, Ts)
    
    # S.n_monte_carlo = np.size(P1)*S.niter_smev #change n_montecarlo for binning
    # _, T_mc1, P_mc1 = S.model_inversion(F_phat1, g_phat1, n1, Ts,gen_P_mc = True,gen_RL=False) 
    # _, T_mc2, P_mc2 = S.model_inversion(F_phat2, g_phat2, n2, Ts,gen_P_mc = True,gen_RL=False) 
    
    

    # S.n_monte_carlo = 20000   
    
    # if F_phats[i][2]==0:
    #     dof=3
    #     alpha1=1; # b parameter is not significantly different from 0; 3 degrees of freedom for the LR test
    # else: 
    #     dof=4
    #     alpha1=0  # b parameter is significantly different from 0; 4 degrees of freedom for the LR test
    
    
    
    
    # #check magnitude model the same in both periods
    # lambda_LR = -2*( loglik - (loglik1+loglik2) )
    # pval = chi2.sf(lambda_LR, dof)
    
    # #modelling second model based on first magnitude and changes in mean/std
    # mu_delta = np.mean(T2)-np.mean(T1)
    # sigma_factor = np.std(T2)/np.std(T1)
    
    # g_phat2_predict = [g_phat1[0]+mu_delta, g_phat1[1]*sigma_factor]
    # RL2_predict, _,_ = S.model_inversion(F_phat1,g_phat2_predict,n2,Ts)
    
    
    # #fig 7a
    
    # TNX_FIG_temp_model(T=T1, g_phat=g_phat1,beta=4,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Temperature model '+str(yrs_unique[0])+'-'+str(midway),xlimits = [eT[0],eT[-1]])
    # TNX_FIG_temp_model(T=T2, g_phat=g_phat2_predict,beta=4,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Temperature model '+str(midway+1)+'-'+str(yrs_unique[-1]),xlimits = [eT[0],eT[-1]]) # model based on temp ave and std changes
    # plt.title('fig 7a')
    # plt.show() #this is slightly different in code and paper I think.. using predicted T vs fitted T
    
    # #fig 7b
    
    # TNX_FIG_valid(AMS1,S.return_period,RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'The TENAX model '+str(yrs_unique[0])+'-'+str(midway),obslabel='Observed annual maxima '+str(yrs_unique[0])+'-'+str(midway))
    # TNX_FIG_valid(AMS2,S.return_period,RL2_predict,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'The predicted TENAX model '+str(midway+1)+'-'+str(yrs_unique[-1]),obslabel='Observed annual maxima '+str(midway+1)+'-'+str(yrs_unique[-1]),ylimits = [0,np.max(AMS.AMS)+3])
    # plt.title('fig 7b')
    # plt.show()
    
    
    print('finished loop '+str(i+1)+' out of '+str(n_stations))
    




