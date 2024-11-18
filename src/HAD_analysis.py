# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:21:11 2024

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
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
import xarray as xr
import time
import matplotlib.pyplot as plt

#country = 'Japan'
country = 'Germany'
#code_str = 'JP_' 
code_str = 'DE_'
#minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_col = 'ppt'
temp_name_col = "temperatures"
min_yrs = 19 #atm this probably introduces a bug... need to put in if statement or something



Had_stations = pd.read_csv("D:/HadISD/HAD_metadata.txt",names = ['station','latitude','longitude','elevation'],sep=r"\s+")
Had_spec = Had_stations[(Had_stations.latitude>=minlat) & (Had_stations.latitude<=maxlat)]
Had_spec = Had_spec[(Had_spec.longitude>=minlon) & (Had_spec.longitude<=maxlon)]
Had_spec.latitude = round(Had_spec.latitude, 2)
Had_spec.longitude = round(Had_spec.longitude, 2)


info = pd.read_csv('D:/metadata/'+country+'_fulldata.csv')
info.station = info['station'].apply(lambda x: f'{int(x):05}') #need to edit this according to file
info = info.rename(columns={'station': 'station_info'})
info.latitude = round(info.latitude, 2)
info.longitude = round(info.longitude, 2)



#get stations with identical lat and lon to Had
matched_stations_full = info.merge(Had_spec, on=['latitude', 'longitude'])
matched_stations = matched_stations_full[matched_stations_full['cleaned_years']>=min_yrs]


plt.scatter(info.longitude,info.latitude,label = 'GSDR')
plt.scatter(Had_spec.longitude,Had_spec.latitude,label = 'HAD')
plt.scatter(matched_stations.longitude,matched_stations.latitude,label ='matches')
plt.legend()
plt.show()



# GET LIST OF FILES OF MATCHED HAD STATIONS
files = [glob.glob("D:/HadISD/unzipped/*/*"+str(file)+"*.nc") for file in matched_stations.station]
G_files = [glob.glob('D:/'+country+'/*'+str(file)+'*') for file in matched_stations.station_info]


###############################################################################




g_phats = []
F_phats = []
scaling_rate_Ws, scaling_rate_qs  = [],[]

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
    )

for n in np.arange(1,len(G_files)):
    HAD = xr.open_dataset(files[n][0]).temperatures
    HAD[HAD < -1000] = np.nan
    
    
    
    G = pd.read_csv(G_files[n][0], skiprows=21, names=[name_col])
    data_meta = readIntense(G_files[n][0], only_metadata=True, opened=False)
    
    
       
    #extract start and end dates from metadata
    start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
    end_date_G= dt.datetime.strptime(data_meta.end_datetime, "%Y%m%d%H")
    
    time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
    # replace -999 with nan
    G[G == -999] = np.nan
    
    G['prec_time'] = time_list_G
    G = G.set_index('prec_time')
    
    
    start_time = time.time()
    data = S.remove_incomplete_years(G, name_col)
    
    #get data from pandas to numpy array
    df_arr = np.array(data[name_col])
    df_dates = np.array(data.index)
    
    idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
        
    
    #get ordinary events by removing too short events
    #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
    arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
    
    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
    dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
    
    elapsed_time = time.time() - start_time
    # Print the elapsed time
    print(f"Elapsed time get OE: {elapsed_time:.4f} seconds")
    
    
    t_data = HAD.to_dataframe()
    
    
    start_time = time.time()
    df_arr_t_data = np.array(t_data[temp_name_col])
    df_dates_t_data = np.array( t_data.index)
    
    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
    
    elapsed_time = time.time() - start_time
    # Print the elapsed time
    print(f"Elapsed time : {elapsed_time:.4f} seconds")
    
    
    start_time = time.time()
    # Your data (P, T arrays) and threshold thr=3.8
    P = dict_ordinary["60"]["ordinary"].to_numpy() # Replace with your actual data
    T = dict_ordinary["60"]["T"].to_numpy()  # Replace with your actual data
    
    
    
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
    RL, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,n_mc = np.size(P)*S.niter_smev) 
    print(RL)
    
    g_phats.append(g_phat)
    F_phats.append(F_phat)
    
    #PLOTTING THE GRAPHS
    
    eT = np.arange(np.min(T)-4,np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    # fig 2a
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs,xlimits = [eT[0],eT[-1]])
    plt.ylabel('60-minute precipitation (mm)')
    plt.title(f'fig 2a. {data_meta.latitude,data_meta.longitude}')
    plt.show()
    
    #fig 2b
    TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
    plt.title('fig 2b')
    plt.show()
    
    #fig 4 
    AMS = dict_AMS['60'] # yet the annual maxima
    TNX_FIG_valid(AMS,S.return_period,RL,ylimits = [0,np.max(AMS.AMS)+3])
    plt.title('fig 4')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    
    #fig 5 
    iTs = np.arange(np.min(T)-4,np.max(T)+10,1.5) #idk why we need a different T range here 
    
    scaling_rate_W, scaling_rate_q = TNX_FIG_scaling(P,T,P_mc,T_mc,F_phat,S.niter_smev,eT,iTs,xlimits = [eT[0],eT[-1]])
    plt.title('fig 5')
    plt.ylabel('60-minute precipitation (mm)')
    plt.show()
    
    scaling_rate_Ws.append(scaling_rate_W)
    scaling_rate_qs.append(scaling_rate_q)
    
    # #SPLITTING INTO SUMMER/WINTER
    # season_separations = [5, 10]
    # months = dict_ordinary["60"]["oe_time"].dt.month
    # winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
    # summer_inds = months.index[(months<season_separations[1]+1)&(months>season_separations[0]-1)]
    # T_winter = T[winter_inds]
    # T_summer = T[summer_inds]
    
    
    # g_phat_winter = S.temperature_model(T_winter,beta = 2)
    # g_phat_summer = S.temperature_model(T_summer,beta = 2)
    
    
    # winter_pdf = gen_norm_pdf(eT, g_phat_winter[0], g_phat_winter[1], 2)
    # summer_pdf = gen_norm_pdf(eT, g_phat_summer[0], g_phat_summer[1], 2)
    
    # combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))
    
    
    # #fig 3
    
    
    # TNX_FIG_temp_model(T=T_summer, g_phat=g_phat_summer,beta=2,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Summer')
    # TNX_FIG_temp_model(T=T_winter, g_phat=g_phat_winter,beta=2,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Winter')
    # TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,obscol='k',valcol='k',obslabel = None,vallabel = 'Annual',xlimits = [eT[0],eT[-1]],ylimits = [0,0.1])
    # plt.plot(eT,combined_pdf,'m',label = 'Combined summer and winter')
    # plt.title('fig 3')
    # plt.legend()
    # plt.show()
    
    
    # #TENAX MODEL VALIDATION
    # yrs = dict_ordinary["60"]["oe_time"].dt.year
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
    
    
    # F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr)
    # RL1, T_mc1, P_mc1 = S.model_inversion(F_phat1, g_phat1, n1, Ts,n_mc = np.size(P1)*S.niter_smev)
       
    # F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr)
    # RL2, T_mc2, P_mc2 = S.model_inversion(F_phat2, g_phat2, n2, Ts,n_mc = np.size(P2)*S.niter_smev)   
    
    # if F_phat[2]==0:
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
    # plt.ylim(0,np.max(AMS.AMS)+5)
    
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
    # plt.ylim(0,np.max(AMS.AMS)+5)
    
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
    # plt.ylim(0,np.max(AMS.AMS)+5)
    
    
    
    # plt.show()
    
    #TODO: n looks a litte different from in paper
    
    
    
