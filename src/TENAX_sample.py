# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:07:20 2024

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
import xarray as xr
import time
import pickle


country = 'Germany'
code_str = 'DE' 
n_stations = 5 #number of stations to sample
min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
max_yrs = 100 #if no max, set to very high
name_col = 'ppt'
temp_name_col = "t2m"



#READ IN META INFO FOR COUNTRY
comb = pd.read_csv('D:/metadata/'+country+'_fulldata.csv')
comb.station = comb['station'].apply(lambda x: f'{int(x):05}') #need to edit this according to file


#select stations
val_comb = comb[comb['cleaned_years']>=min_yrs] #filter out stations that are less than min
val_comb = val_comb[val_comb['cleaned_years']<=max_yrs] #filter out stations that are more than max 


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


for i in np.arange(0, n_stations):
    G[i] = pd.read_csv(files_sel[i], skiprows=21, names=[name_col])
    data_meta[i] = readIntense(files_sel[i], only_metadata=True, opened=False)


       
    #extract start and end dates from metadata
    start_date_G= dt.datetime.strptime(data_meta[i].start_datetime, "%Y%m%d%H")
    end_date_G= dt.datetime.strptime(data_meta[i].end_datetime, "%Y%m%d%H")
    
    time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G[i].size)] #make timelist of size of FI
    # replace -999 with nan
    G[i][G[i] == -999] = np.nan
    
    G[i]['prec_time'] = time_list_G
    G[i] = G[i].set_index('prec_time')


#READ IN ERA5 DATA
T_files = sorted(glob.glob('D:/ERA5_land/'+country+'*/*'))

#get selected lats and lons
lats_sel = [selected.latitude[i] for i in selected.index]
lons_sel = [selected.longitude[i] for i in selected.index]


#need to add in start and end dates here
start_time = time.time()
T_ERA = [0]*n_stations

for i in np.arange(0,n_stations): #for each selected station
    
    T_temp = [0]*np.size(T_files)
    for n in np.arange(0,np.size(T_files)): #select temperature data a gridpoint closest to station
        T_temp[n] = xr.open_dataarray(T_files[n]).sel(latitude = lats_sel[i],method = 'nearest').sel(longitude = lons_sel[i],method = 'nearest')
        
    T_ERA[i] = xr.concat(T_temp,dim = 'valid_time')  #combine multiple time files

print('time to read era5 '+str(time.time()-start_time))
print('Data reading finished.')

#NOW WE  DO TENAX
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


RL = [0]*n_stations
dicts = [0]*n_stations
thr = [0]*n_stations
F_phats = [0]*n_stations
g_phats = [0]*n_stations
ns = [0]*n_stations
dict_AMS = [0]*n_stations

for i in np.arange(0,n_stations):
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
    
    #TENAX MODEL HERE
    #magnitude model
    F_phats[i], loglik, _, _ = S.magnitude_model(P, T, thr[i])
    #temperature model
    g_phats[i] = S.temperature_model(T)
    # M is mean n of ordinary events
    ns[i] = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    #estimates return levels using MC samples
    RL[i], T_mc, P_mc = S.model_inversion(F_phats[i], g_phats[i], ns[i], Ts,n_mc = np.size(P)*S.niter_smev)
    print(RL[i])
    
    
   #PLOTTING THE GRAPHS
    titles = str(i)+': Latitude: '+str(lats_sel[i])+'. Longitude: '+str(lons_sel[i])
    
    
    eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    
    # fig 2a
    qs = [.85,.95,.99,.999]
    TNX_FIG_magn_model(P,T,F_phats[i],thr[i],eT,qs,xlimits = [eT[0],eT[-1]])
    plt.title(titles)
    plt.show()
    
    #fig 2b
    TNX_FIG_temp_model(T=T, g_phat=g_phats[i],beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
    plt.title(titles)
    plt.show()
    
    #fig 4 (without SMEV and uncertainty) 
    AMS = dict_AMS[i]['60'] # yet the annual maxima
    TNX_FIG_valid(AMS,S.return_period,RL[i],ylimits = [0,np.max(AMS.AMS)+3])
    plt.title(titles)
    plt.show()
    
    #fig 5 
    iTs = np.arange(-2.5,37.5,1.5) #idk why we need a different T range here 
    
    TNX_FIG_scaling(P,T,P_mc,T_mc,F_phats[i],S.niter_smev,eT,iTs,xlimits = [eT[0],eT[-1]])
    plt.title(titles)
    plt.show()
    
    #SPLITTING INTO SUMMER/WINTER
    season_separations = [5, 10]
    months = dicts[i]["60"]["oe_time"].dt.month
    winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
    summer_inds = months.index[(months<season_separations[1]+1)&(months>season_separations[0]-1)]
    T_winter = T[winter_inds]
    T_summer = T[summer_inds]
    
    
    g_phat_winter = S.temperature_model(T_winter, 2)
    g_phat_summer = S.temperature_model(T_summer, 2)
    
    
    winter_pdf = gen_norm_pdf(eT, g_phat_winter[0], g_phat_winter[1], 2)
    summer_pdf = gen_norm_pdf(eT, g_phat_summer[0], g_phat_summer[1], 2)
    
    combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))
    
    
    #fig 3
    
    
    TNX_FIG_temp_model(T=T_summer, g_phat=g_phat_summer,beta=2,eT=eT,obscol='r',valcol='r')
    TNX_FIG_temp_model(T=T_winter, g_phat=g_phat_winter,beta=2,eT=eT,obscol='b',valcol='b')
    TNX_FIG_temp_model(T=T, g_phat=g_phats[i],beta=4,eT=eT,obscol='k',valcol='k',xlimits = [eT[0],eT[-1]],ylimits = [0,0.1])
    plt.plot(eT,combined_pdf,'m',label = 'Combined summer and winter')
    plt.title(titles)
    plt.show()
    
    
    #TENAX MODEL VALIDATION
    yrs = dicts[i]["60"]["oe_time"].dt.year
    yrs_unique = np.unique(yrs)
    midway = yrs_unique[int(np.ceil(np.size(yrs_unique)/2))]
    
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
    
    
    F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr[i])
    RL1, T_mc1, P_mc1 = S.model_inversion(F_phat1, g_phat1, n1, Ts,n_mc = np.size(P1)*S.niter_smev)
       
    F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr[i])
    RL2, T_mc2, P_mc2 = S.model_inversion(F_phat2, g_phat2, n2, Ts,n_mc = np.size(P2)*S.niter_smev)   
    
    if F_phats[i][2]==0:
        dof=3
        alpha1=1; # b parameter is not significantly different from 0; 3 degrees of freedom for the LR test
    else: 
        dof=4
        alpha1=0  # b parameter is significantly different from 0; 4 degrees of freedom for the LR test
    
    
    
    
    #check magnitude model the same in both periods
    lambda_LR = -2*( loglik - (loglik1+loglik2) )
    pval = chi2.sf(lambda_LR, dof)
    
    #modelling second model based on first magnitude and changes in mean/std
    mu_delta = np.mean(T2)-np.mean(T1)
    sigma_factor = np.std(T2)/np.std(T1)
    
    g_phat2_predict = [g_phat1[0]+mu_delta, g_phat1[1]*sigma_factor]
    RL2_predict, _,_ = S.model_inversion(F_phat1,g_phat2_predict,n2,Ts)
    
    
    #fig 7a
    
    TNX_FIG_temp_model(T=T1, g_phat=g_phat1,beta=4,eT=eT,obscol='b',valcol='b',obslabel = None,vallabel = 'Temperature model '+str(yrs_unique[0])+'-'+str(midway),xlimits = [eT[0],eT[-1]])
    TNX_FIG_temp_model(T=T2, g_phat=g_phat2_predict,beta=4,eT=eT,obscol='r',valcol='r',obslabel = None,vallabel = 'Temperature model '+str(midway+1)+'-'+str(yrs_unique[-1]),xlimits = [eT[0],eT[-1]]) # model based on temp ave and std changes
    plt.title('fig 7a')
    plt.show() #this is slightly different in code and paper I think.. using predicted T vs fitted T
    
    #fig 7b
    
    TNX_FIG_valid(AMS1,S.return_period,RL1,TENAXcol='b',obscol_shape = 'b+',TENAXlabel = 'The TENAX model '+str(yrs_unique[0])+'-'+str(midway),obslabel='Observed annual maxima '+str(yrs_unique[0])+'-'+str(midway))
    TNX_FIG_valid(AMS2,S.return_period,RL2_predict,TENAXcol='r',obscol_shape = 'r+',TENAXlabel = 'The predicted TENAX model '+str(midway+1)+'-'+str(yrs_unique[-1]),obslabel='Observed annual maxima '+str(midway+1)+'-'+str(yrs_unique[-1]),ylimits = [0,np.max(AMS.AMS)+3])
    plt.title('fig 7b')
    plt.show()
    
    
    print('finished loop '+str(i+1)+' out of '+str(n_stations))
    
    
    
    
#SAVE EXTRACTED DATA
# for i in np.arange(0,n_stations):
#     file_save = 'D:/sample_outputs/'+country+'_'+selected['station'][selected.index[i]]+'_dict'


    
