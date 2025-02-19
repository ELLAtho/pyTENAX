# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:50:10 2025

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

country = 'Germany'
country_save = 'Germany'
code_str = 'DE' 
n_stations = 2 #number of stations to sample
min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
max_yrs = 1000 #if no max, set to very high
name_col = 'ppt'
temp_name_col = "t2m"



#READ IN META INFO FOR COUNTRY
comb = pd.read_csv('D:/metadata/'+country+'_fulldata.csv', dtype={'station': str})

comb.startdate = pd.to_datetime(comb.startdate)
comb.enddate = pd.to_datetime(comb.enddate)


#select stations
val_comb = comb[comb['cleaned_years']>=min_yrs] #filter out stations that are less than min
val_comb = val_comb[val_comb['cleaned_years']<=max_yrs] #filter out stations that are more than max 


#TODO: add in lat and lon conditions to choose from regions. also add plots of region

comb_sort = val_comb.sort_values(by=['cleaned_years'],ascending=0) #sort by size so can choose top sizes
max_stations = comb_sort[comb_sort.cleaned_years == comb_sort.cleaned_years.max()]


# if there are enough stations with the maximum cleaned years, choose from these with a spread so they aren't all in the same place
if len(max_stations)>=n_stations:
    average_index_difference = np.trunc(len(max_stations)/n_stations) #to select stations rainging throught the max ones
    select_indices = np.arange(0,n_stations*average_index_difference,average_index_difference)
    
    selected = max_stations.iloc[select_indices]
else:
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
T_ERA = [0]*n_stations


start_time = time.time()
for i in np.arange(0, n_stations):
    G[i], data_meta[i] =  read_GSDR_file(files_sel[i],name_col)
    T_path = drive + ':/'+country+'_temp\\'+code_str+'_'+str(selected.station[selected.index[i]]) + '.nc'
    T_ERA[i] = xr.load_dataarray(T_path)
    

print('time to read temp and ppt '+str(time.time()-start_time))



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

# set percentage to get temp GOF 
GOF_perc = 0.8

dicts = [0]*n_stations
thr = [0]*n_stations
g_phats = [0]*n_stations
ns = [0]*n_stations
dict_AMS = [0]*n_stations
eRP = [0]*n_stations
diff = [0]*n_stations
AD = [0]*n_stations
FRMSE_upper_perc = [0]*n_stations
upper_perc_length = [0]*n_stations

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
    
    AMS = dict_AMS[i]['60']
    AMS_sort = AMS.sort_values(by=['AMS'])['AMS']
    plot_pos = np.arange(1,np.size(AMS_sort)+1)/(1+np.size(AMS_sort))
    eRP[i] = 1/(1-plot_pos)
    S.return_period = eRP[i]
    
    #TENAX MODEL HERE
    #temperature model
    g_phats[i] = S.temperature_model(T)
    # M is mean n of ordinary events
    ns[i] = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
    #estimates return levels using MC samples
    
    
    min_T_upper = np.quantile(T,GOF_perc)
    
    
   #PLOTTING THE GRAPHS
    titles = str(i)+': Latitude: '+str(lats_sel[i])+'. Longitude: '+str(lons_sel[i])
    
    
    eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end
    ylim_perc = [-100,100]
       
    #fig 2b
    
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 2.5]})
    
    hist, pdf_values = TNX_FIG_temp_model(T=T, g_phat=g_phats[i],beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
    kde  = gaussian_kde(T) #use kernel density to get probability
    prob = kde(eT)
    
    diff[i] = pdf_values - prob
    RMSE = np.sqrt(np.sum(diff[i]**2)/len(diff[i]))
    FRMSE = np.sqrt(
        np.sum(diff[i]**2)/len(diff[i]))/(np.sum(hist)/len(diff[i]))
    
    
    eT_upper_perc = eT[eT>=min_T_upper]
    
    hist_upper_perc = hist[eT>=min_T_upper]
    pdf_values_upper_perc = pdf_values[eT>=min_T_upper]
    diff_upper_perc = pdf_values_upper_perc - hist_upper_perc
    FRMSE_upper_perc1 = np.sqrt(
        np.sum(diff_upper_perc**2)/len(diff_upper_perc))/(np.sum(hist_upper_perc)/len(diff_upper_perc))
    
    
    ax1.plot([min_T_upper]*2,[-0.01,0.07],color = 'k', alpha = 0.4)
    ax1.set_title(f"{len(eT)} bins. FRMSE = {FRMSE:.3f}. FRMSE on top {(1-GOF_perc)*100:.0f}% data: {FRMSE_upper_perc1:.3f}")
   
    
    
    ###########################################################################
    # Anderson-Darling GOF statistic
    pdf_cum, hist_cum = np.cumsum(pdf_values), np.cumsum(hist)
    T_order = np.sort(T)
    
    numb = np.zeros(len(T))
    for ind in np.arange(1,len(T)+1):
        numb[ind-1] = (2 * ind - 1)*(np.log(gen_norm_cdf(T_order[ind-1],g_phats[i][0],g_phats[i][1],4)) 
                                     + np.log(1 - (gen_norm_cdf(T_order[len(T) - ind],g_phats[i][0],g_phats[i][1],4))))
    AD[i] = -len(T) - (1/len(T)) * np.nansum(numb)
    
    
    ax2.plot(eT,diff[i])
    ax2.plot(eT,[0]*eT,'--',alpha = 0.5, color = 'k')
    ax2.set_xlim(eT[0],eT[-1])
    ax2.tick_params(labelbottom=False, bottom=False)
    ax2.tick_params(labelsize=14)
    
    ax2.set_title(titles+'\n absolute difference in observations and model')
    plt.show()
    
    coeffs=np.polyfit(hist,pdf_values,1)
    delt = (np.max(hist)-np.min(hist))/10
    x = np.arange(np.min(hist),np.max(hist)+delt,delt)
    y = coeffs[0]*x+coeffs[1]
    
    plt.scatter(hist,pdf_values)
    plt.plot(np.arange(min(hist),max(hist),0.005),np.arange(min(hist),max(hist),0.005),'--',color = 'k',alpha = 0.4,label = 'Equality line')
    plt.plot(x,y,label = 'fit')
    plt.xlabel('Observed')
    plt.ylabel('Normal fit')
    plt.legend()
    plt.show()
    
    ###########################################################################
    # Error at upper end of temp distribution THIS IS WHERE BINS MATTER
    
    
    FRMSE_upper_perc[i] = [0]*6
    upper_perc_length[i] = [0]*6
    fig = plt.figure(figsize = (15,10))
    
    for bins_check_itn in np.arange(0,6):
        n_bins = bins_check_itn * 10 + 20 # checking 20,30,40,50,60,70 total bins
        eT_diff = (eT[-1]-eT[0])/(n_bins-1) # size of individual bin
        
        eT_check = np.arange(eT[0],eT[-1]+eT_diff,eT_diff) #new eT
        
        ax = fig.add_subplot(2,3,bins_check_itn + 1)
        
        hist_check, pdf_values_check = TNX_FIG_temp_model(
            T=T,
            g_phat=g_phats[i],beta=4,eT=eT_check,xlimits = [eT[0],eT[-1]])
        
        diff_check = pdf_values_check - hist_check
        
        eT_upper_perc = eT_check[eT_check>=min_T_upper]
        upper_perc_length[i][bins_check_itn] = len(eT_upper_perc)
        
        hist_upper_perc = hist_check[eT_check>=min_T_upper]
        pdf_values_upper_perc = pdf_values_check[eT_check>=min_T_upper]
        diff_upper_perc = pdf_values_upper_perc - hist_upper_perc
        
        FRMSE_check = np.sqrt(
            np.sum(diff_check**2)/len(diff_check))/(np.sum(hist)/len(diff_check))
        
        FRMSE_upper_perc[i][bins_check_itn] = np.sqrt(
            np.sum(diff_upper_perc**2)/len(diff_upper_perc))/(np.sum(hist_upper_perc)/len(diff_upper_perc))
        
        ax.set_title(f"{n_bins} bins. FRMSE = {FRMSE_check:.3f} \n FRMSE on top {(1-GOF_perc)*100:.0f}% data: {FRMSE_upper_perc[i][bins_check_itn]:.3f}")
    fig.tight_layout()
    plt.show()
    
    ###########################################################################

#Temp monte carlo uncertainty
i=0
percentages = [0.05,0.95]
n_itn = 1000
hist_gen = [0]*n_stations


for i in np.arange(0,n_stations):

    titles = str(i)+': Latitude: '+str(lats_sel[i])+'. Longitude: '+str(lons_sel[i])
    T = dicts[i]["60"]["T"].to_numpy()
    n_events = len(T)
    S.n_monte_carlo = n_events
    
    Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)
    pdf_values = gen_norm_pdf(Ts, g_phats[i][0], g_phats[i][1], S.beta)
    df = np.vstack([pdf_values, Ts])
    
    start_time = time.time()
    hist_gen[i] = [0]*n_itn
    
    for mc_i in np.arange(0,n_itn):
        _, T_mc, _ = S.model_inversion([1,0,3,0], g_phats[i], ns[i], Ts,gen_P_mc = False,gen_RL=False) 
        T_mc = T_mc.reshape(-1)
        eT_edges = np.concatenate([np.array([eT[0]-(eT[1]-eT[0])/2]),(eT + (eT[1]-eT[0])/2)]) #convert bin centres into bin edges
        hist_gen[i][mc_i], bin_edges_gen = np.histogram(T_mc, bins=eT_edges, density=True)
    
    hist_gen[i] = np.array(hist_gen[i])
    time_to_MC = (time.time() - start_time)/60
    print(f"{time_to_MC:.1f} mins")
    
    mins = [np.quantile(hist_gen[i][:,j],percentages[0]) for j in np.arange(0,len(eT))]
    maxes = [np.quantile(hist_gen[i][:,j],percentages[1]) for j in np.arange(0,len(eT))]
    
    
    hist, pdf_values = TNX_FIG_temp_model(T=T, g_phat=g_phats[i],beta=4,eT=eT,xlimits = [eT[0],eT[-1]])
    plt.fill_between(eT,mins,maxes,color = 'b',alpha = 0.3,label = f'Monte carlo generated {(percentages[1]-percentages[0])*100:.0f}% spread')
    plt.legend()
    plt.title(titles)
    plt.show()









