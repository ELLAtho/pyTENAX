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
import time

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

from scipy.stats import norm

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob

drive='D' #name of drive
alpha_set = 0
remake = 1

# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9

# country = 'UK' 
# ERA_country = 'UK'
# country_save = 'UK'
# code_str = 'UK_'
# name_len = 0
# min_startdate = dt.datetime(1981,1,1) #this is for if havent read all ERA5 data yet


# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


# country = 'Belgium'
# ERA_country = 'Germany' #country where the era files are
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# country_save = 'Belgium'
# code_str = 'BE_'
# name_len = 8 #how long the numbers are at the end of the files
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9

# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany_b0'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# alpha_set = 1
# remake = 0

# country = 'Japan' 
# ERA_country = 'Japan'
# country_save = 'Japan_b0'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# alpha_set = 1
# remake = 0

country = 'US' 
ERA_country = 'US'
country_save = 'US_main_b0'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
alpha_set = 1
remake = 0


name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 


info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)
val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min
val_info = val_info[val_info['startdate']>=min_startdate]

val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
val_info = val_info[val_info['latitude']<=maxlat]
val_info = val_info[val_info['longitude']>=minlon]
val_info = val_info[val_info['longitude']<=maxlon]


# files = glob.glob(drive+':/'+country+'/*') #list of files in country folder
# files_sel = files[0]
# G,data_meta = read_GSDR_file(files_sel,name_col)
# code = files_sel[-4-5:-4]
# T_path = drive + ':/'+country+'_temp\\'+code_str+code + '.nc'
# T_ERA = xr.load_dataarray(T_path)
# t_data = (T_ERA.squeeze()-273.15).to_dataframe()
#TODO: make code selection generic


## READ IN FILES
save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename) 
TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters

if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg)

    #dataframe with all values
    new_df = df_parameters[['latitude','longitude','b']].copy()
    mask = new_df['b'] == 0
    
    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
    new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
    new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()

else:
    new_df = df_parameters.copy()


# LOOKING AT DISTRIBUTION OF b
###############################################################################
#Fit observed F_hat values to normal distribution

kappa_mu_sigma = norm.fit(new_df.kappa.copy().dropna())
b_mu_sigma = norm.fit(new_df.b.copy().dropna())
lambda_mu_sigma = norm.fit(new_df['lambda'].copy().dropna())
a_mu_sigma = norm.fit(new_df.a.copy().dropna())


# #now with beta = 4
# kappa_mu_sigma_4 = normal_model(new_df.kappa.copy().dropna(), 4)
# b_mu_sigma_4 = normal_model(new_df.b.copy().dropna(), 4)
# lambda_mu_sigma_4 = normal_model(new_df['lambda'].copy().dropna(), 4)
# a_mu_sigma_4 = normal_model(new_df.a.copy().dropna(), 4)

# #now with skew
# kappa_skew = normal_model(new_df.kappa.copy().dropna(), method = 'skewnorm')
# b_skew = normal_model(new_df.b.copy().dropna(), method = 'skewnorm')
# lambda_skew = normal_model(new_df['lambda'].copy().dropna(), method = 'skewnorm')
# a__skew = normal_model(new_df.a.copy().dropna(), method = 'skewnorm')



#plot distribution of values b
bins = np.arange(np.min(new_df.b),np.max(new_df.b)+0.005,0.005)
if np.size(bins) > 1:
    bin_edge = np.concatenate([np.array([bins[0]-(bins[1]-bins[0])/2]),(bins + (bins[1]-bins[0])/2)]) #convert bin centres into bin edges
    hist, bin_edges = np.histogram(new_df.b, bins=bin_edge, density=True)
    plt.plot(bins, hist, '--', color='b',label = 'observed values of b')
    
    # Plot analytical PDF of b (validation)
    plt.plot(bins, gen_norm_pdf(bins, b_mu_sigma[0], b_mu_sigma[1], 2), '-', color='r', label='fitted b')
    
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
    
    plt.plot(bins, gen_norm_pdf(bins, b_mu_sigma_4[0], b_mu_sigma_4[1], 4), '-', color='r', label='fitted b, beta = 4')
    
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
                           obslabel = 'observed b values',
                           vallabel = f'skewnorm fit, loc = {b_skew[1]:.2} \n scale = {b_skew[2]:.2}, skew = {b_skew[0]:.2}',
                           xlimits = [-0.08,0.03],
                           ylimits = [0,30], 
                           method = "skewnorm") 
    plt.show()
    
else:
    pass
    


# get mean of mu and sigma for temp model
mu_mu_sigma = norm.fit(df_parameters.mu.copy().dropna())
sigma_mu_sigma = norm.fit(df_parameters.sigma.copy().dropna())



#define mean F_phat and g_phat... using the normal distribution
F_phat = np.array([kappa_mu_sigma[0],b_mu_sigma[0],lambda_mu_sigma[0],a_mu_sigma[0]])
g_phat = np.array([mu_mu_sigma[0],sigma_mu_sigma[0]])

print(F_phat)
###############################################################################
df_gen_savename = drive + ':/outputs/'+country_save+'\\synth_generated_parameters.csv'
saved_output_files = glob.glob(drive + ':/outputs/'+country_save+'/*')
total_events = df_parameters.n_events_per_yr.to_numpy() * val_info.cleaned_years.to_numpy()
total_events_mean = np.nanmean(total_events) #average total events for each station to. do this many monte carl samples

n_stations = np.size(df_parameters.mu) #how many resamples we need to do

if df_gen_savename not in saved_output_files:
    # number of stations and average number events
    
    S = TENAX(
            return_period = [2,5,10,20,50,100, 200],  #for some reason it doesnt like calculating RP =<1
            durations = [60, 180],
            left_censoring = [0, 0.90],
            alpha = alpha_set,
            n_monte_carlo = round(total_events_mean),
            
        )
    
    # data = G 
    # data = S.remove_incomplete_years(data, name_col)
    # df_arr = np.array(data[name_col])
    # df_dates = np.array(data.index)
    
    # idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
        
    
    # #get ordinary events by removing too short events
    # #returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
    # arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
    
    # #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
    # dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
    # df_arr_t_data = np.array(t_data[temp_name_col])
    # df_dates_t_data = np.array(t_data.index)
    # dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
    # # Your data (P, T arrays) and threshold thr=3.8
    # P = dict_ordinary["60"]["ordinary"].to_numpy() 
    # T = dict_ordinary["60"]["T"].to_numpy()  
    
    n = np.mean(df_parameters.n_events_per_yr) #average events per year
    Ts = np.arange(mu_mu_sigma[0]-2*sigma_mu_sigma[0] - S.temp_delta, mu_mu_sigma[0]+2*sigma_mu_sigma[0] + S.temp_delta, S.temp_res_monte_carlo)

    
    
    # define empty arrays
    thr_gen = np.zeros(n_stations)
    F_phat_gen = [0]*n_stations
    g_phat_gen = [0]*n_stations
    start_time = [0]*n_stations
    
    
    # model inversion loop

    
    for i in np.arange(0,n_stations):
        start_time[i] = time.time()
        #generate T and P
        _, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts, gen_P_mc = True,gen_RL=False) 
        T_mc = T_mc.reshape(-1)
        
        #recalculate g_phat and F_phat
        thr_gen[i] = np.nanquantile(P_mc,S.left_censoring[1])
        
        #magnitude model
        F_phat_gen[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen[i])
        #temperature model
        g_phat_gen[i] = S.temperature_model(T_mc)
        
        if (i+1)%50 == 0:
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (n_stations-i)*time_taken/60
            print(f"{i}/{n_stations}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
    df_generated_parameters = pd.DataFrame({'mu':np.array(g_phat_gen)[:,0],'sigma':np.array(g_phat_gen)[:,1],'kappa':np.array(F_phat_gen)[:,0],'b':np.array(F_phat_gen)[:,1],'lambda':np.array(F_phat_gen)[:,2],'a':np.array(F_phat_gen)[:,3],'thr':np.array(thr_gen)})
    df_generated_parameters.to_csv(df_gen_savename) #save calculated parameters

else:
    print('file made already')
    df_generated_parameters = pd.read_csv(df_gen_savename) #save calculated parameters

    
sd_obs = np.std(new_df.b)
sd_gen = np.std(df_generated_parameters.b)


kappa_mu_sigma_gen = norm.fit(df_generated_parameters.kappa.copy().dropna()) #norm fit to generated ones (this seems to give the same as just calculating the mean with np.mean)
b_mu_sigma_gen = norm.fit(df_generated_parameters.b.copy().dropna())
lambda_mu_sigma_gen = norm.fit(df_generated_parameters['lambda'].copy().dropna())
a_mu_sigma_gen = norm.fit(df_generated_parameters.a.copy().dropna())


F_phat_gen_mean = np.array([kappa_mu_sigma_gen[0],b_mu_sigma_gen[0],lambda_mu_sigma_gen[0],a_mu_sigma_gen[0]])

if sd_obs != 0:
    ratio = 100*sd_gen/sd_obs
    print(f'Ratio of generated spread to observed spread in b: {ratio:.2f}%')

else:
    ratio = 'b is 0 so inf'



print(f'average observed F_phat: {F_phat}')
print(f'average generated F_phat: {F_phat_gen_mean}')


ratio_a = 100*np.std(df_generated_parameters.a)/np.std(new_df.a)

ratio_kappa = 100*np.std(df_generated_parameters.kappa)/np.std(new_df.kappa)

ratio_lambda = 100*np.std(df_generated_parameters['lambda'])/np.std(new_df['lambda'])

print(f'Ratio a: {ratio_a:.2f}% \n kappa: {ratio_kappa:.2f}%\n lambda {ratio_lambda:.2f}% ')


if remake == 1: #this is for if we need to plot 3 violins 
    plt.violinplot([new_df.b.copy().dropna(),df_parameters.b.copy().dropna(),df_generated_parameters.b],vert=False)
    plt.xlabel('b')
    if sd_obs != 0:
        plt.yticks([1,2,3],['b allowed to be non sig','b forced to zero if not sig',f'Monte Carlo generated samples. \n sd ratio = {ratio:.2f}%'])
    else:
        plt.yticks([1,2,3],['b allowed to be non sig','b forced to zero if not sig','Monte Carlo generated samples.'])
    plt.title(f'{ERA_country} b')
    plt.show()
    
    plt.violinplot([new_df.a.copy().dropna(),df_parameters.a.copy().dropna(),df_generated_parameters.a],vert=False)
    plt.xlabel('a')
    plt.yticks([1,2,3],['a allowed to be non sig','a when b forced to zero if non sig',f'Monte Carlo generated samples. \n sd ratio = {ratio_a:.2f}%'])
    plt.title(f'{ERA_country} a')
    plt.show()
    
    plt.violinplot([new_df.kappa.copy().dropna(),df_parameters.kappa.copy().dropna(),df_generated_parameters.kappa],vert=False)
    plt.xlabel('kappa')
    plt.yticks([1,2,3],['kappa allowed to be non sig','kappa when b forced to zero if non sig',f'Monte Carlo generated samples. \n sd ratio = {ratio_kappa:.2f}%'])
    plt.title(f'{ERA_country} kappa_0')
    plt.show()
    
    plt.violinplot([new_df['lambda'].copy().dropna(),df_parameters['lambda'].copy().dropna(),df_generated_parameters['lambda']],vert=False)
    plt.xlabel('lambda')
    plt.yticks([1,2,3],['lambda allowed to be non sig','lambda when b forced to zero if non sig',f'Monte Carlo generated samples. \n sd ratio = {ratio_lambda:.2f}%'])
    plt.title(f'{ERA_country} lambda_0')
    plt.show()
    
else:
    plt.violinplot([df_parameters.b.copy().dropna(),df_generated_parameters.b],vert=False)
    plt.xlabel('b')
    if sd_obs != 0:
        plt.yticks([1,2],['b observed',f'Monte Carlo generated samples. \n sd ratio = {ratio:.2f}%'])
    else:
        plt.yticks([1,2],['b observed','Monte Carlo generated samples.'])
    plt.title(f'{ERA_country} b')
    plt.show()
    
    plt.violinplot([df_parameters.a.copy().dropna(),df_generated_parameters.a],vert=False)
    plt.xlabel('a')
    plt.yticks([1,2],['a observed',f'Monte Carlo generated samples. \n sd ratio = {ratio_a:.2f}%'])
    plt.title(f'{ERA_country} a')
    plt.show()
    
    plt.violinplot([df_parameters.kappa.copy().dropna(),df_generated_parameters.kappa],vert=False)
    plt.xlabel('kappa')
    plt.yticks([1,2],['kappa observed',f'Monte Carlo generated samples. \n sd ratio = {ratio_kappa:.2f}%'])
    plt.title(f'{ERA_country} kappa_0')
    plt.show()
    
    plt.violinplot([df_parameters['lambda'].copy().dropna(),df_generated_parameters['lambda']],vert=False)
    plt.xlabel('lambda')
    plt.yticks([1,2],['lambda observed',f'Monte Carlo generated samples. \n sd ratio = {ratio_lambda:.2f}%'])
    plt.title(f'{ERA_country} lambda_0')
    plt.show()