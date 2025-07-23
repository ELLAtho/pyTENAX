# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:09:47 2025

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


country = 'Germany' 
ERA_country = 'Germany'
country_save = 'Germany'
code_str = 'DE_'
minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9


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
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet


# country = 'UK' 
# ERA_country = 'UK'
# country_save = 'UK_b0'
# code_str = 'UK_'
# name_len = 0
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# alpha_set = 1
# remake = 0

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

# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main_b0'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# alpha_set = 1
# remake = 0


name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 


info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)
val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min
val_info = val_info[val_info['startdate']>=min_startdate]

if 'minlat' in locals():
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
else:
    pass

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


df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 
TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters

if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg)

    #dataframe with all values
    new_df = df_parameters[['latitude','longitude','b','kappa','lambda','a']].copy()
    mask = new_df['b'] == 0
    
    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
    new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
    new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()

else:
    new_df = df_parameters.copy()



if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg, dtype={'station': str})

    #dataframe with all values
    new_df = df_parameters[['station','latitude','longitude','b','kappa','lambda','a','thr','mu','sigma','n_events_per_yr']].copy()
    
    mask = new_df['b'] == 0
    
    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
    new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
    new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()
    

else:
    new_df = df_parameters.copy()

#merging the dataframes to ensure station consistency
missing_rows = pd.merge(df_parameters.station, val_info.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    new_df = new_df.drop(missing_rows.index)
else:
    pass


# LOOKING AT DISTRIBUTION OF b
###############################################################################
#Fit observed F_hat values to normal distribution

kappa_mu_sigma = norm.fit(new_df.kappa.copy().dropna())
b_mu_sigma = norm.fit(new_df.b.copy().dropna())
lambda_mu_sigma = norm.fit(new_df['lambda'].copy().dropna())
a_mu_sigma = norm.fit(new_df.a.copy().dropna())


# get mean of mu and sigma for temp model
mu_mu_sigma = norm.fit(df_parameters.mu.copy().dropna())
sigma_mu_sigma = norm.fit(df_parameters.sigma.copy().dropna())



#define mean F_phat and g_phat... using the normal distribution
F_phat = np.array([kappa_mu_sigma[0],b_mu_sigma[0],lambda_mu_sigma[0],a_mu_sigma[0]])
g_phat = np.array([mu_mu_sigma[0],sigma_mu_sigma[0]])

print(F_phat)
###############################################################################
df_gen_savename = f"D:/outputs/{country_save}\\synth_generated_parameters_one2one.csv"
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
    
    
    Ts = np.arange(mu_mu_sigma[0]-2*sigma_mu_sigma[0] - S.temp_delta, mu_mu_sigma[0]+2*sigma_mu_sigma[0] + S.temp_delta, S.temp_res_monte_carlo)

    
    df_list = [0]*5
    for j in range(5):
        # define empty arrays
        thr_gen = [np.nan]*n_stations
        F_phat_gen = [np.array([np.nan,np.nan,np.nan,np.nan])]*n_stations
        g_phat_gen = [np.array([np.nan,np.nan])]*n_stations
        start_time = [np.nan]*n_stations
        
        print(f"round {j+1} of 5")
        # model inversion loop
    
        
        for i in np.arange(0,n_stations):
            start_time[i] = time.time()
            #generate T and P
            if not np.isnan(total_events[i]):
                
                S.n_monte_carlo = int(total_events[i]) # set the number of events to the same for each station
                n = df_parameters.n_events_per_yr.iloc[i]
                
                
                _, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts, gen_P_mc = True,gen_RL=False) 
                T_mc = T_mc.reshape(-1)
                
                #recalculate g_phat and F_phat
                thr_gen[i] = np.nanquantile(P_mc,S.left_censoring[1])
                
                #magnitude model
                F_phat_gen[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen[i])
                #temperature model
                g_phat_gen[i] = S.temperature_model(T_mc)
            else:
                pass
                
            if (i+1)%50 == 0:
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (n_stations-i)*time_taken/60
                print(f"{i}/{n_stations}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
            else:
                pass
        df_list[j] = pd.DataFrame({f'mu{j}':np.array(g_phat_gen)[:,0],f'sigma{j}':np.array(g_phat_gen)[:,1],f'kappa{j}':np.array(F_phat_gen)[:,0],f'b{j}':np.array(F_phat_gen)[:,1],f'lambda{j}':np.array(F_phat_gen)[:,2],f'a{j}':np.array(F_phat_gen)[:,3],f'thr{j}':np.array(thr_gen)})
        
        
    df_generated_parameters_one2one = pd.concat(df_list, axis = 1)
    df_generated_parameters_one2one.to_csv(df_gen_savename) #save calculated parameters

else:
    print('file made already')
    df_generated_parameters_one2one  = pd.read_csv(df_gen_savename) #load calculated parameters

df_generated_parameters =  pd.read_csv(f"D:/outputs/{country_save}\\synth_generated_parameters.csv")

###############################################################################
## exponential version

df_gen_savename_exp = f"D:/outputs/{country_save}\\synth_generated_parameters_one2one_exp.csv"


if df_gen_savename_exp not in saved_output_files:
    # number of stations and average number events
    
    S = TENAX(
            return_period = [2,5,10,20,50,100, 200],  #for some reason it doesnt like calculating RP =<1
            durations = [60, 180],
            left_censoring = [0, 0.90],
            alpha = alpha_set,
            n_monte_carlo = round(total_events_mean),
            
        )
    
    
    Ts = np.arange(mu_mu_sigma[0]-2*sigma_mu_sigma[0] - S.temp_delta, mu_mu_sigma[0]+2*sigma_mu_sigma[0] + S.temp_delta, S.temp_res_monte_carlo)

    
    df_list_exp = [0]*5
    for j in range(5):
        # define empty arrays
        thr_gen_exp = [np.nan]*n_stations
        F_phat_gen_exp = [np.array([np.nan,np.nan,np.nan,np.nan])]*n_stations
        g_phat_gen_exp = [np.array([np.nan,np.nan])]*n_stations
        start_time = [np.nan]*n_stations
        
        print(f"round {j+1} of 5")
        # model inversion loop
    
        
        for i in np.arange(0,n_stations):
            start_time[i] = time.time()
            #generate T and P
            if not np.isnan(total_events[i]):
                
                S.n_monte_carlo = int(total_events[i]) # set the number of events to the same for each station
                n = df_parameters.n_events_per_yr.iloc[i]
                
                
                _, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts, gen_P_mc = True,gen_RL=False,b_exp = True) 
                T_mc = T_mc.reshape(-1)
                
                #recalculate g_phat and F_phat
                thr_gen_exp[i] = np.nanquantile(P_mc,S.left_censoring[1])
                
                #magnitude model
                F_phat_gen_exp[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen_exp[i],b_exp = True)
                #temperature model
                g_phat_gen_exp[i] = S.temperature_model(T_mc)
            else:
                pass
                
            if (i+1)%50 == 0:
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (n_stations-i)*time_taken/60
                print(f"{i}/{n_stations}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
            else:
                pass
        df_list_exp[j] = pd.DataFrame({f'mu{j}':np.array(g_phat_gen_exp)[:,0],f'sigma{j}':np.array(g_phat_gen_exp)[:,1],f'kappa{j}':np.array(F_phat_gen_exp)[:,0],f'b{j}':np.array(F_phat_gen_exp)[:,1],f'lambda{j}':np.array(F_phat_gen_exp)[:,2],f'a{j}':np.array(F_phat_gen_exp)[:,3],f'thr{j}':np.array(thr_gen_exp)})
        
        
    df_generated_parameters_one2one_exp = pd.concat(df_list_exp, axis = 1)
    df_generated_parameters_one2one_exp.to_csv(df_gen_savename_exp) #save calculated parameters

else:
    print('file made already')
    df_generated_parameters_one2one_exp  = pd.read_csv(df_gen_savename_exp) #load calculated parameters



###############################################################################

comb_df = pd.DataFrame()

params = ["b", "kappa", "lambda", "a"]
for param in params:
    comb_df[param] = pd.concat([df_generated_parameters_one2one[f"{param}{j}"].copy().dropna() for j in range(5)])
    
comb_df.reset_index(inplace = True)


# b 

violin_list = [new_df.b.copy().dropna()]+[df_generated_parameters_one2one[f"b{j}"].copy().dropna() for j in range(5)]+[df_generated_parameters.b] + [comb_df.b.copy().dropna()]
IQR = [violin_list[j].quantile(0.75) - violin_list[j].quantile(0.25) for j in range(len(violin_list))]
violin_list_labels = [f'b observed. IQR: {IQR[0]}']+[f" MC gen 1 to 1 {j} IQR: {IQR[j+1]}" for j in range(5)]+[f'Monte Carlo generated samples old version. IQR: {IQR[6]}',f'all 5 combined. IQR: {IQR[7]}']



plt.violinplot(violin_list,vert=False)
plt.xlabel('b')
plt.yticks([1,2,3,4,5,6,7,8],violin_list_labels)
plt.title(f'{ERA_country} b')
plt.show()


#KAPPA


violin_list = [new_df.kappa.copy().dropna()]+[df_generated_parameters_one2one[f"kappa{j}"].copy().dropna() for j in range(5)]+[df_generated_parameters.kappa] 
IQR = [violin_list[j].quantile(0.75) - violin_list[j].quantile(0.25) for j in range(len(violin_list))]
violin_list_labels = [f'kappa observed. IQR: {IQR[0]}']+[f" MC gen 1 to 1 {j} IQR: {IQR[j+1]}" for j in range(5)]+[f'Monte Carlo generated samples old version. IQR: {IQR[6]}']



plt.violinplot(violin_list,vert=False)
plt.xlabel('kappa')
plt.yticks([1,2,3,4,5,6,7],violin_list_labels)
plt.title(f'{ERA_country} kappa')
plt.show()


# LAMBDA

violin_list = [new_df["lambda"].copy().dropna()]+[df_generated_parameters_one2one[f"lambda{j}"].copy().dropna() for j in range(5)]+[df_generated_parameters["lambda"]] 
IQR = [violin_list[j].quantile(0.75) - violin_list[j].quantile(0.25) for j in range(len(violin_list))]
violin_list_labels = [f'lambda observed. IQR: {IQR[0]}']+[f" MC gen 1 to 1 {j} IQR: {IQR[j+1]}" for j in range(5)]+[f'Monte Carlo generated samples old version. IQR: {IQR[6]}']



plt.violinplot(violin_list,vert=False)
plt.xlabel('lambda')
plt.yticks([1,2,3,4,5,6,7],violin_list_labels)
plt.title(f'{ERA_country} lambda')
plt.show()



# a

violin_list = [new_df["a"].copy().dropna()]+[df_generated_parameters_one2one[f"a{j}"].copy().dropna() for j in range(5)]+[df_generated_parameters["a"]] + [df_generated_parameters_one2one[[f"a{n}" for n in range(5)]].mean(axis = 1).copy().dropna()]
IQR = [violin_list[j].quantile(0.75) - violin_list[j].quantile(0.25) for j in range(len(violin_list))]
violin_list_labels = [f'a observed. IQR: {IQR[0]}']+[f" MC gen 1 to 1 {j} IQR: {IQR[j+1]}" for j in range(5)]+[f'Monte Carlo generated samples old version. IQR: {IQR[6]}']+["aaverage of the 5"]



plt.violinplot(violin_list,vert=False)
plt.xlabel('a')
plt.yticks([1,2,3,4,5,6,7,8],violin_list_labels)
plt.title(f'{ERA_country} a')
plt.show()

comb_df = pd.DataFrame()

params = ["b", "kappa", "lambda", "a"]
for param in params:
    comb_df[param] = pd.concat([df_generated_parameters_one2one[f"{param}{j}"].copy().dropna() for j in range(5)])
    
comb_df.reset_index(inplace = True)


