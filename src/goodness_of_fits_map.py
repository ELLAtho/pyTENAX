# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:12:29 2025

@author: ellar
"""


from os.path import dirname, join
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
from scipy.stats import gaussian_kde

import datetime as dt
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

import xarray as xr
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.interpolate import interp1d
from matplotlib import cm



drive = 'D'
alpha_set = 0
beta_set = "" # for beta = 4, use ""
if beta_set == "":
    beta_set2 = 4
else:
    beta_set2 = beta_set



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

country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9



name_col = 'ppt' 
temp_name_col = "t2m"
min_yrs = 10 

#READ IN META INFO FOR COUNTRY
info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})


# shouldn't need this anymore as changed files
# if name_len!=0:
#     info.station = info['station'].apply(lambda x: f'{int(x):0{name_len}}') #need to edit this according to file
# else:
#     pass

info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)

#select stations


val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = alpha_set,
        min_ev_dur = 60,
        niter_smev = 1000, 
        beta = beta_set2
    )

#getting info of correct size
if 'min_startdate' in locals():    
    val_info = val_info[val_info['startdate']>=min_startdate]
else:
    pass

if 'minlat' in locals():
    
    val_info = val_info[val_info['latitude']>=minlat] #filter station locations to within ERA bounds
    val_info = val_info[val_info['latitude']<=maxlat]
    val_info = val_info[val_info['longitude']>=minlon]
    val_info = val_info[val_info['longitude']<=maxlon]
    
else:
    pass



files = glob.glob(drive+':/'+country+'/*') #list of files in country folder
files_sel = [files[i] for i in val_info.index]


## READ IN FILES
save_path_neg = drive + ':/outputs/'+country_save+'\\parameters_neg.csv'
df_savename = drive + ':/outputs/'+country_save+'\\parameters.csv'


df_parameters = pd.read_csv(df_savename, dtype={'station': str}) 
TENAX_use = pd.read_csv(drive + ':/outputs/'+country_save+'/TENAX_parameters.csv') #save calculated parameters

df_parameters_0 = pd.read_csv(f"{drive}:/outputs/{country_save}_b0/parameters.csv", dtype={'station': str})

# for some reason in germany there is one less row...
    



if np.size(glob.glob(save_path_neg)) != 0:
    df_parameters_neg = pd.read_csv(save_path_neg, dtype={'station': str})

    #dataframe with all values
    new_df = df_parameters[['station','latitude','longitude','b','kappa','lambda','a']].copy()
    
    mask = new_df['b'] == 0
    
    new_df.loc[mask, 'b'] = df_parameters_neg['b2'].to_numpy()
    new_df.loc[mask, 'kappa'] = df_parameters_neg['kappa2'].to_numpy()
    new_df.loc[mask, 'lambda'] = df_parameters_neg['lambda2'].to_numpy()
    new_df.loc[mask, 'a'] = df_parameters_neg['a2'].to_numpy()

else:
    new_df = df_parameters.copy()

#merging the dataframes to ensure station consistency
missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    new_df = new_df.drop(missing_rows.index)
else:
    pass

S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = alpha_set,
        min_ev_dur = 60,
        niter_smev = 1000, 
        beta = beta_set2
    )

save_name = f"{drive}:/outputs/{country_save}\\return_levels{beta_set}.csv"
output_files = glob.glob(f"{drive}:/outputs/{country_save}/*")

#calculating and saving return levels for 0, free, 5% sig
if save_name not in output_files:
    print("levels not calculated yet, doing now.")
    
    
    RL = [0] * len(new_df)
    RL_0 = [0] * len(new_df)
    RL_5 = [0] * len(new_df)
    AMS_sort = [0] * len(new_df)
    start_time = [0] * len(new_df)
    FRMSE = [0] * len(new_df)
    FRMSE_5 = [0] * len(new_df)
    FRMSE_0 = [0] * len(new_df)
    station_flags_df = pd.DataFrame({"index_number" :[], "station": []})
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time() 
        #read in ppt data
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        
        if 'code_str' in locals():
            G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
        else:
            G = pd.read_csv(f"{file_name}.csv")
            G['prec_time'] = pd.to_datetime(G['prec_time'])
            G.set_index('prec_time', inplace=True)
            
        ######################################################################
        #TENAX  AMS
    
        data = G 
        data = S.remove_incomplete_years(data, name_col)
        
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
        
        AMS = dict_AMS['60']
        
        # Define the model parameters by reading in those already saved
        if S.beta == 4:
            g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
        else:
            T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
            
            if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                print('skip')
                g_phat = [np.nan,np.nan]
            else:
                T_ERA = xr.load_dataarray(T_path)
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
                
                AMS = dict_AMS['60']
                
                
                df_arr_t_data = np.array(t_data[temp_name_col])
                df_dates_t_data = np.array(t_data.index)
                
                dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                
                
                
                # Your data (P, T arrays) and threshold thr=3.8
                P = dict_ordinary["60"]["ordinary"].to_numpy() 
                T = dict_ordinary["60"]["T"].to_numpy()  
                g_phat = S.temperature_model(T)
        
        
        
        if np.any(np.isnan(g_phat[0])):
            print(f"no gphat. {g_phat}")
            RL[i] = np.nan
            RL_5[i] = np.nan
            RL_0[i]  = np.nan
            
            FRMSE[i] = np.nan
            FRMSE_5[i] = np.nan
            FRMSE_0[i] = np.nan
        else:
            #free
            F_phat = [new_df.kappa.iloc[i],new_df.b.iloc[i],
                      new_df["lambda"].iloc[i],new_df.a.iloc[i]]
            #5% sig
            F_phat_5 = [df_parameters.kappa.iloc[i],df_parameters.b.iloc[i],
                        df_parameters["lambda"].iloc[i],df_parameters.a.iloc[i]]
            #b always 0
            F_phat_0 = [df_parameters_0.kappa.iloc[i],df_parameters_0.b.iloc[i],
                        df_parameters_0["lambda"].iloc[i],df_parameters_0.a.iloc[i]]
            
            n = df_parameters.n_events_per_yr.iloc[i]
            
            # Getting predicted return levels
            AMS_sort[i] = AMS.sort_values(by=['AMS'])['AMS']
            plot_pos = np.arange(1,np.size(AMS_sort[i])+1)/(1+np.size(AMS_sort[i]))
            
            eRP = 1/(1-plot_pos)
            S.return_period = eRP
            
            T_min = g_phat[0] - 2.5 * g_phat[1]
            T_max = g_phat[0] + 2.5 * g_phat[1]
            Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
            
            RL[i], __, __ = S.model_inversion(F_phat, g_phat, n, Ts)
            RL_5[i], __, __ = S.model_inversion(F_phat_5, g_phat, n, Ts)
            RL_0[i], __, __ = S.model_inversion(F_phat_0, g_phat, n, Ts)
            
            diffs = RL[i] - AMS_sort[i]
            diffs_5 = RL_5[i] - AMS_sort[i]
            diffs_0 = RL_0[i] - AMS_sort[i]
            
            FRMSE[i] = np.sqrt(np.sum(diffs**2)/len(diffs))/(np.sum(AMS_sort[i])/len(diffs))
            FRMSE_5[i] = np.sqrt(np.sum(diffs_5**2)/len(diffs_5))/(np.sum(AMS_sort[i])/len(diffs_5))
            FRMSE_0[i] = np.sqrt(np.sum(diffs_0**2)/len(diffs_0))/(np.sum(AMS_sort[i])/len(diffs_0))
            if np.any(np.isnan(RL_5[i])):
                print("There is a NaN value in the RL.")
                station_flags_df = pd.concat([station_flags_df,pd.DataFrame({"index_number" :[i], "station": [df_parameters.station.iloc[i]]})])
            else:
                pass
            
        print(f"Free {FRMSE[i]}, 5% sig {FRMSE_5[i]}, b always 0 {FRMSE_0[i]}")
        print(f"normal gphat: {[df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]}, beta = {S.beta}: {g_phat}")
        time_taken = (time.time()-start_time[i-9])/10
        time_left = (len(new_df)-i)*time_taken/60
        print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
    
    
    nan_locs = np.where(np.isnan(FRMSE))
    replace_range = np.arange(0,len(AMS_sort))
    replace_range = np.setdiff1d(replace_range, nan_locs)
    
    AMS_sort_save = AMS_sort
    for j in replace_range:
        AMS_sort_save[j] = AMS_sort[j].to_numpy()
    
    RL_df = pd.DataFrame({'station': df_parameters.station, 'obs_AMS': AMS_sort_save, 'return_levels': RL, 'return_levels_5': RL_5, 'return_levels_b0': RL_0})
    RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels{beta_set}.csv",index=False)
    
    FRMSE_df = pd.DataFrame({'station': df_parameters.station,
                             'FRMSE': FRMSE,
                             'FRMSE_5': FRMSE_5,
                             'FRMSE_0': FRMSE_0})
    
    FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE{beta_set}.csv",index=False)
else:
    print("Files already saved, reading")
    RL_df = pd.read_csv(save_name, dtype={'station': str})
    nan_locs = RL_df.return_levels[RL_df.return_levels.isna()].index
    replace_range = np.arange(0,len(RL_df))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
        RL_df.loc[j, "return_levels"] = np.fromstring(RL_df.return_levels.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        RL_df.loc[j, "return_levels_5"] = np.fromstring(RL_df["return_levels_5"].iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        RL_df.loc[j, "return_levels_b0"] = np.fromstring(RL_df.return_levels_b0.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        RL_df.loc[j, "obs_AMS"] = np.fromstring(RL_df.obs_AMS.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
        
        if "return_levels_bset" in RL_df.columns:
            RL_df.loc[j, "return_levels_bset"] = np.fromstring(RL_df.return_levels_bset.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        
        if "return_levels_bexp" in RL_df.columns:
            RL_df.loc[j, "return_levels_bexp"] = np.fromstring(RL_df.return_levels_bexp.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        
    if "return_levels_roll" in RL_df.columns:
        nan_locs = RL_df.return_levels[RL_df.return_levels_roll.isna()].index
        replace_range = np.arange(0,len(RL_df))
        for k in range(len(nan_locs)):
            replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
        for j in replace_range:
            RL_df.loc[j, "return_levels_roll"] = np.fromstring(RL_df.return_levels_roll.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        
       
    
    FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE{beta_set}.csv", dtype={'station': str})



###############################################################################
# Probs and stuff


save_name_lik = f"{drive}:/outputs/{country_save}\\liklihood{beta_set}.csv"

if save_name_lik not in output_files:
        
    mult_prob = [0] * len(new_df)
    ave_prob = [0] * len(new_df)
    mins = [0] * len(new_df)
    maxes = [0] * len(new_df)
    n_bad_RL = [0] * len(new_df)
    
    
    mult_prob_5 = [0] * len(new_df)
    ave_prob_5 = [0] * len(new_df)
    mins_5 = [0] * len(new_df)
    maxes_5 = [0] * len(new_df)
    n_bad_RL_5 = [0] * len(new_df)
    
    
    mult_prob_0 = [0] * len(new_df)
    ave_prob_0 = [0] * len(new_df)
    mins_0 = [0] * len(new_df)
    maxes_0 = [0] * len(new_df)
    n_bad_RL_0 = [0] * len(new_df)
    g_phats = [0] * len(new_df)
    
    start_time = [0] * len(new_df)
    
    for i in np.arange(0,len(new_df)):
        start_time[i] = time.time()
        n_itn = 1000
        percentages = [0.05,0.95]
        if S.beta == 4:
            g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
        else:
            T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
            
            if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                print('skip')
                g_phat = [np.nan,np.nan]
            else:
                
                file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
                
                if 'code_str' in locals():
                    G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
                else:
                    G = pd.read_csv(f"{file_name}.csv")
                    G['prec_time'] = pd.to_datetime(G['prec_time'])
                    G.set_index('prec_time', inplace=True)
                    
                ######################################################################
                #TENAX  AMS
            
                data = G 
                data = S.remove_incomplete_years(data, name_col)
                
                T_ERA = xr.load_dataarray(T_path)
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
                
                AMS = dict_AMS['60']
                
                
                df_arr_t_data = np.array(t_data[temp_name_col])
                df_dates_t_data = np.array(t_data.index)
                
                dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                
                
                
                # Your data (P, T arrays) and threshold thr=3.8
                P = dict_ordinary["60"]["ordinary"].to_numpy() 
                T = dict_ordinary["60"]["T"].to_numpy()  
                
                g_phats[i] = S.temperature_model(T)
                g_phat = g_phats[i]
                
                
                T_min = g_phat[0] - 2.5 * g_phat[1]
                T_max = g_phat[0] + 2.5 * g_phat[1]
                Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
                
                
                
                
        if np.any(np.isnan(g_phat[0])):
            print(f"no gphat. {g_phat}")
            
            mult_prob[i] = np.nan
            ave_prob[i] = np.nan
        else:
            n = round(df_parameters.n_events_per_yr.iloc[i])
            #free
            F_phat = [new_df.kappa.iloc[i],new_df.b.iloc[i],
                      new_df["lambda"].iloc[i],new_df.a.iloc[i]]
            #5% sig
            F_phat_5 = [df_parameters.kappa.iloc[i],df_parameters.b.iloc[i],
                        df_parameters["lambda"].iloc[i],df_parameters.a.iloc[i]]
            #b always 0
            F_phat_0 = [df_parameters_0.kappa.iloc[i],df_parameters_0.b.iloc[i],
                        df_parameters_0["lambda"].iloc[i],df_parameters_0.a.iloc[i]]
                
            
            
            AMS_stat = RL_df.obs_AMS.iloc[i] 
            plot_pos = np.arange(1,np.size(AMS_stat)+1)/(1+AMS_stat)
            eRP = 1/(1-plot_pos)
            
            
            n_years = len(AMS_stat)
            S.n_monte_carlo = int(n_years*n)
            
            AMS_sim = np.zeros([n_itn,n_years])
            for itn in np.arange(0,n_itn):
                _, _, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,gen_P_mc = True,gen_RL=False,method_root_scalar="secant") 
                AMS_sim[itn,:] = [np.max(P_mc[j:j+n]) for j in np.arange(0,int(n_years*n),int(n))]
                AMS_sim[itn,:].sort()
            
            mins[i] = [np.quantile(AMS_sim[:,j],percentages[0]) for j in np.arange(0,n_years)]
            maxes[i] = [np.quantile(AMS_sim[:,j],percentages[1]) for j in np.arange(0,n_years)]
            
            
            outs = AMS_stat[(AMS_stat > maxes[i]) | (AMS_stat < mins[i])]
            n_bad_RL[i] = len(outs)
            
            prob = [0]*len(eRP)
            total_prob =[0]*len(eRP)
            kde = [0]*len(eRP)
            
            for RP_rank in np.arange(0,len(eRP)):
                valid_data = AMS_sim[:, RP_rank]
                valid_data = valid_data[np.isfinite(valid_data)]
                valid_data = valid_data[valid_data < 10000]
                data_removed = n_itn - len(valid_data)
                if data_removed > 50:
                    print(f"warning. {data_removed} values inf, nan, or too large. index {i}")
                else:
                    pass
                kde[RP_rank]  = gaussian_kde(valid_data) #use kernel density to get probability
                prob[RP_rank] = kde[RP_rank](AMS_stat[RP_rank])
                
            mult_prob[i] = np.prod(prob)
            ave_prob[i] = np.mean(prob)
            
            AMS_sim_0 = np.zeros([n_itn,n_years])
            for itn in np.arange(0,n_itn):
                _, _, P_mc = S.model_inversion(F_phat_0, g_phat, n, Ts,gen_P_mc = True,gen_RL=False,method_root_scalar="secant") 
                AMS_sim_0[itn,:] = [np.max(P_mc[j:j+n]) for j in np.arange(0,int(n_years*n),int(n))]
                AMS_sim_0[itn,:].sort()
            
            mins_0[i] = [np.quantile(AMS_sim_0[:,j],percentages[0]) for j in np.arange(0,n_years)]
            maxes_0[i] = [np.quantile(AMS_sim_0[:,j],percentages[1]) for j in np.arange(0,n_years)]
            
            
            outs_0 = AMS_stat[(AMS_stat > maxes_0[i]) | (AMS_stat < mins_0[i])]
            n_bad_RL_0[i] = len(outs_0)
            
            prob_0 = [0]*len(eRP)
            total_prob_0 =[0]*len(eRP)
            kde_0 = [0]*len(eRP)
            
            for RP_rank in np.arange(0,len(eRP)):
                valid_data = AMS_sim_0[:, RP_rank]
                valid_data = valid_data[np.isfinite(valid_data)]
                valid_data = valid_data[valid_data < 10000]
                data_removed = n_itn - len(valid_data)
                if data_removed > 50:
                    print(f"warning. {data_removed} values inf, nan, or too large. index {i}")
                else:
                    pass
                kde_0[RP_rank]  = gaussian_kde(valid_data) #use kernel density to get probability
                prob_0[RP_rank] = kde_0[RP_rank](AMS_stat[RP_rank])
                
            mult_prob_0[i] = np.prod(prob_0)
            ave_prob_0[i] = np.mean(prob_0)
            
            
            AMS_sim_5 = np.zeros([n_itn,n_years])
            for itn in np.arange(0,n_itn):
                _, _, P_mc = S.model_inversion(F_phat_5, g_phat, n, Ts,gen_P_mc = True,gen_RL=False,method_root_scalar="secant") 
                AMS_sim_5[itn,:] = [np.max(P_mc[j:j+n]) for j in np.arange(0,int(n_years*n),int(n))]
                AMS_sim_5[itn,:].sort()
            
            mins_5[i] = [np.quantile(AMS_sim_5[:,j],percentages[0]) for j in np.arange(0,n_years)]
            maxes_5[i] = [np.quantile(AMS_sim_5[:,j],percentages[1]) for j in np.arange(0,n_years)]
            
            
            outs_5 = AMS_stat[(AMS_stat > maxes_5[i]) | (AMS_stat < mins_5[i])]
            n_bad_RL_5[i] = len(outs_5)
            
            prob_5 = [0]*len(eRP)
            total_prob_5 =[0]*len(eRP)
            kde_5 = [0]*len(eRP)
            
            for RP_rank in np.arange(0,len(eRP)):
                valid_data = AMS_sim_5[:, RP_rank]
                valid_data = valid_data[np.isfinite(valid_data)]
                valid_data = valid_data[valid_data < 10000]
                data_removed = n_itn - len(valid_data)
                if data_removed > 50:
                    print(f"warning. {data_removed} values inf, nan, or too large. index {i}")
                else:
                    pass
                kde_5[RP_rank]  = gaussian_kde(valid_data) #use kernel density to get probability
                prob_5[RP_rank] = kde_5[RP_rank](AMS_stat[RP_rank])
                
            mult_prob_5[i] = np.prod(prob_5)
            ave_prob_5[i] = np.mean(prob_5)
            
        if i%50 == 0:
            print(f"multiplied prob {mult_prob[i]}")
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
        
    liklihood_df = pd.DataFrame({'station': df_parameters.station,
                                 "mult_prob": mult_prob,
                                 "ave_prob": ave_prob,
                                 "mins": mins,
                                 "maxes": maxes,
                                 "n_bad_RL": n_bad_RL,
                                 "mult_prob_0": mult_prob_0,
                                 "ave_prob_0": ave_prob_0,
                                 "mins_0": mins_0,
                                 "maxes_0": maxes_0,
                                 "n_bad_RL_0": n_bad_RL_0,
                                 "mult_prob_5": mult_prob_5,
                                 "ave_prob_5": ave_prob_5,
                                 "mins_5": mins_5,
                                 "maxes_5": maxes_5,
                                 "n_bad_RL_5": n_bad_RL_5,
                                 })
    
    liklihood_df.to_csv(f"{drive}:/outputs/{country_save}/liklihood{beta_set}.csv",index=False)
    
    if beta_set2 != 4:
        g_phat_df = pd.DataFrame({"station" : df_parameters.station,
                                  "mu": np.array(g_phats)[:,0],
                                  "sigma": np.array(g_phats)[:,1]})
        g_phat_df.to_csv(f"{drive}:/outputs/{country_save}/g_phat{beta_set}",index= False)
    
    
    
else:
    liklihood_df = pd.read_csv(f"{drive}:/outputs/{country_save}/liklihood{beta_set}.csv",dtype={'station': str})
    
    nan_locs = liklihood_df.mult_prob[liklihood_df.mult_prob.isna()].index
    replace_range = np.arange(0,len(RL_df))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
        liklihood_df.at[j, "maxes"] = eval(liklihood_df.maxes.iloc[j], {"np": np, "nan": np.nan})
        liklihood_df.at[j, "mins"] = eval(liklihood_df.mins.iloc[j], {"np": np, "nan": np.nan})
        
        liklihood_df.at[j, "maxes_0"] = eval(liklihood_df.maxes_0.iloc[j], {"np": np, "nan": np.nan})
        liklihood_df.at[j, "mins_0"] = eval(liklihood_df.mins_0.iloc[j], {"np": np, "nan": np.nan})
        
        liklihood_df.at[j, "maxes_5"] = eval(liklihood_df.maxes_5.iloc[j], {"np": np, "nan": np.nan})
        liklihood_df.at[j, "mins_5"] = eval(liklihood_df.mins_5.iloc[j], {"np": np, "nan": np.nan})
        
        
        # if "mins_bset" in liklihood_df.columns:
        #     liklihood_df.at[j, "maxes_bset"] = eval(liklihood_df.maxes_bset.iloc[j], {"np": np})
        #     liklihood_df.at[j, "mins_bset"] = eval(liklihood_df.mins_bset.iloc[j], {"np": np})
            
        # if "mins_bexp" in liklihood_df.columns:
        #     liklihood_df.at[j, "maxes_bexp"] = eval(liklihood_df.maxes_bexp.iloc[j], {"np": np})
        #     liklihood_df.at[j, "mins_bexp"] = eval(liklihood_df.mins_bexp.iloc[j], {"np": np})
    #TODO: mins_bexp etc are wrong. may need to reread    
    
################################################################################
# With set b

save_bset = f"{drive}:/outputs/{country_save}\\parameters_bset{beta_set}.csv"

if save_bset in output_files:
    print("hell yeah lets do some mean b liklihood")
    df_parameters_bset = pd.read_csv(save_bset,dtype={'station': str})
    S = TENAX(
            return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
            durations = [60, 180, 360, 720, 1440],
            left_censoring = [0, censor_thr],
            alpha = alpha_set,
            min_ev_dur = 60,
            niter_smev = 1000, 
            beta = beta_set2
        )
            
    
    if "mult_prob_bset" in liklihood_df.columns:
        print("you've already done it! bset data is ready")
    else:
        print(f"bset liklihoods not yet calculated for {country}")
        mult_prob_bset = [0] * len(new_df)
        ave_prob_bset = [0] * len(new_df)
        mins_bset = [0] * len(new_df)
        maxes_bset = [0] * len(new_df)
        n_bad_RL_bset = [0] * len(new_df)
        FRMSE_bset = [0] * len(new_df)
        
        RL = [0] * len(new_df)
        
        start_time = [0] * len(new_df)
        
        for i in np.arange(0,len(new_df)):
            start_time[i] = time.time()
            n_itn = 1000
            percentages = [0.05,0.95]
            
            if S.beta == 4:
                g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
            else:
                T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
                
                if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                    print('skip')
                    g_phat = [np.nan,np.nan]
                else:
                    file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
                    
                    if 'code_str' in locals():
                        G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
                    else:
                        G = pd.read_csv(f"{file_name}.csv")
                        G['prec_time'] = pd.to_datetime(G['prec_time'])
                        G.set_index('prec_time', inplace=True)
                        
                    ######################################################################
                    #TENAX  AMS
                
                    data = G 
                    data = S.remove_incomplete_years(data, name_col)
                    T_ERA = xr.load_dataarray(T_path)
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
                    
                    AMS = dict_AMS['60']
                    
                    
                    df_arr_t_data = np.array(t_data[temp_name_col])
                    df_dates_t_data = np.array(t_data.index)
                    
                    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                    
                    
                    
                    # Your data (P, T arrays) and threshold thr=3.8
                    P = dict_ordinary["60"]["ordinary"].to_numpy() 
                    T = dict_ordinary["60"]["T"].to_numpy()  
                    g_phat = S.temperature_model(T)
                    
                    
            if np.any(np.isnan(g_phat[0])):
                print(f" {i} no gphat. {g_phat}")
                
                mult_prob_bset[i] = np.nan
                ave_prob_bset[i] = np.nan
                FRMSE_bset[i] = np.nan
                n_bad_RL_bset[i] = np.nan
            else:
                n = round(df_parameters.n_events_per_yr.iloc[i])
                #free
                F_phat = [df_parameters_bset.kappa.iloc[i],df_parameters_bset.b.iloc[i],
                          df_parameters_bset["lambda"].iloc[i],df_parameters_bset.a.iloc[i]]
                    
                AMS_stat = RL_df.obs_AMS.iloc[i] 
                plot_pos = np.arange(1,np.size(AMS_stat)+1)/(1+AMS_stat)
                eRP = 1/(1-plot_pos)
                
                
                T_min = g_phat[0] - 2.5 * g_phat[1]
                T_max = g_phat[0] + 2.5 * g_phat[1]
                Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
                
                n_years = len(AMS_stat)
                S.n_monte_carlo = int(n_years*n)
                
                AMS_sim = np.zeros([n_itn,n_years])
                for itn in np.arange(0,n_itn):
                    _, _, P_mc = S.model_inversion(F_phat, g_phat, n, Ts, gen_P_mc = True,gen_RL=False,method_root_scalar="secant") 
                    AMS_sim[itn,:] = [np.max(P_mc[j:j+n]) for j in np.arange(0,int(n_years*n),int(n))]
                    AMS_sim[itn,:].sort()
                
                mins_bset[i] = [np.quantile(AMS_sim[:,j],percentages[0]) for j in np.arange(0,n_years)]
                maxes_bset[i] = [np.quantile(AMS_sim[:,j],percentages[1]) for j in np.arange(0,n_years)]
                
                
                outs = AMS_stat[(AMS_stat > maxes_bset[i]) | (AMS_stat < mins_bset[i])]
                n_bad_RL_bset[i] = len(outs)
                
                prob = [0]*len(eRP)
                total_prob =[0]*len(eRP)
                kde = [0]*len(eRP)
                
                for RP_rank in np.arange(0,len(eRP)):
                    valid_data = AMS_sim[:, RP_rank]
                    valid_data = valid_data[np.isfinite(valid_data)]
                    valid_data = valid_data[valid_data < 10000]
                    data_removed = n_itn - len(valid_data)
                    if data_removed > 50:
                        print(f"warning. {data_removed} values inf, nan, or too large. index {i}")
                    else:
                        pass
                    kde[RP_rank]  = gaussian_kde(valid_data) #use kernel density to get probability
                    prob[RP_rank] = kde[RP_rank](AMS_stat[RP_rank])
                    
                mult_prob_bset[i] = np.prod(prob)
                ave_prob_bset[i] = np.mean(prob)
                
                
                
                RL[i] = np.fromstring(df_parameters_bset.return_levels.iloc[i].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
                
                
                diffs = RL[i] - AMS_stat
                
                FRMSE_bset[i] = np.sqrt(np.sum(diffs**2)/len(diffs))/(np.sum(AMS_stat)/len(diffs))
                
                
            if i%50 == 0:
                print(f"multiplied prob {mult_prob_bset[i]}")
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
                
        liklihood_df["mult_prob_bset"] = mult_prob_bset
        liklihood_df["ave_prob_bset"] = ave_prob_bset
        liklihood_df["mins_bset"] = mins_bset
        liklihood_df["maxes_bset"] = maxes_bset
        liklihood_df["n_bad_RL_bset"] = n_bad_RL_bset
        
        FRMSE_df["FRMSE_bset"] = FRMSE_bset
        RL_df["return_levels_bset"] = RL
        
        liklihood_df.to_csv(f"{drive}:/outputs/{country_save}/liklihood{beta_set}.csv",index=False)
        RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels{beta_set}.csv",index=False)
        FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE{beta_set}.csv",index=False)

else:
    print("go to Calc_b if you want to look at b mean")

###############################################################################
# With exp b

save_bexp = f"{drive}:/outputs/{country_save}\\parameters_exp{beta_set}.csv"

if save_bexp in output_files:
    print("hell yeah lets do some exponential b liklihood")
    df_parameters_bexp = pd.read_csv(save_bexp,dtype={'station': str})
        
    
    if "mult_prob_bexp" in liklihood_df.columns:
        print("you've already done it! bexp data is ready")
    else:
        print(f"bexp liklihoods not yet calculated for {country}")
        mult_prob_bexp = [0] * len(new_df)
        ave_prob_bexp = [0] * len(new_df)
        mins_bexp = [0] * len(new_df)
        maxes_bexp = [0] * len(new_df)
        n_bad_RL_bexp = [0] * len(new_df)
        FRMSE_bexp = [0] * len(new_df)
        
        RL = [0] * len(new_df)
        
        start_time = [0] * len(new_df)
        
        for i in np.arange(0,len(new_df)):
            start_time[i] = time.time()
            n_itn = 1000
            percentages = [0.05,0.95]
            
            if S.beta == 4:
                g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
            else:
                T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
                
                if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                    print('skip')
                    g_phat = [np.nan,np.nan]
                else:
                    T_ERA = xr.load_dataarray(T_path)
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
                    
                    AMS = dict_AMS['60']
                    
                    
                    df_arr_t_data = np.array(t_data[temp_name_col])
                    df_dates_t_data = np.array(t_data.index)
                    
                    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                    
                    
                    
                    # Your data (P, T arrays) and threshold thr=3.8
                    P = dict_ordinary["60"]["ordinary"].to_numpy() 
                    T = dict_ordinary["60"]["T"].to_numpy()  
                    g_phat = S.temperature_model(T)
                    
                    
            if np.any(np.isnan(g_phat[0])):
                print(f"no gphat. {g_phat}")
                
                mult_prob_bexp[i] = np.nan
                mult_prob_bexp[i] = np.nan
                FRMSE_bexp[i] = np.nan
                n_bad_RL_bexp[i] = np.nan 
                
            else:
                n = round(df_parameters.n_events_per_yr.iloc[i])
                #free
                F_phat = [df_parameters_bexp.kappa.iloc[i],df_parameters_bexp.b.iloc[i],
                          df_parameters_bexp["lambda"].iloc[i],df_parameters_bexp.a.iloc[i]]
                    
                AMS_stat = RL_df.obs_AMS.iloc[i] 
                plot_pos = np.arange(1,np.size(AMS_stat)+1)/(1+AMS_stat)
                eRP = 1/(1-plot_pos)
                
                
                T_min = g_phat[0] - 2.5 * g_phat[1]
                T_max = g_phat[0] + 2.5 * g_phat[1]
                Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
                
                n_years = len(AMS_stat)
                S.n_monte_carlo = int(n_years*n)
                
                AMS_sim = np.zeros([n_itn,n_years])
                for itn in np.arange(0,n_itn):
                    _, _, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,gen_P_mc = True,gen_RL=False,method_root_scalar="secant",b_exp = True) 
                    AMS_sim[itn,:] = [np.max(P_mc[j:j+n]) for j in np.arange(0,int(n_years*n),int(n))]
                    AMS_sim[itn,:].sort()
                
                mins_bexp[i] = [np.quantile(AMS_sim[:,j],percentages[0]) for j in np.arange(0,n_years)]
                maxes_bexp[i] = [np.quantile(AMS_sim[:,j],percentages[1]) for j in np.arange(0,n_years)]
                
                
                outs = AMS_stat[(AMS_stat > maxes_bexp[i]) | (AMS_stat < mins_bexp[i])]
                n_bad_RL_bexp[i] = len(outs)
                
                prob = [0]*len(eRP)
                total_prob =[0]*len(eRP)
                kde = [0]*len(eRP)
                
                for RP_rank in np.arange(0,len(eRP)):
                    valid_data = AMS_sim[:, RP_rank]
                    valid_data = valid_data[np.isfinite(valid_data)]
                    valid_data = valid_data[valid_data < 10000]
                    data_removed = n_itn - len(valid_data)
                    if data_removed > 50:
                        print(f"warning. {data_removed} values inf, nan, or too large. index {i}")
                    else:
                        pass
                    kde[RP_rank]  = gaussian_kde(valid_data) #use kernel density to get probability
                    prob[RP_rank] = kde[RP_rank](AMS_stat[RP_rank])
                    
                mult_prob_bexp[i] = np.prod(prob)
                ave_prob_bexp[i] = np.mean(prob)
                
                
                
                RL[i] = np.fromstring(df_parameters_bexp.return_levels.iloc[i].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
                
                
                diffs = RL[i] - AMS_stat
                
                FRMSE_bexp[i] = np.sqrt(np.sum(diffs**2)/len(diffs))/(np.sum(AMS_stat)/len(diffs))
                
                
            if i%50 == 0:
                print(f"multiplied prob {mult_prob_bexp[i]}")
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
                
        liklihood_df["mult_prob_bexp"] = mult_prob_bexp
        liklihood_df["ave_prob_bexp"] = ave_prob_bexp
        liklihood_df["mins_bexp"] = mins_bexp
        liklihood_df["maxes_bexp"] = maxes_bexp
        liklihood_df["n_bad_RL_bexp"] = n_bad_RL_bexp
        
        FRMSE_df["FRMSE_bexp"] = FRMSE_bexp
        RL_df["return_levels_bexp"] = RL
        
        liklihood_df.to_csv(f"{drive}:/outputs/{country_save}/liklihood{beta_set}.csv",index=False)
        RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels{beta_set}.csv",index=False)
        FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE{beta_set}.csv",index=False)

else:
    print("go to Calc_b if you want to look at b exp")

###############################################################################
# With rolling b

save_roll = glob.glob(f"{drive}:/outputs/{country_save}\\parameters_rolling*")[0]

if save_roll in output_files:
    print("hell yeah lets do some rolling b liklihood")
    df_parameters_rolling = pd.read_csv(save_roll,dtype={'station': str})
    nan_locs = df_parameters_rolling.b[df_parameters_rolling.b.isna()].index
    replace_range = np.arange(0,len(df_parameters_rolling))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
        df_parameters_rolling.at[j, "return_levels"] = np.fromstring(df_parameters_rolling.return_levels.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
    
    if "mult_prob_roll" in liklihood_df.columns:
        print("you've already done it! b roll data is ready")
    else:
        print(f"b roll liklihoods not yet calculated for {country}")
        mult_prob_roll = [0] * len(new_df)
        ave_prob_roll = [0] * len(new_df)
        mins_roll = [0] * len(new_df)
        maxes_roll = [0] * len(new_df)
        n_bad_RL_roll = [0] * len(new_df)
        FRMSE_roll = [0] * len(new_df)
        
        
        start_time = [0] * len(new_df)
        
        for i in np.arange(0,len(new_df)):
            start_time[i] = time.time()
            n_itn = 1000
            percentages = [0.05,0.95]
            
            if S.beta == 4:
                g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
            else:
                T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
                
                if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                    print('skip')
                    g_phat = [np.nan,np.nan]
                else:
                    T_ERA = xr.load_dataarray(T_path)
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
                    
                    AMS = dict_AMS['60']
                    
                    
                    df_arr_t_data = np.array(t_data[temp_name_col])
                    df_dates_t_data = np.array(t_data.index)
                    
                    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                    
                    
                    
                    # Your data (P, T arrays) and threshold thr=3.8
                    P = dict_ordinary["60"]["ordinary"].to_numpy() 
                    T = dict_ordinary["60"]["T"].to_numpy()  
                    g_phat = S.temperature_model(T)
                    
                    
            if np.any(np.isnan(g_phat[0])):
                print(f"no gphat. {g_phat}")
                
                mult_prob_roll[i] = np.nan
                mult_prob_roll[i] = np.nan
                FRMSE_roll[i] = np.nan
                n_bad_RL_roll[i] = np.nan 
                
            else:
                n = round(df_parameters.n_events_per_yr.iloc[i])
                #free
                F_phat = [df_parameters_rolling.kappa.iloc[i],df_parameters_rolling.b.iloc[i],
                          df_parameters_rolling["lambda"].iloc[i],df_parameters_rolling.a.iloc[i]]
                if np.any(np.isnan(F_phat[0])):
                    print(f"{i}. no F_phat")
                    mins_roll[i] = [np.nan,np.nan]
                    maxes_roll[i] = [np.nan,np.nan]
                    mult_prob_roll[i] = np.nan
                    mult_prob_roll[i] = np.nan
                    FRMSE_roll[i] = np.nan
                    n_bad_RL_roll[i] = np.nan
                
                else:
                    AMS_stat = RL_df.obs_AMS.iloc[i] 
                    plot_pos = np.arange(1,np.size(AMS_stat)+1)/(1+AMS_stat)
                    eRP = 1/(1-plot_pos)
                    
                    
                    T_min = g_phat[0] - 2.5 * g_phat[1]
                    T_max = g_phat[0] + 2.5 * g_phat[1]
                    Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
                    
                    n_years = len(AMS_stat)
                    S.n_monte_carlo = int(n_years*n)
                    
                    AMS_sim = np.zeros([n_itn,n_years])
                    for itn in np.arange(0,n_itn):
                        _, _, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,gen_P_mc = True,gen_RL=False,method_root_scalar="secant") 
                        AMS_sim[itn,:] = [np.max(P_mc[j:j+n]) for j in np.arange(0,int(n_years*n),int(n))]
                        AMS_sim[itn,:].sort()
                    
                    mins_roll[i] = [np.quantile(AMS_sim[:,j],percentages[0]) for j in np.arange(0,n_years)]
                    maxes_roll[i] = [np.quantile(AMS_sim[:,j],percentages[1]) for j in np.arange(0,n_years)]
                    
                    
                    outs = AMS_stat[(AMS_stat > maxes_roll[i]) | (AMS_stat < mins_roll[i])]
                    n_bad_RL_roll[i] = len(outs)
                    
                    prob = [0]*len(eRP)
                    total_prob =[0]*len(eRP)
                    kde = [0]*len(eRP)
                    
                    for RP_rank in np.arange(0,len(eRP)):
                        valid_data = AMS_sim[:, RP_rank]
                        valid_data = valid_data[np.isfinite(valid_data)]
                        valid_data = valid_data[valid_data < 10000]
                        data_removed = n_itn - len(valid_data)
                        if data_removed > 50:
                            print(f"warning. {data_removed} values inf, nan, or too large. index {i}")
                        else:
                            pass
                        kde[RP_rank]  = gaussian_kde(valid_data) #use kernel density to get probability
                        prob[RP_rank] = kde[RP_rank](AMS_stat[RP_rank])
                        
                    mult_prob_roll[i] = np.prod(prob)
                    ave_prob_roll[i] = np.mean(prob)
                    
                    
                    
                    RL = df_parameters_rolling.return_levels.iloc[i]
                    
                    diffs = RL - AMS_stat
                    
                    FRMSE_roll[i] = np.sqrt(np.sum(diffs**2)/len(diffs))/(np.sum(AMS_stat)/len(diffs))
                    
                
            if i%50 == 0:
                print(f"multiplied prob {mult_prob_roll[i]}")
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
                
        liklihood_df["mult_prob_roll"] = mult_prob_roll
        liklihood_df["ave_prob_roll"] = ave_prob_roll
        liklihood_df["mins_roll"] = mins_roll
        liklihood_df["maxes_roll"] = maxes_roll
        liklihood_df["n_bad_RL_roll"] = n_bad_RL_roll
        
        FRMSE_df["FRMSE_roll"] = FRMSE_roll
        RL_df["return_levels_roll"] = df_parameters_rolling.return_levels
        
        liklihood_df.to_csv(f"{drive}:/outputs/{country_save}/liklihood{beta_set}.csv",index=False)
        RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels{beta_set}.csv",index=False)
        FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE{beta_set}.csv",index=False)

else:
    print("go to parameter_space_mean if you want to look at b roll")




###############################################################################
# With rolling b exponential

save_roll_exp = glob.glob(f"{drive}:/outputs/{country_save}\\parameters_rolling_exp*")[0]

if save_roll_exp in output_files:
    print("hell yeah lets do some rolling b exponential liklihood")
    df_parameters_rolling_exp = pd.read_csv(save_roll_exp,dtype={'station': str})
    nan_locs = df_parameters_rolling_exp.b[df_parameters_rolling_exp.b.isna()].index
    replace_range = np.arange(0,len(df_parameters_rolling_exp))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
        df_parameters_rolling_exp.at[j, "return_levels"] = np.fromstring(df_parameters_rolling_exp.return_levels.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
    
    if "mult_prob_roll_exp" in liklihood_df.columns:
        print("you've already done it! b roll data is ready")
    else:
        print(f"b roll exp liklihoods not yet calculated for {country}")
        mult_prob_roll_exp = [0] * len(new_df)
        ave_prob_roll_exp = [0] * len(new_df)
        mins_roll_exp = [0] * len(new_df)
        maxes_roll_exp = [0] * len(new_df)
        n_bad_RL_roll_exp = [0] * len(new_df)
        FRMSE_roll_exp = [0] * len(new_df)
        
        
        start_time = [0] * len(new_df)
        
        for i in np.arange(0,len(new_df)):
            start_time[i] = time.time()
            n_itn = 1000
            percentages = [0.05,0.95]
            
            if S.beta == 4:
                g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
            else:
                T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters_rolling_exp.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
                
                if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                    print('skip')
                    g_phat = [np.nan,np.nan]
                else:
                    T_ERA = xr.load_dataarray(T_path)
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
                    
                    AMS = dict_AMS['60']
                    
                    
                    df_arr_t_data = np.array(t_data[temp_name_col])
                    df_dates_t_data = np.array(t_data.index)
                    
                    dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                    
                    
                    
                    # Your data (P, T arrays) and threshold thr=3.8
                    P = dict_ordinary["60"]["ordinary"].to_numpy() 
                    T = dict_ordinary["60"]["T"].to_numpy()  
                    g_phat = S.temperature_model(T)
                    
                    
            if np.any(np.isnan(g_phat[0])):
                print(f"no gphat. {g_phat}")
                
                mult_prob_roll_exp[i] = np.nan
                mult_prob_roll_exp[i] = np.nan
                FRMSE_roll_exp[i] = np.nan
                n_bad_RL_roll_exp[i] = np.nan 
                
            else:
                n = round(df_parameters.n_events_per_yr.iloc[i])
                #free
                F_phat = [df_parameters_rolling_exp.kappa.iloc[i],df_parameters_rolling_exp.b.iloc[i],
                          df_parameters_rolling_exp["lambda"].iloc[i],df_parameters_rolling_exp.a.iloc[i]]
                if np.any(np.isnan(F_phat[0])):
                    print(f"{i}. no F_phat")
                    mins_roll_exp[i] = [np.nan,np.nan]
                    maxes_roll_exp[i] = [np.nan,np.nan]
                    mult_prob_roll_exp[i] = np.nan
                    mult_prob_roll_exp[i] = np.nan
                    FRMSE_roll_exp[i] = np.nan
                    n_bad_RL_roll_exp[i] = np.nan
                
                else:
                    AMS_stat = RL_df.obs_AMS.iloc[i] 
                    plot_pos = np.arange(1,np.size(AMS_stat)+1)/(1+AMS_stat)
                    eRP = 1/(1-plot_pos)
                    
                    
                    T_min = g_phat[0] - 2.5 * g_phat[1]
                    T_max = g_phat[0] + 2.5 * g_phat[1]
                    Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
                    
                    n_years = len(AMS_stat)
                    S.n_monte_carlo = int(n_years*n)
                    
                    AMS_sim = np.zeros([n_itn,n_years])
                    for itn in np.arange(0,n_itn):
                        _, _, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,gen_P_mc = True,gen_RL=False,method_root_scalar="secant",b_exp = True) 
                        AMS_sim[itn,:] = [np.max(P_mc[j:j+n]) for j in np.arange(0,int(n_years*n),int(n))]
                        AMS_sim[itn,:].sort()
                    
                    mins_roll_exp[i] = [np.quantile(AMS_sim[:,j],percentages[0]) for j in np.arange(0,n_years)]
                    maxes_roll_exp[i] = [np.quantile(AMS_sim[:,j],percentages[1]) for j in np.arange(0,n_years)]
                    
                    
                    outs = AMS_stat[(AMS_stat > maxes_roll_exp[i]) | (AMS_stat < mins_roll_exp[i])]
                    n_bad_RL_roll_exp[i] = len(outs)
                    
                    prob = [0]*len(eRP)
                    total_prob =[0]*len(eRP)
                    kde = [0]*len(eRP)
                    
                    for RP_rank in np.arange(0,len(eRP)):
                        valid_data = AMS_sim[:, RP_rank]
                        valid_data = valid_data[np.isfinite(valid_data)]
                        valid_data = valid_data[valid_data < 10000]
                        data_removed = n_itn - len(valid_data)
                        if data_removed > 50:
                            print(f"warning. {data_removed} values inf, nan, or too large. index {i}")
                        else:
                            pass
                        kde[RP_rank]  = gaussian_kde(valid_data) #use kernel density to get probability
                        prob[RP_rank] = kde[RP_rank](AMS_stat[RP_rank])
                        
                    mult_prob_roll_exp[i] = np.prod(prob)
                    ave_prob_roll_exp[i] = np.mean(prob)
                    
                    
                    
                    RL = df_parameters_rolling_exp.return_levels.iloc[i]
                    
                    diffs = RL - AMS_stat
                    
                    FRMSE_roll_exp[i] = np.sqrt(np.sum(diffs**2)/len(diffs))/(np.sum(AMS_stat)/len(diffs))
                    
                
            if i%50 == 0:
                print(f"multiplied prob {mult_prob_roll_exp[i]}")
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins")
                
        liklihood_df["mult_prob_roll_exp"] = mult_prob_roll_exp
        liklihood_df["ave_prob_roll_exp"] = ave_prob_roll_exp
        liklihood_df["mins_roll_exp"] = mins_roll_exp
        liklihood_df["maxes_roll_exp"] = maxes_roll_exp
        liklihood_df["n_bad_RL_roll_exp"] = n_bad_RL_roll_exp
        
        FRMSE_df["FRMSE_roll_exp"] = FRMSE_roll_exp
        RL_df["return_levels_roll_exp"] = df_parameters_rolling_exp.return_levels
        
        liklihood_df.to_csv(f"{drive}:/outputs/{country_save}/liklihood{beta_set}.csv",index=False)
        RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels{beta_set}.csv",index=False)
        FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE{beta_set}.csv",index=False)

else:
    print("go to parameter_space_mean if you want to look at b roll")




###############################################################################

#maps
significants = df_parameters[df_parameters.b != 0]
show_sig_locs = False


lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
s = 5
cmap = 'magma_r'


fig = plt.figure(figsize=(15, 10))
norm = mcolors.Normalize(vmin=0, vmax=1)


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=FRMSE_df.FRMSE,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b free")



ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=FRMSE_df.FRMSE_5,
    cmap=cmap,
    norm = norm,  
    s = s,
)

#plot the locations of significant stations
if show_sig_locs:
    scsig = ax2.scatter(
        significants.longitude,
        significants.latitude,s=20, facecolors='none', edgecolors='r'
    )
else:
    pass

ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b significant at 5% level")


ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=FRMSE_df.FRMSE_0,
    cmap=cmap,
    norm = norm,  
    s = s,  
)
ax3.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax3.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax3.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax3.set_title("b = 0")

ax4 = fig.add_subplot(2, 2, 4, projection=proj)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS, linestyle=':')


sc4 = ax4.scatter(
    val_info.longitude,
    val_info.latitude,
    c=val_info.cleaned_years,
    cmap="viridis",
    s = s,  
)
ax4.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax4.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax4.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

fig.subplots_adjust(right=0.85)

cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])  # Position for the colorbar outside
cb4 = plt.colorbar(sc4, cax=cbar_ax4)  # Colorbar for ax4
cb4.set_label('Number of complete years', fontsize=14)
cb4.ax.tick_params(labelsize=12)

ax4.set_title("cleaned years")





# Add a colorbar at the bottom
cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label('FRMSE', fontsize=14)
cb.ax.tick_params(labelsize=12)

# Set x and y ticks


#fig.tight_layout()
fig.suptitle(f'GSDR: {ERA_country}. FRMSE on RL', fontsize=16)
plt.show()

#differences

fig = plt.figure(figsize=(15, 10))
norm = mcolors.Normalize(vmin=-0.2, vmax=0.2)
cmap = 'seismic'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df.FRMSE - FRMSE_df.FRMSE_5),
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b free - 5% sig")



ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df.FRMSE - FRMSE_df.FRMSE_0),
    cmap=cmap,
    norm = norm,  
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b free - b 0")


ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df.FRMSE_5 - FRMSE_df.FRMSE_0),
    cmap=cmap,
    norm = norm,  
    s = s,  
)



ax3.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax3.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax3.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax3.set_title("b 5 - b 0")

ax4 = fig.add_subplot(2, 2, 4, projection=proj)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS, linestyle=':')


sc4 = ax4.scatter(
    val_info.longitude,
    val_info.latitude,
    c=val_info.cleaned_years,
    cmap="viridis",
    s = s,  
)
ax4.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax4.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax4.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

fig.subplots_adjust(right=0.85)

cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])  # Position for the colorbar outside
cb4 = plt.colorbar(sc4, cax=cbar_ax4)  # Colorbar for ax4
cb4.set_label('Number of complete years', fontsize=14)
cb4.ax.tick_params(labelsize=12)

ax4.set_title("cleaned years")





# Add a colorbar at the bottom
cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label(r'$\Delta$FRMSE', fontsize=14)
cb.ax.tick_params(labelsize=12)

# Set x and y ticks


#fig.tight_layout()
fig.suptitle(f'GSDR: {ERA_country}. FRMSE on RL', fontsize=16)
plt.show()
#############################################################################
#probabilities

fig = plt.figure(figsize=(15, 10))
norm = mcolors.Normalize(vmin=0, vmax=0.5)
cmap = 'hsv'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = liklihood_df.ave_prob,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b free")



ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=liklihood_df.ave_prob_5,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("5%")


ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=liklihood_df.ave_prob_0,
    cmap=cmap,  
    norm = norm,
    s = s,  
)



ax3.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax3.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax3.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax3.set_title("b = 0")

ax4 = fig.add_subplot(2, 2, 4, projection=proj)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS, linestyle=':')


sc4 = ax4.scatter(
    val_info.longitude,
    val_info.latitude,
    c=val_info.cleaned_years,
    cmap="viridis",
    s = s,  
)
ax4.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax4.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax4.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

fig.subplots_adjust(right=0.85)

cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])  # Position for the colorbar outside
cb4 = plt.colorbar(sc4, cax=cbar_ax4)  # Colorbar for ax4
cb4.set_label('Number of complete years', fontsize=14)
cb4.ax.tick_params(labelsize=12)

ax4.set_title("cleaned years")





# Add a colorbar at the bottom
cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label('Average probability', fontsize=14)
cb.ax.tick_params(labelsize=12)


#fig.tight_layout()
fig.suptitle(f'GSDR: {ERA_country}.', fontsize=16)
plt.show()





if "mult_prob_bset" in liklihood_df.columns:
    
    #difference
    fig = plt.figure(figsize=(10, 10))
    norm = mcolors.Normalize(vmin=-0.05, vmax=0.05)
    cmap = 'seismic'
    
    
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(2, 2, 1, projection=proj)
    
    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax1.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c = liklihood_df.ave_prob_bexp - liklihood_df.ave_prob_0,
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  
    
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax1.set_title(f"b exp - b 0")
    
    
    
    ax2 = fig.add_subplot(2, 2, 2, projection=proj)
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax2.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=liklihood_df.ave_prob - liklihood_df.ave_prob_0,
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax2.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax2.set_title("free - 0")
    
    
    ax3 = fig.add_subplot(2, 2, 3, projection=proj)
    ax3.coastlines()
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax3.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=liklihood_df.ave_prob_bset - liklihood_df.ave_prob_0,
        cmap=cmap,  
        norm = norm,
        s = s,  
    )
    
    
    
    ax3.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax3.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax3.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax3.set_title("bset - 0")
    
    ax4 = fig.add_subplot(2, 2, 4, projection=proj)
    ax4.coastlines()
    ax4.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc4 = ax4.scatter(
        val_info.longitude,
        val_info.latitude,
        c=val_info.cleaned_years,
        cmap="viridis",
        s = s,  
    )
    ax4.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax4.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax4.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    
    fig.subplots_adjust(right=0.85)
    
    cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])  # Position for the colorbar outside
    cb4 = plt.colorbar(sc4, cax=cbar_ax4)  # Colorbar for ax4
    cb4.set_label('Number of complete years', fontsize=14)
    cb4.ax.tick_params(labelsize=12)
    
    ax4.set_title("cleaned years")
    
    
    # Add a colorbar at the bottom
    cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Average probability', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    
    #fig.tight_layout()
    fig.suptitle(f'GSDR: {ERA_country}. \n average b: {df_parameters_bset.b.iloc[0]:.3f}', fontsize=16)
    plt.show()
    
    
    #mult logs
    fig = plt.figure(figsize=(10, 10))
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = 'seismic'
    
    
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(2, 2, 1, projection=proj)
    
    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax1.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c = np.log(liklihood_df.mult_prob_bexp) - np.log(liklihood_df.mult_prob_0),
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  
    
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ave = np.mean(np.log(liklihood_df.mult_prob_bexp) - np.log(liklihood_df.mult_prob_0))
    ax1.set_title(f"b exp - b 0. ave = {ave:.3f}")
    
    
    
    ax2 = fig.add_subplot(2, 2, 2, projection=proj)
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax2.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=np.log(liklihood_df.mult_prob) - np.log(liklihood_df.mult_prob_0),
        cmap=cmap,
        norm = norm,
        s = s,
    )
    
    ave = np.mean(np.log(liklihood_df.mult_prob) - np.log(liklihood_df.mult_prob_0))
    
    ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax2.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax2.set_title(f"free - 0. ave = {ave:.3f}")
    
    
    ax3 = fig.add_subplot(2, 2, 3, projection=proj)
    ax3.coastlines()
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax3.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=np.log(liklihood_df.mult_prob_bset) - np.log(liklihood_df.mult_prob_0),
        cmap=cmap,  
        norm = norm,
        s = s,  
    )
    
    ave = np.mean(np.log(liklihood_df.mult_prob_bset) - np.log(liklihood_df.mult_prob_0))
    
    ax3.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax3.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax3.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax3.set_title(f"bset - 0. ave = {ave:.3f}")
    
    ax4 = fig.add_subplot(2, 2, 4, projection=proj)
    ax4.coastlines()
    ax4.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc4 = ax4.scatter(
        val_info.longitude,
        val_info.latitude,
        c=val_info.cleaned_years,
        cmap="viridis",
        s = s,  
    )
    ax4.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax4.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax4.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    
    fig.subplots_adjust(right=0.85)
    
    cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])  # Position for the colorbar outside
    cb4 = plt.colorbar(sc4, cax=cbar_ax4)  # Colorbar for ax4
    cb4.set_label('Number of complete years', fontsize=14)
    cb4.ax.tick_params(labelsize=12)
    
    ax4.set_title("cleaned years")
    
    
    # Add a colorbar at the bottom
    cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('difference in logs', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    
    #fig.tight_layout()
    fig.suptitle(f'GSDR: {ERA_country}.', fontsize=16)
    plt.show()

elif "mult_prob_bexp" in liklihood_df.columns:
    #mult logs
    fig = plt.figure(figsize=(15, 7))
    norm = mcolors.Normalize(vmin=-5, vmax=5)
    cmap = 'seismic'
    
    
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    
    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax1.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c = np.log(liklihood_df.mult_prob_bexp) - np.log(liklihood_df.mult_prob_0),
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  
    
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ave = np.nanmean(np.log(liklihood_df.mult_prob_bexp.replace(0, np.nan)) - np.log(liklihood_df.mult_prob_0.replace(0, np.nan)))
    ax1.set_title(f"b exp - b 0. ave = {ave:.3f}")
    
    
    
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax2.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=np.log(liklihood_df.mult_prob) - np.log(liklihood_df.mult_prob_0),
        cmap=cmap,
        norm = norm,
        s = s,
    )
    
    to_mean = (np.log(liklihood_df.mult_prob.replace(0, np.nan)) - np.log(liklihood_df.mult_prob_0.replace(0, np.nan)))
    ave = np.nanmean(to_mean)
    
    ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax2.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax2.set_title(f"free - 0. ave = {ave:.3f}")
    
    
    
    fig.subplots_adjust(right=0.85)
    
    
    
    # Add a colorbar at the bottom
    cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('difference in logs', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    
    #fig.tight_layout()
    fig.suptitle(f'GSDR: {ERA_country}. beta = {S.beta}', fontsize=16)
    plt.show()
    
else:
    print("sorry can't plot that graph :(")
    #difference
    fig = plt.figure(figsize=(20, 20))
    norm = mcolors.Normalize(vmin=-0.05, vmax=0.05)
    cmap = 'seismic'
    
    
    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(2, 2, 1, projection=proj)
    
    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax1.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c = liklihood_df.ave_prob - liklihood_df.ave_prob_5,
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  
    
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax1.set_title(f"b free - 5")
    
    
    
    ax2 = fig.add_subplot(2, 2, 2, projection=proj)
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax2.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=liklihood_df.ave_prob - liklihood_df.ave_prob_0,
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax2.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax2.set_title("free - 0")
    
    
    ax3 = fig.add_subplot(2, 2, 3, projection=proj)
    ax3.coastlines()
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc = ax3.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=liklihood_df.ave_prob_5 - liklihood_df.ave_prob_0,
        cmap=cmap,  
        norm = norm,
        s = s,  
    )
    
    
    
    ax3.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax3.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax3.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax3.set_title("5 - 0")
    
    ax4 = fig.add_subplot(2, 2, 4, projection=proj)
    ax4.coastlines()
    ax4.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    sc4 = ax4.scatter(
        val_info.longitude,
        val_info.latitude,
        c=val_info.cleaned_years,
        cmap="viridis",
        s = s,  
    )
    ax4.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax4.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax4.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    
    fig.subplots_adjust(right=0.85)
    
    cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])  # Position for the colorbar outside
    cb4 = plt.colorbar(sc4, cax=cbar_ax4)  # Colorbar for ax4
    cb4.set_label('Number of complete years', fontsize=14)
    cb4.ax.tick_params(labelsize=12)
    
    ax4.set_title("cleaned years")
    
    
    # Add a colorbar at the bottom
    cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
    cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Average probability', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    
    #fig.tight_layout()
    fig.suptitle(f'GSDR: {ERA_country}. beta = {S.beta}', fontsize=16)
    plt.show()

##############################################################################
# 


##############################################################################
#Plot return levels
RL10_df = pd.DataFrame({
    "return_levels":np.zeros(len(new_df)),
    "return_levels_b0":np.zeros(len(new_df)),
    "return_levels_5":np.zeros(len(new_df)),
    })

for i in np.arange(0,len(new_df)):
    plot_pos = np.arange(1,np.size(RL_df.obs_AMS.iloc[i])+1)/(1+np.size(RL_df.obs_AMS.iloc[i]))

    eRP = 1/(1-plot_pos)
    RL_free = RL_df.return_levels.iloc[i]
    RL_0 = RL_df.return_levels_b0.iloc[i]
    RL_5 = RL_df.return_levels_5.iloc[i]
    
    if np.size(RL_free) == 1:
        RL10_df.loc[i, 'return_levels'] = np.nan
    else:
        interp_func = interp1d(eRP, RL_free)
        RL10_df.loc[i, 'return_levels'] = interp_func(10)
        
    if np.size(RL_0) == 1:
        RL10_df.loc[i, 'return_levels_b0'] = np.nan
    else:
        interp_func = interp1d(eRP, RL_0)
        RL10_df.loc[i, 'return_levels_b0'] = interp_func(10)
    
    if np.size(RL_5) == 1:
        RL10_df.loc[i, 'return_levels_5'] = np.nan
    else:
        interp_func = interp1d(eRP, RL_5)
        RL10_df.loc[i, 'return_levels_5'] = interp_func(10)
    





# Define the boundaries and number of bins for the discrete colormap
num_bins = 12
cmap = plt.cm.rainbow  # You can use 'rainbow' or any other colormap
norm = mcolors.BoundaryNorm(boundaries=np.linspace(10, 70, num_bins + 1), ncolors=num_bins)

# Create a discrete colormap and add black for values above 70
colors = cmap(np.linspace(0, 1, num_bins))
colors = np.vstack([colors, [0, 0, 0, 1]])  # Add black as the last color
discrete_cmap = mcolors.ListedColormap(colors)
# Create the figure
fig = plt.figure(figsize=(10, 10))

proj = ccrs.PlateCarree()

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL10_df.return_levels,
    cmap=discrete_cmap,
    norm=norm,
    s=s,
)
ax1.set_xticks(np.arange(lon_lims[0], lon_lims[1] + 1, 5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0], lat_lims[1] + 1, 2.5), crs=proj)
ax1.tick_params(labelsize=12)
plt.xlim(lon_lims[0] - 1, lon_lims[1] + 1)
plt.ylim(lat_lims[0] - 1, lat_lims[1] + 1)
ax1.set_title("free")

# Second subplot
ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL10_df.return_levels_b0,
    cmap=discrete_cmap,
    norm=norm,
    s=s,
)
ax2.set_xticks(np.arange(lon_lims[0], lon_lims[1] + 1, 5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0], lat_lims[1] + 1, 2.5), crs=proj)
ax2.tick_params(labelsize=12)
plt.xlim(lon_lims[0] - 1, lon_lims[1] + 1)
plt.ylim(lat_lims[0] - 1, lat_lims[1] + 1)
ax2.set_title("b = 0")

# Third subplot
ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = RL10_df.return_levels_5,
    cmap=discrete_cmap,
    norm = norm,
    s=s,
)
ax3.set_xticks(np.arange(lon_lims[0], lon_lims[1] + 1, 5), crs=proj)
ax3.set_yticks(np.arange(lat_lims[0], lat_lims[1] + 1, 2.5), crs=proj)
ax3.tick_params(labelsize=12)
plt.xlim(lon_lims[0] - 1, lon_lims[1] + 1)
plt.ylim(lat_lims[0] - 1, lat_lims[1] + 1)
ax3.set_title("5% sig")

# Fourth subplot
ax4 = fig.add_subplot(2, 2, 4, projection=proj)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS, linestyle=':')
sc4 = ax4.scatter(
    val_info.longitude,
    val_info.latitude,
    c=val_info.cleaned_years,
    cmap="viridis",  # You can use discrete colormap here if desired
    s=s,
)
ax4.set_xticks(np.arange(lon_lims[0], lon_lims[1] + 1, 5), crs=proj)
ax4.set_yticks(np.arange(lat_lims[0], lat_lims[1] + 1, 2.5), crs=proj)
ax4.tick_params(labelsize=12)
plt.xlim(lon_lims[0] - 1, lon_lims[1] + 1)
plt.ylim(lat_lims[0] - 1, lat_lims[1] + 1)
fig.subplots_adjust(right=0.85)

# Colorbar for the fourth subplot
cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])
cb4 = plt.colorbar(sc4, cax=cbar_ax4)
cb4.set_label('Number of complete years', fontsize=14)
cb4.ax.tick_params(labelsize=12)
ax4.set_title("cleaned years")

# Colorbar for the first three subplots
cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
#cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
#cb = mcolors.colorbarbase(sc,cax=cbar_ax, cmap=discrete_cmap, norm=norm, orientation='horizontal')
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=discrete_cmap),
    cax=cbar_ax,
    orientation='horizontal',
    extend='max'
)

cb.set_label('10 year 1 hour return level (mm)', fontsize=14)
cb.ax.tick_params(labelsize=12)

fig.suptitle(f'{ERA_country} 10 year return levels. beta = {S.beta}', fontsize=16)
plt.show()



############################################################################
#SCATTERS/CORRELATIONS

fig = plt.figure()
s= 10

ax1 = fig.add_subplot(2,2,1)
ax1.scatter(val_info.cleaned_years,FRMSE_df.FRMSE, s=s)
ax1.set_xlabel("years")
ax1.set_ylabel("FRMSE")
ax1.plot()


ax2 = fig.add_subplot(2,2,2)
ax2.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),"--k",alpha = 0.4)
ax2.scatter(FRMSE_df.FRMSE,FRMSE_df.FRMSE_0, s=s,alpha = 0.5)

diff_all_0 = FRMSE_df.FRMSE_0 -FRMSE_df.FRMSE
n_above = diff_all_0[diff_all_0>0.01].count()
n_below = diff_all_0[diff_all_0<-0.01].count()

ax2.set_xlabel(f"free ({n_below} more than 0.01 difference)")
ax2.set_ylabel(f"b = 0 ({n_above})")
ax2.set_xlim(0,0.6)
ax2.set_ylim(0,0.6)


ax3 = fig.add_subplot(2,2,3)
ax3.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),"--k",alpha = 0.4)
ax3.scatter(FRMSE_df.FRMSE,FRMSE_df.FRMSE_5, s=s,alpha = 0.5)

diff_all_5 = FRMSE_df.FRMSE_5 -FRMSE_df.FRMSE
n_above = diff_all_5[diff_all_5>0.01].count()
n_below = diff_all_5[diff_all_5<-0.01].count()

ax3.set_xlabel(f"free ({n_below})")
ax3.set_ylabel(f"b = 5% sig ({n_above})")
ax3.set_xlim(0,0.6)
ax3.set_ylim(0,0.6)


ax4 = fig.add_subplot(2,2,4)
ax4.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),"--k",alpha = 0.4)
ax4.scatter(FRMSE_df.FRMSE_0,FRMSE_df.FRMSE_5, s=s,alpha = 0.5)

diff_0_5 = FRMSE_df.FRMSE_5 -FRMSE_df.FRMSE_0
n_above = diff_0_5[diff_0_5>0.01].count()
n_below = diff_0_5[diff_0_5<-0.01].count()

ax4.set_xlabel(f"b = 0 ({n_below})")
ax4.set_ylabel(f"b = 5% sig ({n_above})")
ax4.set_xlim(0,0.6)
ax4.set_ylim(0,0.6)

fig.suptitle(f"{country}")
fig.tight_layout()
plt.show()



plt.scatter(FRMSE_df.FRMSE,liklihood_df.ave_prob)
plt.xlabel("FRMSE")
plt.ylabel("average prob")
plt.show()

###############################################################################
#Box plots of errors
if "FRMSE_bset" in FRMSE_df.columns:
    plt.boxplot([FRMSE_df.FRMSE.copy().dropna(),FRMSE_df.FRMSE_0.copy().dropna(),FRMSE_df.FRMSE_bset.copy().dropna()],vert=False)
    plt.xlabel('FRMSE')
    #plt.xlim(-0.1,0.2)
    plt.yticks([1,2,3],['free','b = 0','b set'])
    plt.title(f'{ERA_country} FRMSE. beta = {S.beta}')
    plt.show()
    
    plt.boxplot([liklihood_df.ave_prob.copy().dropna(),liklihood_df.ave_prob_0.copy().dropna(),liklihood_df.ave_prob_bset.copy().dropna()],vert=False)
    plt.xlabel('Average probability')
    #plt.xlim(-0.1,0.2)
    plt.yticks([1,2,3],['free','b = 0','b set'])
    plt.title(f'{ERA_country} average probability. beta = {S.beta}')
    plt.show()
    
    plt.boxplot([
        np.log(liklihood_df.copy().dropna().mult_prob_bexp) - np.log(liklihood_df.copy().dropna().mult_prob_0),
        np.log(liklihood_df.copy().dropna().mult_prob) - np.log(liklihood_df.copy().dropna().mult_prob_0),
        np.log(liklihood_df.copy().dropna().mult_prob_bset) - np.log(liklihood_df.copy().dropna().mult_prob_0)],
        vert=False)
    
    plt.xlabel('log products')
    #plt.xlim(-0.1,0.2)
    plt.yticks([1,2,3],['bexp - b 0','b free - b 0','bset - b 0'])
    plt.title(f'{ERA_country} log products difference. beta = {S.beta}')
    plt.grid()
    plt.show()
elif "FRMSE_bexp" in FRMSE_df.columns:
    
    plt.boxplot([
        (np.log(liklihood_df.copy().dropna().mult_prob_bexp) - np.log(liklihood_df.copy().dropna().mult_prob_0)).copy().dropna(),
        (np.log(liklihood_df.copy().dropna().mult_prob) - np.log(liklihood_df.copy().dropna().mult_prob_0)).copy().dropna()],
        vert=False)
    
    plt.xlabel('log products')
    #plt.xlim(-0.1,0.2)
    plt.yticks([1,2],['bexp - b 0','b free - b 0'])
    plt.title(f'{ERA_country} log products difference. beta = {S.beta}')
    plt.grid()
    plt.show()
    
    plt.boxplot([
        (np.log(liklihood_df.copy().dropna().mult_prob_bexp) - np.log(liklihood_df.copy().dropna().mult_prob_0)).copy().dropna(),
        (np.log(liklihood_df.copy().dropna().mult_prob) - np.log(liklihood_df.copy().dropna().mult_prob_0)).copy().dropna()],
        vert=False)
    
    lw_q = (np.log(liklihood_df.copy().dropna().mult_prob_bexp) - np.log(liklihood_df.copy().dropna().mult_prob_0)).quantile(0.25)
    
    plt.xlabel('log products')
    plt.xlim(lw_q*3,-lw_q*3)
    plt.yticks([1,2],['bexp - b 0','b free - b 0'])
    plt.title(f'{ERA_country} log products difference. beta = {S.beta}')
    plt.grid()
    plt.show()

else:
    plt.boxplot([FRMSE_df.FRMSE.copy().dropna(),FRMSE_df.FRMSE_0.copy().dropna(),FRMSE_df.FRMSE_5.copy().dropna()],vert=False)
    plt.xlabel('FRMSE')
    #plt.xlim(-0.1,0.2)
    plt.yticks([1,2,3],['free','b = 0','5% sig'])
    plt.title(f'{ERA_country} FRMSE. beta = {S.beta}')
    plt.show()
    
    plt.boxplot([liklihood_df.ave_prob.copy().dropna(),liklihood_df.ave_prob_0.copy().dropna(),liklihood_df.ave_prob_5.copy().dropna()],vert=False)
    plt.xlabel('Average probability')
    #plt.xlim(-0.1,0.2)
    plt.yticks([1,2,3],['free','b = 0','5% sig'])
    plt.title(f'{ERA_country} average probability. beta = {S.beta}')
    plt.show()


    




# CHECKS
for j in np.arange(0,5):
    plot_pos = np.arange(1,np.size(RL_df.obs_AMS.iloc[j])+1)/(1+np.size(RL_df.obs_AMS.iloc[j]))
    
    eRP = 1/(1-plot_pos)
    
    TNX_FIG_valid(RL_df.obs_AMS.iloc[j],eRP,RL_df.return_levels_b0.iloc[j],TENAXlabel = f"b = 0. FRMSE: {FRMSE_df.FRMSE_0.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob_0.iloc[j]):.1f}",obslabel='AMS')
    plt.plot(eRP,RL_df.return_levels.iloc[j],"r",label = f"b = free. FRMSE: {FRMSE_df.FRMSE.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob.iloc[j]):.1f}")
    #plt.plot(eRP,RL_df.return_levels_5.iloc[j],"g",alpha = 0.5, label = "b = 5% sig")
    if "return_levels_bset" in RL_df.columns:
        plt.plot(eRP,RL_df.return_levels_bset.iloc[j],"y", label = f"b = mean. FRMSE: {FRMSE_df.FRMSE_bset.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob_bset.iloc[j]):.1f}")
    if "return_levels_bexp" in RL_df.columns:
        plt.plot(eRP,RL_df.return_levels_bexp.iloc[j],"m", label = f"b exp. FRMSE: {FRMSE_df.FRMSE_bexp.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob_bexp.iloc[j]):.1f}")
    if "return_levels_roll" in RL_df.columns:
        plt.plot(eRP,RL_df.return_levels_roll.iloc[j],"g", label = f"b rolling average. FRMSE: {FRMSE_df.FRMSE_roll.iloc[j]:.3f}. log: {np.log(liklihood_df.mult_prob_roll.iloc[j]):.1f}")
    
        
    plt.fill_between(eRP,liklihood_df.mins_0.iloc[j],liklihood_df.maxes_0.iloc[j],color = "b", alpha = 0.1)
    plt.fill_between(eRP,liklihood_df.mins.iloc[j],liklihood_df.maxes.iloc[j],color = "r", alpha = 0.1)
    
    plt.ylim(0,np.max(RL_df.return_levels.iloc[j])+5)
    plt.xlim(1,np.max(eRP)+2)
    
    plt.legend()
    if "df_parameters_bset" in locals():
        plt.title(f"station {j}: {FRMSE_df.station.iloc[j]}.b mean = {df_parameters_bset.b.iloc[0]:.3f}")
    else:
        plt.title(f"station {j}: {FRMSE_df.station.iloc[j]}.")
    plt.show()
    
    

