# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:52:34 2025

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
from scipy.stats import kendalltau, pearsonr, spearmanr, skewnorm



drive = 'D'
alpha_set = 0.05

#FOR BETA = 4, USE beta_set = ""
beta_set = 6
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

missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    df_parameters = df_parameters.reindex(index = range(len(df_parameters)))
    new_df = new_df.drop(missing_rows.index)
    new_df = new_df.reindex(index = range(len(new_df)))
else:
    pass




save_name = f"{drive}:/outputs/{country_save}\\temp_FRMSE{beta_set}.csv"
output_files = glob.glob(f"{drive}:/outputs/{country_save}/*")
GOF_perc = 0.8
S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
        beta = beta_set2
    )

if save_name not in output_files:
    print("temp FRMSE not calculated yet. here we gooooooo")
    

    FRMSE_upper_perc = [0] * len(new_df)
    start_time = [0] * len(new_df)
    FRMSE = [0] * len(new_df)
    diff = [0] * len(new_df)
    
    if beta_set == "":
        beta_set2 = 4
    else:
        beta_set2 = beta_set
    S = TENAX(
            return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
            durations = [60, 180, 360, 720, 1440],
            left_censoring = [0, 0.90],
            alpha = 0.05,
            min_ev_dur = 60,
            beta = beta_set2
        )
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time()
        
        
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            if type(df_parameters.station.iloc[i]) == float:
                T = [np.nan]
            else:
                
                if 'code_str' in locals():
                    G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
                else:
                    G = pd.read_csv(f"{file_name}.csv")
                    G['prec_time'] = pd.to_datetime(G['prec_time'])
                    G.set_index('prec_time', inplace=True)
                    
                    
                data = S.remove_incomplete_years(G, name_col)
                
                T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
                
                if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                    print('skip')
                    diff[i] = np.nan
                    FRMSE_upper_perc[i] = np.nan
                    FRMSE[i] = np.nan
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
                    _,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
                    
                    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
                    dict_ordinary, _ = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
                    
                    
                    
                    df_arr_t_data = np.array(t_data[temp_name_col])
                    df_dates_t_data = np.array(t_data.index)
                    
                    if type(df_dates_t_data[0]) != np.datetime64:
                            
                        df_dates_t_data = pd.Series([item[0] for item in df_dates_t_data])
                        df_dates_t_data = np.array(df_dates_t_data)
                    else:
                        pass
                    
                    dicts, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                    
                    
                    
                    # Your data (P, T arrays) and threshold thr=3.8
                    P = dicts["60"]["ordinary"].to_numpy() 
                    T = dicts["60"]["T"].to_numpy()  
                
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv")
        if len(T) <= 2:
            diff[i] = np.nan
            FRMSE_upper_perc[i] = np.nan
            FRMSE[i] = np.nan
        else:
            g_phat = S.temperature_model(T)
            min_T_upper = np.quantile(T,GOF_perc)
            
            eT = np.arange(np.min(T),np.max(T)+4,0.1)
            
            kde  = gaussian_kde(T) #use kernel density to get probability
            prob = kde(eT)
            pdf_values = gen_norm_pdf(eT, g_phat[0], g_phat[1], S.beta)
            
            
            diff[i] = pdf_values - prob
            FRMSE[i] = np.sqrt(
                np.sum(diff[i]**2)/len(diff[i]))/(np.sum(prob)/len(diff[i]))
            
            
            eT_upper_perc = eT[eT>=min_T_upper]
            
            prob_upper_perc = prob[eT>=min_T_upper]
            pdf_values_upper_perc = pdf_values[eT>=min_T_upper]
            diff_upper_perc = pdf_values_upper_perc - prob_upper_perc
            FRMSE_upper_perc[i] = np.sqrt(
                np.sum(diff_upper_perc**2)/len(diff_upper_perc))/(np.sum(prob_upper_perc)/len(diff_upper_perc))
                
              
        if i%50 == 0:    
            print(f"FRMSE {FRMSE[i]:.3f}. FRMSE upper {FRMSE_upper_perc[i]:.3f}")
            print(S.beta)
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
        
    temp_FRMSE_df = pd.DataFrame({
        'station': df_parameters.station,
        "FRMSE_upper_perc": FRMSE_upper_perc,
        "FRMSE": FRMSE,
        "difference": diff,
        })
    temp_FRMSE_df.to_csv(save_name,index = False)
    
else:
    temp_FRMSE_df = pd.read_csv(save_name,dtype = {"station":str})
        
        
##############################################################################
#skewnorm
skew_parameters_savename = f"{drive}:/outputs/{country_save}\\temp_skew.csv"
if skew_parameters_savename in output_files:
    print("skew calculated")
    df_parameters_skew = pd.read_csv(skew_parameters_savename,dtype = {"station":str})
    skew_FRMSE_savename = f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv"
    if skew_FRMSE_savename not in output_files:
        print("skewed FRMSE not calculated yet. calculating now")
        
        
        FRMSE_upper_perc = [0] * len(new_df)
        start_time = [0] * len(new_df)
        FRMSE = [0] * len(new_df)
        diff = [0] * len(new_df)
        
        for i in np.arange(0, len(new_df)):
            start_time[i] = time.time()
            g_phat = [df_parameters_skew.skewness.iloc[i],
                      df_parameters_skew.g_phat1.iloc[i],
                      df_parameters_skew.g_phat2.iloc[i]]
            
            
            oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
            if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
                  
                file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
                
                if 'code_str' in locals():
                    G,data_meta = read_GSDR_file(f"{file_name}.txt",name_col)
                else:
                    G = pd.read_csv(f"{file_name}.csv")
                    G['prec_time'] = pd.to_datetime(G['prec_time'])
                    G.set_index('prec_time', inplace=True)
                    
                    
                data = S.remove_incomplete_years(G, name_col)
                
                T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
                
                if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
                    print('skip')
                    diff[i] = np.nan
                    FRMSE_upper_perc[i] = np.nan
                    FRMSE[i] = np.nan
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
                    _,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)
                    
                    #assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
                    dict_ordinary, _ = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)
                    
                    
                    
                    df_arr_t_data = np.array(t_data[temp_name_col])
                    df_dates_t_data = np.array(t_data.index)
                    
                    if type(df_dates_t_data[0]) != np.datetime64:
                            
                        df_dates_t_data = pd.Series([item[0] for item in df_dates_t_data])
                        df_dates_t_data = np.array(df_dates_t_data)
                    else:
                        pass
                    
                    dicts, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)
                    
                    
                    # Your data (P, T arrays) and threshold thr=3.8
                    P = dicts["60"]["ordinary"].to_numpy() 
                    T = dicts["60"]["T"].to_numpy()  
                
            else:
                T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
                P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv")
                
            if len(T) <= 2:
                FRMSE_upper_perc[i] = np.nan
                start_time[i] = np.nan
                FRMSE[i] = np.nan
                diff[i] = np.nan
            else:
                
                min_T_upper = np.quantile(T,GOF_perc)
                
                eT = np.arange(np.min(T),np.max(T)+4,0.1)
                
                kde  = gaussian_kde(T) #use kernel density to get probability
                prob = kde(eT)
                pdf_values = skewnorm.pdf(eT, *g_phat)
                
                
                diff[i] = pdf_values - prob
                FRMSE[i] = np.sqrt(
                    np.sum(diff[i]**2)/len(diff[i]))/(np.sum(prob)/len(diff[i]))
                
                
                eT_upper_perc = eT[eT>=min_T_upper]
                
                prob_upper_perc = prob[eT>=min_T_upper]
                pdf_values_upper_perc = pdf_values[eT>=min_T_upper]
                diff_upper_perc = pdf_values_upper_perc - prob_upper_perc
                FRMSE_upper_perc[i] = np.sqrt(
                    np.sum(diff_upper_perc**2)/len(diff_upper_perc))/(np.sum(prob_upper_perc)/len(diff_upper_perc))
                    
                  
            if i%50 == 0:    
                print(f"FRMSE {FRMSE[i]:.3f}. FRMSE upper {FRMSE_upper_perc[i]:.3f}")
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                plt.plot(eT,prob,label = "observations")
                plt.plot(eT,pdf_values,label = "skew fit")
                plt.legend()
                plt.title(f"{df_parameters.station.iloc[i]}")
                plt.show()
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
            else:
                pass
            
        skew_FRMSE_df = pd.DataFrame({
            'station': df_parameters.station,
            "FRMSE_upper_perc": FRMSE_upper_perc,
            "FRMSE": FRMSE,
            "difference": diff,
            })
        skew_FRMSE_df.to_csv(skew_FRMSE_savename,index = False)
        
        
    else:
        skew_FRMSE_df = pd.read_csv(skew_FRMSE_savename,dtype = {"station":str})
    
    #plots
    lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
    lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
    s = 5


    fig = plt.figure(figsize=(10, 10))
    norm = mcolors.Normalize(vmin=np.min(skew_FRMSE_df.FRMSE), vmax=np.max(skew_FRMSE_df.FRMSE_upper_perc))
    cmap = 'plasma_r'


    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(2, 1, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax1.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c = skew_FRMSE_df.FRMSE,
        cmap=cmap,
        norm = mcolors.Normalize(vmin=np.min(skew_FRMSE_df.FRMSE), vmax=np.max(skew_FRMSE_df.FRMSE)),
        s = s,
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax1.set_title(f"FRMSE skew fit")
    plt.colorbar(sc)


    ax2 = fig.add_subplot(2, 1, 2, projection=proj)
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax2.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c=skew_FRMSE_df.FRMSE_upper_perc,
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax2.tick_params(labelsize=12)  
    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax2.set_title("FRMSE upper_perc")
    plt.colorbar(sc)

    plt.show()
    
    
    fig = plt.figure(figsize=(15, 5))
    norm = mcolors.Normalize(vmin=-0.3, vmax=0.3)
    cmap = 'seismic'


    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax1.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c = skew_FRMSE_df.FRMSE - temp_FRMSE_df.FRMSE,
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax1.set_title(f"skew - beta = 4")
    plt.colorbar(sc,extend = "both")
    
    
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax2.scatter(
        df_parameters.longitude,
        df_parameters.latitude,
        c = skew_FRMSE_df.FRMSE_upper_perc - temp_FRMSE_df.FRMSE_upper_perc,
        cmap=cmap,
        norm = norm,
        s = s,
    )
    ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax2.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax2.set_title(f"skew - beta = 4 upper 20%")
    plt.colorbar(sc,extend = "both")
    plt.show()
    
else:
    print("Can't do FRMSE on skew, calculate the skew values first")


##############################################################################
#plots
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
s = 5


fig = plt.figure(figsize=(10, 10))
norm = mcolors.Normalize(vmin=np.min(temp_FRMSE_df.FRMSE), vmax=np.max(temp_FRMSE_df.FRMSE_upper_perc))
cmap = 'plasma_r'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c = temp_FRMSE_df.FRMSE,
    cmap=cmap,
    norm = mcolors.Normalize(vmin=np.min(temp_FRMSE_df.FRMSE), vmax=np.max(temp_FRMSE_df.FRMSE)),
    s = s,
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title(f"FRMSE beta = {S.beta}")
plt.colorbar(sc)


ax2 = fig.add_subplot(2, 1, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=temp_FRMSE_df.FRMSE_upper_perc,
    cmap=cmap,
    norm = norm,
    s = s,
)
ax2.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax2.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax2.tick_params(labelsize=12)  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("FRMSE upper_perc")
plt.colorbar(sc)

plt.show()
###############################################################################
#correlations
FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE.csv", dtype={'station': str})
liklihood_df = pd.read_csv(f"{drive}:/outputs/{country_save}/liklihood.csv",dtype={'station': str})


def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]


r_val = FRMSE_df.FRMSE.corr(temp_FRMSE_df.FRMSE_upper_perc)
p_val = FRMSE_df.FRMSE.corr(temp_FRMSE_df.FRMSE_upper_perc,method=pearsonr_pval)

coeffs=np.polyfit(FRMSE_df.FRMSE.dropna(),temp_FRMSE_df.FRMSE_upper_perc.dropna(),1)
delt = (np.max(FRMSE_df.FRMSE)-np.min(FRMSE_df.FRMSE))/10
x = np.arange(np.min(FRMSE_df.FRMSE),np.max(FRMSE_df.FRMSE)+delt,delt)
y = coeffs[0]*x+coeffs[1]


plt.scatter(FRMSE_df.FRMSE,temp_FRMSE_df.FRMSE_upper_perc,alpha = val_info.cleaned_years/np.max(val_info.cleaned_years))
plt.plot(x,y,color = 'r',label = f'y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
plt.xlabel("FRMSE on the return levels")
plt.ylabel(f'FRMSE on the top {(1-GOF_perc)*100:.0f}% temperature')
plt.text(np.min(FRMSE_df.FRMSE),np.min(temp_FRMSE_df.FRMSE_upper_perc),f'r = {r_val:.3f}\n p = {p_val:.5f}')

plt.xlim(0,0.5)
plt.legend()
plt.show()





r_val = np.log(liklihood_df.mult_prob).corr(temp_FRMSE_df.FRMSE_upper_perc)
p_val = np.log(liklihood_df.mult_prob).corr(temp_FRMSE_df.FRMSE_upper_perc,method=pearsonr_pval)

coeffs=np.polyfit(np.log(liklihood_df.mult_prob).dropna(),temp_FRMSE_df.FRMSE_upper_perc.dropna(),1)
delt = (np.max(np.log(liklihood_df.mult_prob))-np.min(np.log(liklihood_df.mult_prob)))/10
x = np.arange(np.min(np.log(liklihood_df.mult_prob)),np.max(np.log(liklihood_df.mult_prob))+delt,delt)
y = coeffs[0]*x+coeffs[1]


plt.scatter(np.log(liklihood_df.mult_prob),temp_FRMSE_df.FRMSE_upper_perc,alpha = val_info.cleaned_years/np.max(val_info.cleaned_years))
plt.plot(x,y,color = 'r',label = f'y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
plt.xlabel("log prob")
plt.ylabel(f'FRMSE on the top {(1-GOF_perc)*100:.0f}% temperature')
plt.text(np.min(np.log(liklihood_df.mult_prob)),np.min(temp_FRMSE_df.FRMSE_upper_perc),f'r = {r_val:.3f}\n p = {p_val:.5f}')

plt.legend()
plt.show()


r_val = np.log(liklihood_df.mult_prob).corr(new_df.b)
p_val = np.log(liklihood_df.mult_prob).corr(new_df.b,method=pearsonr_pval)

coeffs=np.polyfit(np.log(liklihood_df.mult_prob).dropna(),new_df.b.dropna(),1)
delt = (np.max(np.log(liklihood_df.mult_prob))-np.min(np.log(liklihood_df.mult_prob)))/10
x = np.arange(np.min(np.log(liklihood_df.mult_prob)),np.max(np.log(liklihood_df.mult_prob))+delt,delt)
y = coeffs[0]*x+coeffs[1]


plt.scatter(np.log(liklihood_df.mult_prob),new_df.b,alpha = val_info.cleaned_years/np.max(val_info.cleaned_years))
plt.plot(x,y,color = 'r',label = f'y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
plt.xlabel("log prob")
plt.ylabel('b')
plt.text(np.min(np.log(liklihood_df.mult_prob)),np.min(new_df.b),f'r = {r_val:.3f}\n p = {p_val:.5f}')

plt.legend()
plt.show()






















