# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:39:33 2025

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
    new_df = new_df.drop(missing_rows.index)
else:
    pass




save_name = f"{drive}:/outputs/{country_save}\\average_temp_shape.csv"
output_files = glob.glob(f"{drive}:/outputs/{country_save}/*")
GOF_perc = 0.8
S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
        beta = 4
    )

if save_name not in output_files:
    print("temp average shape not calculated yet. here we gooooooo")
    

    kde = [0] * len(new_df)
    prob = [0] * len(new_df)
    start_time = [0] * len(new_df)
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time()
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
                
            
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
                T = [np.nan]
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
                
                #g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]] 
                
                
                # Your data (P, T arrays) and threshold thr=3.8
                P = dicts["60"]["ordinary"].to_numpy() 
                T = dicts["60"]["T"].to_numpy()  
                
                np.savetxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv",T)
                np.savetxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv",P)
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv")
            
        
        if len(T) <= 2:
            prob[i] = [np.nan]*1000
        
        else:
            eT_sep = (np.max(T) - np.min(T)+8)/1000
            eT = np.arange(np.min(T)-4,np.max(T)+4,eT_sep)
            
            if len(eT) != 1000:
                eT = eT[0:1000]
            
            kde[i]  = gaussian_kde(T) #use kernel density to get probability
            prob[i] = kde[i](eT)
            
            
              
        if i%50 == 0:    
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        else:
            pass
        
    temp_aves_full = np.array(prob)
    temp_aves_full_ = np.concatenate([df_parameters.station.to_numpy()[:, np.newaxis], temp_aves_full], axis=1)
    
    df = pd.DataFrame(temp_aves_full_)
    df.rename(columns = {0:"station"},inplace = True)
    df.to_csv(save_name, index=False)
    
    
    
else:
    df = pd.read_csv(save_name,dtype = {0:str})
        


temp_aves = df.drop(columns = "station").mean(axis = 0)

x_vals = np.arange(-0.5,0.5,1/1000)
ymax = np.max(df.drop(columns = "station"))


plt.plot(x_vals ,temp_aves)
for i in np.arange(0,len(df_parameters)):    
    plt.plot(x_vals ,df.iloc[i][1:],alpha = 0.01,color = "b")

#plt.plot(x_vals ,df.iloc[3][1:],color = "r",label = df_parameters.station.iloc[3])
plt.legend()
plt.ylim(0,ymax)
plt.title(f"Average temperature distribution {country_save}")
plt.show()


max_lat = 30

df_south = df[df_parameters.latitude <= max_lat]


temp_aves_south = df_south.drop(columns = "station").mean(axis = 0)
plt.plot(x_vals ,temp_aves_south)
for i in np.arange(0,len(df_south)):    
    plt.plot(x_vals ,df_south.iloc[i][1:],alpha = 0.01,color = "b")

plt.ylim(0,ymax)
plt.title(f"Average temperature distribution {country_save} below {max_lat} °")
plt.show()


df_north = df[df_parameters.latitude > max_lat]


temp_aves_north = df_north.drop(columns = "station").mean(axis = 0)
plt.plot(x_vals ,temp_aves_north)
for i in np.arange(0,len(df_north)):    
    plt.plot(x_vals ,df_north.iloc[i][1:],alpha = 0.01,color = "b")


plt.ylim(0,ymax)
plt.title(f"Average temperature distribution {country_save} above {max_lat} °")
plt.show()







