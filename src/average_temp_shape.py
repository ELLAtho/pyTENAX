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
from scipy.signal import find_peaks

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
from matplotlib import colormaps
from matplotlib.colors import to_rgba



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
# max_lat = 50
# region_lats = [minlat,51,maxlat]
# region_lons = [minlon,9,maxlon]


# country = 'Japan'
# ERA_country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP_'
# minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 30
# region_lats = [minlat,31,35.8,41.3,maxlat]
# region_lons = [minlon,maxlon]



country = 'US' 
ERA_country = 'US'
country_save = 'US_main'
code_str = 'US_'
minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
name_len = 6
min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
censor_thr = 0.9
max_lat = 30
region_lats = [minlat,37.5,maxlat]
region_lons = [minlon,-116,-105,-90,maxlon]

# country = 'UK' 
# ERA_country = 'UK'
# country_save = 'UK'
# code_str = 'UK_'
# name_len = 0
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9
# max_lat = 50



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



average_filename = f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv"
if average_filename not in output_files:
    print("aves not calculated")
    eTs = [0] * len(new_df)
    aves = [0] * len(new_df)
    sds = [0] * len(new_df)
    
    for i in np.arange(0, len(new_df)):
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            eTs[i] = [np.nan]*1000
            aves[i] = np.nan
            sds[i] = np.nan
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            
            aves[i] = np.mean(T)
            sds[i] = np.std(T)
            eT_sep = (np.max(T) - np.min(T)+8)/1000
            eTs[i] = np.arange(np.min(T)-4,np.max(T)+4,eT_sep)
            
            if len(eTs[i]) != 1000:
                eTs[i] = eTs[i][0:1000]
    
    average_df = pd.DataFrame({
        "station":df.station,
        "aves":aves,
        "sds": sds
        })
    eTs_df = pd.DataFrame(eTs)
    eTs_df["station"] = df.station
    
    
    eTs_df.to_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv", index=False)
    average_df.to_csv(average_filename, index=False)
    
else:
    print("reading files")
    eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\eTs_df.csv",dtype = {"station":str})
    average_df = pd.read_csv(average_filename,dtype = {"station":str})
    aves = average_df.aves.to_numpy()
    sds = average_df.sds.to_numpy()
    eTs = eTs_df.drop(columns = "station").to_numpy()
    




################################################################################♦
#interping for shifts
temp_aves = df.drop(columns = "station").mean(axis = 0)

x_vals = np.arange(-0.5,0.5,1/1000)
ymax = np.nanmax(df.drop(columns = "station"))

xmin = np.nanmin(eTs)
xmax = np.nanmax(eTs)



interp_x = np.arange(-4,4.1,0.1)
interp_y = [np.nan] * len(df_parameters)
for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs[i]-aves[i])/sds[i],df.iloc[i][1:])
        interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs[i]-aves[i])/sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs[i]-aves[i])/sds[i])]
        
        interp_y[i][(interp_x>=np.min((eTs[i]-aves[i])/sds[i])) & (interp_x<=np.max((eTs[i]-aves[i])/sds[i]))] = interp_func(interp_x_here)*sds[i]
        



ymax2 = np.nanmax(interp_y)+0.05

temp_aves_proper = np.nanmean(interp_y,axis =0)

df_south = df[df_parameters.reset_index().latitude <= max_lat]
temp_aves_south = df_south.drop(columns = "station").mean(axis = 0)
eTs = np.array(eTs)
eTs_south = eTs[df_parameters.reset_index().latitude <= max_lat]

sds_south = np.array(sds)[df_parameters.reset_index().latitude <= max_lat]
aves_south = np.array(aves)[df_parameters.reset_index().latitude <= max_lat]
interp_y_south  = [np.nan] * len(df_south)

for i in np.arange(0,len(df_south)):  
    if np.isnan(aves_south[i]):
        interp_y_south[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs_south[i]-aves_south[i])/sds_south[i],df_south.iloc[i][1:])
        interp_y_south[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs_south[i]-aves_south[i])/sds_south[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs_south[i]-aves_south[i])/sds_south[i])]
        
        interp_y_south[i][(interp_x>=np.min((eTs_south[i]-aves_south[i])/sds_south[i])) & (interp_x<=np.max((eTs_south[i]-aves_south[i])/sds_south[i]))] = interp_func(interp_x_here)*sds_south[i]
        

temp_aves_proper_south = np.nanmean(interp_y_south,axis =0)




df_north = df[df_parameters.reset_index().latitude > max_lat]
temp_aves_north = df_north.drop(columns = "station").mean(axis = 0)
eTs = np.array(eTs)
eTs_north = eTs[df_parameters.reset_index().latitude > max_lat]


sds_north= np.array(sds)[df_parameters.reset_index().latitude > max_lat]
aves_north= np.array(aves)[df_parameters.reset_index().latitude > max_lat]
interp_y_north = [np.nan] * len(df_north)

for i in np.arange(0,len(df_north)):  
    if np.isnan(aves_north[i]):
        interp_y_north[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((eTs_north[i]-aves_north[i])/sds_north[i],df_north.iloc[i][1:])
        interp_y_north[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((eTs_north[i]-aves_north[i])/sds_north[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((eTs_north[i]-aves_north[i])/sds_north[i])]
        
        interp_y_north[i][(interp_x>=np.min((eTs_north[i]-aves_north[i])/sds_north[i])) & (interp_x<=np.max((eTs_north[i]-aves_north[i])/sds_north[i]))] = interp_func(interp_x_here)*sds_north[i]
        

temp_aves_proper_north= np.nanmean(interp_y_north,axis =0)


################################################################################
#total average temperature shape
non_event_temp_savename = f"{drive}:/outputs/{country_save}\\non_event_temp.csv"
if non_event_temp_savename not in output_files:
    print("temp average shape not calculated yet. here we gooooooo")
    
    kde = [0] * len(new_df)
    prob = [0] * len(new_df)
    start_time = [0] * len(new_df)
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time()
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        
        
        T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
        
        if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
            print('skip')
            t_data = [np.nan]
        else:
            T_ERA = xr.load_dataarray(T_path)
            t_data = (T_ERA.squeeze()-273.15).to_dataframe()
            t = t_data["t2m"]
            
        if len(t_data) <= 2:
            prob[i] = [np.nan]*1000
        
        else:
            eT_sep = (np.max(t) - np.min(t)+8)/1000
            eT = np.arange(np.min(t)-4,np.max(t)+4,eT_sep)
            
            if len(eT) != 1000:
                eT = eT[0:1000]
            
            kde[i]  = gaussian_kde(t) #use kernel density to get probability
            prob[i] = kde[i](eT)
                
                
                  
            if i%50 == 0:    
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
            else:
                pass
    non_event_temp = np.array(prob)
    non_event_temp_ = np.concatenate([df_parameters.station.to_numpy()[:, np.newaxis], non_event_temp], axis=1)
    
    df_nonevent = pd.DataFrame(non_event_temp_)
    df_nonevent.rename(columns = {0:"station"},inplace = True)
    df_nonevent.to_csv(non_event_temp_savename, index=False)
    
else:
    df_nonevent = pd.read_csv(non_event_temp_savename,dtype = {0:str})

average_filename = f"{drive}:/outputs/{country_save}\\average_temp_shape_ave_std.csv"

non_event_average_savename = f"{drive}:/outputs/{country_save}\\average_non_event_temp.csv"
if non_event_average_savename not in output_files:
    print("aves not calculated non events")
    non_event_eTs = [0] * len(new_df)
    non_event_aves = [0] * len(new_df)
    non_event_sds = [0] * len(new_df)
    start_time = [0] * len(new_df)
    
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time()
        file_name = f"{drive}:/{country}/{code_str}{df_parameters.station.iloc[i]}"
        
        
        T_path = f"{drive}:/{country}_temp\\{code_str}{df_parameters.station.iloc[i]}.nc" #TODO: nans case (not there in germany)
        
        if T_path not in glob.glob(f"{drive}:/{country}_temp\\*"): # dont do tenax if no T data saved
          
            non_event_eTs[i] = [np.nan]*1000
            non_event_aves[i] = np.nan
            non_event_sds[i] = np.nan
        else:
            T_ERA = xr.load_dataarray(T_path)
            t_data = (T_ERA.squeeze()-273.15).to_dataframe()
            T = t_data["t2m"]
            
            non_event_aves[i] = np.mean(T)
            non_event_sds[i] = np.std(T)
            non_event_eT_sep = (np.max(T) - np.min(T)+8)/1000
            non_event_eTs[i] = np.arange(np.min(T)-4,np.max(T)+4,non_event_eT_sep)
            
            if len(non_event_eTs[i]) != 1000:
                non_event_eTs[i] = non_event_eTs[i][0:1000]
                
                
            if i%50 == 0:    
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
            else:
                pass
    
    non_event_average_df = pd.DataFrame({
        "station":df.station,
        "aves":non_event_aves,
        "sds": non_event_sds
        })
    non_event_eTs_df = pd.DataFrame(non_event_eTs)
    non_event_eTs_df["station"] = df.station
    
    
    non_event_eTs_df.to_csv(f"{drive}:/outputs/{country_save}\\non_event_eTs_df.csv", index=False)
    non_event_average_df.to_csv(non_event_average_savename, index=False)
    
else:
    print("reading files")
    non_event_eTs_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\non_event_eTs_df.csv",dtype = {"station":str})
    non_event_average_df = pd.read_csv(non_event_average_savename,dtype = {"station":str})
    non_event_aves = non_event_average_df.aves.to_numpy()
    non_event_sds = non_event_average_df.sds.to_numpy()
    non_event_eTs = non_event_eTs_df.drop(columns = "station").to_numpy()


non_event_interp_y = [np.nan] * len(df_parameters)
for i in np.arange(0,len(df_parameters)):  
    if np.isnan(non_event_aves[i]):
        non_event_interp_y[i] = [np.nan]*len(interp_x)
    else:  
        interp_func = interp1d((non_event_eTs[i]-non_event_aves[i])/non_event_sds[i],df_nonevent.iloc[i][1:])
        non_event_interp_y[i] = np.zeros(len(interp_x))
        interp_x_here = interp_x[interp_x>=np.min((non_event_eTs[i]-non_event_aves[i])/non_event_sds[i])]
        interp_x_here = interp_x_here[interp_x_here<=np.max((non_event_eTs[i]-non_event_aves[i])/non_event_sds[i])]
        
        non_event_interp_y[i][(interp_x>=np.min((non_event_eTs[i]-non_event_aves[i])/non_event_sds[i])) & (interp_x<=np.max((non_event_eTs[i]-non_event_aves[i])/non_event_sds[i]))] = interp_func(interp_x_here)*non_event_sds[i]
        

###############################################################################
#plots
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(3,3,1)

for i in np.arange(0,len(df_parameters)):    
    if np.isnan(aves[i]):
        pass
    else:    
        ax1.plot(eTs[i] ,df.iloc[i][1:],alpha = 0.01,color = "b")

plt.ylim(0,ymax)
plt.xlim(-30,40)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("Temperature (°C)")




ax2 = fig.add_subplot(3,3,2)
for i in np.arange(0,len(df_north)):    
    plt.plot(eTs_north[i] ,df_north.iloc[i][1:],alpha = 0.01,color = "b")


plt.ylim(0,ymax)
plt.xlim(-30,40)
plt.title(f"Average temperature distribution above {max_lat} °")
plt.xlabel("Temperature (°C)")


ax3 = fig.add_subplot(3,3,3)
for i in np.arange(0,len(df_south)):    
    plt.plot(eTs_south[i],df_south.iloc[i][1:],alpha = 0.01,color = "b")

plt.ylim(0,ymax)
plt.xlim(xmin,xmax)
plt.title(f"Average temperature distribution {country_save} below {max_lat} °")
plt.xlabel("Temperature (°C)")

ax4 = fig.add_subplot(3,3,4)
for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax4.plot(eTs[i]-aves[i] ,df.iloc[i][1:],alpha = 0.01,color = "b")


plt.ylim(0,ymax)
plt.xlim(xmin,xmax)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("Temperature (°C) - mean")


ax5 = fig.add_subplot(3,3,5)
for i in np.arange(0,len(df_north)):  
    if np.isnan(aves_north[i]):
        pass
    else:    
        ax5.plot(eTs_north[i]-aves_north[i] ,df_north.iloc[i][1:],alpha = 0.01,color = "b")


plt.ylim(0,ymax)
plt.xlim(xmin,xmax)
plt.title(f"Temperature distributions above {max_lat} °")
plt.xlabel("Temperature (°C) - mean")

ax6 = fig.add_subplot(3,3,6)
for i in np.arange(0,len(df_south)):  
    if np.isnan(aves_south[i]):
        pass
    else:    
        ax6.plot(eTs_south[i]-aves_south[i] ,df_south.iloc[i][1:],alpha = 0.01,color = "b")


plt.ylim(0,ymax)
plt.xlim(xmin,xmax)
plt.title(f"Temperature distributions below {max_lat} °")
plt.xlabel("Temperature (°C) - mean")


ax7 = fig.add_subplot(3,3,7)

for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax7.plot(interp_x ,interp_y[i],alpha = 0.01,color = "b")

plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])

plt.ylim(0,ymax2)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("(Temperature - mean)/std")


ax8 = fig.add_subplot(3,3,8)

for i in np.arange(0,len(df_north)):  
    if np.isnan(aves_north[i]):
        pass
    else:    
        ax8.plot(interp_x ,interp_y_north[i],alpha = 0.01,color = "b")

plt.plot(interp_x,temp_aves_proper_north,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])

plt.ylim(0,ymax2)
plt.title(f"Temperature distributions  above {max_lat} °")
plt.xlabel("(Temperature - mean)/std")

ax9 = fig.add_subplot(3,3,9)
for i in np.arange(0,len(df_south)):  
    if np.isnan(aves_south[i]):
        pass
    else:    
        ax9.plot(interp_x ,interp_y_south[i],alpha = 0.01,color = "b")

plt.plot(interp_x,temp_aves_proper_south,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])

plt.ylim(0,ymax2)
plt.title(f"Temperature distributions below {max_lat} °")
plt.xlabel("(Temperature - mean)/std")



plt.show()

# PLOT WITH LAT RAINBOW
df_boundaries = [np.min(df_parameters.latitude),np.min(df_parameters.longitude),np.max(df_parameters.latitude),np.max(df_parameters.longitude)]


normed_lats = (df_parameters.reset_index().latitude - df_boundaries[0])/(df_boundaries[2]-df_boundaries[0])
normed_lons = (df_parameters.reset_index().longitude - df_boundaries[1])/(df_boundaries[3]-df_boundaries[1])



cmap = colormaps['jet']


x = np.arange(df_boundaries[1],df_boundaries[3],0.1)
y = np.arange(df_boundaries[0],df_boundaries[2],0.1)
y_norm = (y - df_boundaries[0])/(df_boundaries[2]-df_boundaries[0])
x_norm = (x - df_boundaries[1])/(df_boundaries[3]-df_boundaries[1])

fig = plt.figure(figsize=(12, 12))

proj = ccrs.PlateCarree()

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


plt.contourf(x,
             y,
             np.transpose([y_norm]*len(x)), cmap = cmap
             )



ax2 = fig.add_subplot(2, 2, 2)

for i in np.arange(0,len(df_parameters)):    
    if np.isnan(aves[i]):
        pass
    else:    
        plt.plot(eTs[i],df.iloc[i][1:],alpha = 0.01,color = cmap(normed_lats[i]))

plt.ylim(0,ymax)
plt.xlim(-30,40)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("Temperature (°C)")


ax3 = fig.add_subplot(2,2,3)

for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax3.plot(interp_x ,interp_y[i],alpha = 0.01,color =  cmap(normed_lats[i]))

#plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])

plt.ylim(0,ymax2)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("(Temperature - mean)/std")



plt.show()


# PLOT WITH LON RAINBOW
cmap = colormaps['jet']


x = np.arange(df_boundaries[1],df_boundaries[3],0.1)
y = np.arange(df_boundaries[0],df_boundaries[2],0.1)
y_norm = (y - df_boundaries[0])/(df_boundaries[2]-df_boundaries[0])
x_norm = (x - df_boundaries[1])/(df_boundaries[3]-df_boundaries[1])

fig = plt.figure(figsize=(12, 12))

proj = ccrs.PlateCarree()

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


plt.contourf(x,
             y,
             [x_norm]*len(y), cmap = cmap
             )



ax2 = fig.add_subplot(2, 2, 2)

for i in np.arange(0,len(df_parameters)):    
    if np.isnan(aves[i]):
        pass
    else:    
        plt.plot(eTs[i],df.iloc[i][1:],alpha = 0.01,color = cmap(normed_lons[i]))

plt.ylim(0,ymax)
plt.xlim(-30,40)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("Temperature (°C)")


ax3 = fig.add_subplot(2,2,3)

for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax3.plot(interp_x ,interp_y[i],alpha = 0.01,color =  cmap(normed_lons[i]))

#plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])
plt.ylim(0,ymax2)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("(Temperature - mean)/std")



plt.show()



# PLOT WITH FULL COLOR GRADIENT
colors = [to_rgba((lat, lon, 0.7, 1)) for lat, lon in zip(normed_lats, normed_lons)]


fig = plt.figure(figsize=(12, 12))

proj = ccrs.PlateCarree()

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


plt.scatter(df_parameters.longitude,
             df_parameters.latitude,
             color = colors
             )



ax2 = fig.add_subplot(2, 2, 2)

for i in np.arange(0,len(df_parameters)):    
    if np.isnan(aves[i]):
        pass
    else:    
        plt.plot(eTs[i],df.iloc[i][1:],alpha = 0.01,color = colors[i])

plt.ylim(0,ymax)
plt.xlim(-30,40)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("Temperature (°C)")


ax3 = fig.add_subplot(2,2,3)

for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax3.plot(interp_x ,interp_y[i],alpha = 0.01,color =  colors[i])

plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])
plt.ylim(0,ymax2)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("(Temperature - mean)/std")



plt.show()



colors = [to_rgba((lat, lon, 0.5, 1)) for lat, lon in zip(normed_lats, normed_lons)]


fig = plt.figure(figsize=(12, 12))

proj = ccrs.PlateCarree()

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


plt.scatter(df_parameters.longitude,
             df_parameters.latitude,
             color = colors
             )



ax2 = fig.add_subplot(2, 2, 2)

for i in np.arange(0,len(df_parameters)):    
    if np.isnan(aves[i]):
        pass
    else:    
        plt.plot(eTs[i],df.iloc[i][1:],alpha = 0.01,color = colors[i])

plt.plot(eTs[i],df.iloc[120][1:])
plt.ylim(0,ymax)
plt.xlim(-30,40)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("Temperature (°C)")


ax3 = fig.add_subplot(2,2,3)

for i in np.arange(0,len(df_parameters)):  
    if np.isnan(aves[i]):
        pass
    else:    
        ax3.plot(interp_x ,interp_y[i],alpha = 0.01,color =  colors[i])

plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])
plt.ylim(0,ymax2)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("(Temperature - mean)/std")



plt.show()

###############################################################################
#regional splits
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]



n_lat = len(region_lats)-1
n_lon = len(region_lons)-1

# show regions
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

for lat_i in range(n_lat-1):
    ax1.plot([minlon-3,maxlon+3],[region_lats[lat_i+1],region_lats[lat_i+1]],  'r', linewidth=2, transform=ccrs.PlateCarree())

for lon_i in range(n_lon-1):
    ax1.plot([region_lons[lon_i+1],region_lons[lon_i+1]],[minlat-3,maxlat+3],  'r', linewidth=2, transform=ccrs.PlateCarree())

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
plt.show()



fig,axs = plt.subplots(n_lat,n_lon,figsize = (n_lon*5,n_lat*5))
for lat_i in range(n_lat):
    for lon_i in range(n_lon):
        interp_y_region = np.array(interp_y)[(df_parameters.longitude<=region_lons[lon_i+1])
                                              & (df_parameters.longitude>region_lons[lon_i])
                                              & (df_parameters.latitude<=region_lats[lat_i+1])
                                              & (df_parameters.latitude>region_lats[lat_i])
                                             ]
        aves_region = aves[(df_parameters.longitude<=region_lons[lon_i+1])
                                              & (df_parameters.longitude>region_lons[lon_i])
                                              & (df_parameters.latitude<=region_lats[lat_i+1])
                                              & (df_parameters.latitude>region_lats[lat_i])
                                             ]
        if n_lon>1:
            for i in np.arange(0,len(interp_y_region)):  
                if np.isnan(aves_region[i]):
                    pass
                else:    
                    axs[n_lat-1-lat_i,lon_i].plot(interp_x ,interp_y_region[i],alpha = 0.01,color =  "b")
            axs[n_lat-1-lat_i,lon_i].plot(interp_x,np.nanmean(interp_y_region,axis=0),color = "r")
            axs[n_lat-1-lat_i,lon_i].set_title(f"latitude: {region_lats[lat_i]} to {region_lats[lat_i+1]}. longitude: {region_lons[lon_i]} to {region_lons[lon_i+1]} ")
        else:
            for i in np.arange(0,len(interp_y_region)):  
                if np.isnan(aves_region[i]):
                    pass
                else:    
                    axs[n_lat-1-lat_i].plot(interp_x ,interp_y_region[i],alpha = 0.01,color =  "b")
            axs[n_lat-1-lat_i].plot(interp_x,np.nanmean(interp_y_region,axis=0),color = "r")
            axs[n_lat-1-lat_i].set_title(f"latitude: {region_lats[lat_i]} to {region_lats[lat_i+1]}. longitude: {region_lons[lon_i]} to {region_lons[lon_i+1]} ")

plt.show()

#total temps


fig,axs = plt.subplots(n_lat,n_lon,figsize = (n_lon*5,n_lat*5))
for lat_i in range(n_lat):
    for lon_i in range(n_lon):
        interp_y_region = np.array(non_event_interp_y)[(df_parameters.longitude<=region_lons[lon_i+1])
                                              & (df_parameters.longitude>region_lons[lon_i])
                                              & (df_parameters.latitude<=region_lats[lat_i+1])
                                              & (df_parameters.latitude>region_lats[lat_i])
                                             ]
        aves_region = np.array(non_event_aves)[(df_parameters.longitude<=region_lons[lon_i+1])
                                              & (df_parameters.longitude>region_lons[lon_i])
                                              & (df_parameters.latitude<=region_lats[lat_i+1])
                                              & (df_parameters.latitude>region_lats[lat_i])
                                             ]
        if n_lon>1:
            for i in np.arange(0,len(interp_y_region)):  
                if np.isnan(aves_region[i]):
                    pass
                else:    
                    axs[n_lat-1-lat_i,lon_i].plot(interp_x ,interp_y_region[i],alpha = 0.01,color =  "b")
            axs[n_lat-1-lat_i,lon_i].plot(interp_x,np.nanmean(interp_y_region,axis=0),color = "r")
            axs[n_lat-1-lat_i,lon_i].set_title(f"latitude: {region_lats[lat_i]} to {region_lats[lat_i+1]}. longitude: {region_lons[lon_i]} to {region_lons[lon_i+1]} ")
        else:
            for i in np.arange(0,len(interp_y_region)):  
                if np.isnan(aves_region[i]):
                    pass
                else:    
                    axs[n_lat-1-lat_i].plot(interp_x ,interp_y_region[i],alpha = 0.01,color =  "b")
            axs[n_lat-1-lat_i].plot(interp_x,np.nanmean(interp_y_region,axis=0),color = "r")
            axs[n_lat-1-lat_i].set_title(f"latitude: {region_lats[lat_i]} to {region_lats[lat_i+1]}. longitude: {region_lons[lon_i]} to {region_lons[lon_i+1]} ")
plt.suptitle("Full temperature (not events)")
plt.show()

# PLOT WITH LON RAINBOW
cmap = colormaps['jet']


x = np.arange(df_boundaries[1],df_boundaries[3],0.1)
y = np.arange(df_boundaries[0],df_boundaries[2],0.1)
y_norm = (y - df_boundaries[0])/(df_boundaries[2]-df_boundaries[0])
x_norm = (x - df_boundaries[1])/(df_boundaries[3]-df_boundaries[1])

fig = plt.figure(figsize=(12, 12))

proj = ccrs.PlateCarree()

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection=proj)
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


plt.contourf(x,
             y,
             [x_norm]*len(y), cmap = cmap
             )



ax2 = fig.add_subplot(2, 2, 2)

for i in np.arange(0,len(df_parameters)):    
    if np.isnan(non_event_aves[i]):
        pass
    else:    
        plt.plot(non_event_eTs[i],df_nonevent.iloc[i][1:],alpha = 0.01,color = cmap(normed_lons[i]))

plt.plot(non_event_eTs[100],df_nonevent.iloc[100][1:])
plt.ylim(0,ymax)
plt.xlim(-30,40)
plt.title(f"Temperature distributions {country_save}")
plt.xlabel("Temperature (°C)")


ax3 = fig.add_subplot(2,2,3)

for i in np.arange(0,len(df_parameters)):  
    if np.isnan(non_event_aves[i]):
        pass
    else:    
        ax3.plot(interp_x ,non_event_interp_y[i],alpha = 0.01,color =  cmap(normed_lons[i]))

#plt.plot(interp_x,temp_aves_proper,label = "mean",color = "r")
#plt.plot(interp_x ,interp_y[1200],color = "r",label = df_parameters.station.iloc[3])

plt.ylim(0,ymax2)
plt.title(f"Temperature distributions non event {country_save}")
plt.xlabel("(Temperature - mean)/std")



plt.show()



###############################################################################4
#PEAK INVESTIGATION
x, y = eTs[84],df.iloc[84][1:]
height = 0
plt.plot(x,y)
peaks = find_peaks(y,height = height)
plt.scatter(x[peaks[0]],y[peaks[0]])
plt.show()

























