# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:13:59 2025

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
import kgcpy
import matplotlib.patches as mpatches


drive='D' #name of drive
countries = ['ISD','Belgium','Finland','Germany','Ireland','Japan','Norway','Portugal','UK','US']

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
country_index = countries.index(country)

# latslons
# n=0
# while n<len(countries):
#     info[n]['startdate'] = info[n]['startdate'].apply(lambda x: dt.datetime.strptime("{:10.0f}".format(x), "%Y%m%d%H"))
#     info[n]['enddate'] = info[n]['enddate'].apply(lambda x: dt.datetime.strptime("{:10.0f}".format(x), "%Y%m%d%H"))
#     info[n].to_csv('D:/metadata/'+countries[n]+'_fulldata.csv')
#     n=n+1
    

info = []

grouped_kgb = dict({"alpine":["ET","Dfc","Dsc","Dwc"], #Dwc added by me.. might not be here
                    "arid_cold":["BSk","BWk"],
                    "arid_hot": ["BSh","BWh"],
                    "humid_continental": ["Dfa","Dfb","Dwa","Dwb"],
                    "humid_subtropical": ["Cfa"],
                    "mediterranean":["Csa","Csb","Csc","Dsa","Dsb"],
                    "oceanic": ["Cfb","Cfc"],
                    "ocean": ["Ocean"],
                    "tropical": ["Af","Am","Aw","As","Cwa",]
                    }) #Cfc, Csc, Dwc, As added by me... could be wrong


inverse_mapping = {item: key for key, values in grouped_kgb.items() for item in values}

for c in countries:
    dd = pd.read_csv(drive+':/metadata/'+c+'_fulldata.csv', dtype={'station': str})
    #n_drop = len(dd.columns)-9
    #dd = dd.drop(dd.columns[0:n_drop], axis=1)
    dd["kgb_zone"] = dd.apply(lambda row: kgcpy.lookupCZ(row["latitude"], row["longitude"]), axis=1)
    dd['kgb_group'] = dd['kgb_zone'].map(inverse_mapping)
    #dd.to_csv(drive+':/metadata/'+c+'_fulldata.csv', index = False)
    info.append(dd)
    

colors_map = dict({"alpine":"grey", 
                    "arid_cold":"pink",
                    "arid_hot": "r",
                    "humid_continental": "c",
                    "humid_subtropical": "greenyellow",
                    "mediterranean":"yellow",
                    "oceanic": "g",
                    "ocean": "k",
                    "tropical": "b"
                    })


for i in np.arange(1,len(info)):
    
    curr_info = info[i]
    curr_info["color"] = curr_info['kgb_group'].map(colors_map)
    
    
    lon_lims = [truncate_neg(np.min(curr_info.longitude),2.5),np.ceil(np.max(curr_info.longitude/2.5))*2.5]
    lat_lims = [truncate_neg(np.min(curr_info.latitude),2.5),np.ceil(np.max(curr_info.latitude/2.5))*2.5]


    fig = plt.figure(figsize=(20, 20))
    norm = mcolors.Normalize(vmin=0, vmax=1)


    proj = ccrs.PlateCarree()
    ax1 = fig.add_subplot(1, 1, 1, projection=proj)

    # Add map features
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')


    sc = ax1.scatter(
        curr_info.longitude,
        curr_info.latitude,
        c=curr_info.color.to_list(),
    )
    ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
    ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
    ax1.tick_params(labelsize=12)  

    plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
    plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
    ax1.set_title("kgb zones")
    
    unique_groups = curr_info[['kgb_group', 'color']].drop_duplicates()
    legend_handles = [
        mpatches.Patch(color=row['color'], label=row['kgb_group']) 
        for _, row in unique_groups.iterrows()
        ]

    # Add legend
    plt.legend(handles=legend_handles, title='KGB Groups')
    
    plt.show()




val_info = info[country_index]

val_info.startdate = pd.to_datetime(val_info.startdate)
val_info.enddate = pd.to_datetime(val_info.enddate)

val_info = val_info[val_info['cleaned_years']>=min_yrs] 

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




curr_info = val_info
curr_info["color"] = curr_info['kgb_group'].map(colors_map)


lon_lims = [truncate_neg(np.min(curr_info.longitude),2.5),np.ceil(np.max(curr_info.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(curr_info.latitude),2.5),np.ceil(np.max(curr_info.latitude/2.5))*2.5]


fig = plt.figure(figsize=(20, 20))
norm = mcolors.Normalize(vmin=0, vmax=1)


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    curr_info.longitude,
    curr_info.latitude,
    c=curr_info.color.to_list(),
)
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("kgb zones")

unique_groups = curr_info[['kgb_group', 'color']].drop_duplicates()
legend_handles = [
    mpatches.Patch(color=row['color'], label=row['kgb_group']) 
    for _, row in unique_groups.iterrows()
    ]

# Add legend
plt.legend(handles=legend_handles, title='KGB Groups')

plt.show()







S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, censor_thr],
        alpha = 0,
        min_ev_dur = 60,
        niter_smev = 1000, 
    )

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
    
    
new_df["kgb_group"] = val_info.kgb_group.to_list()
df_parameters_0["kgb_group"] = val_info.kgb_group.to_list()

parameter_kgb_means = new_df[new_df.columns[3:]].groupby("kgb_group").mean()

parameter_kgb_means_0 = df_parameters_0[df_parameters_0.columns[4:]].groupby("kgb_group").mean()



df_savename = drive + ':/outputs/'+country_save+'\\parameters_kgb.csv'
saved_output_files = glob.glob(drive + ':/outputs/'+country_save+'/*')

if df_savename not in saved_output_files: #read in files and create t time series and do TENAX if it hasnt been done already
    print('TENAX not done yet on '+country_save+' with set b using kgb. making data.')
    
    T_files = sorted(glob.glob(drive+':/ERA5_land/'+ERA_country+'*/*')) #make list of era5 files
    saved_files = glob.glob(drive+':/'+country+'_temp/*') #temp files already saved
    
    F_phats = [0]*len(new_df)
    RL = [0]*len(new_df)
    
    nans = xr.open_dataarray(T_files[0])[0] 
    nans = np.invert(np.isnan(nans)).astype(int)
    
    saved_counter = 0
    
    start_time = [0]*len(new_df)
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time() 
        #read in ppt data
        if 'code_str' in locals():
            G,data_meta = read_GSDR_file(f"{drive}:/{country}/{code_str}{new_df.station.iloc[i]}.txt",
                                         name_col)
        else:
            G = pd.read_csv(files_sel[i])
            G['prec_time'] = pd.to_datetime(G['prec_time'])
            G.set_index('prec_time', inplace=True)
            
        ######################################################################
        #read in T data
        if 'code_str' in locals():
            save_path = f"{drive}:/{country}_temp\\{code_str}{new_df.station.iloc[i]}.nc"
        else:
            save_path = drive + ':/'+country+'_temp\\'+str(df_parameters.station[val_info.index[i]]) + '.nc'
        print(save_path)
        print(f"{drive}:/{country}\\{code_str}{new_df.station.iloc[i]}.txt")
        
        # Check if file already exists before saving
        
        if save_path not in saved_files:
            print(f'file {save_path} not there')
            T_ERA = []
            
        else:
            print(f"File {save_path} already exists. Skipping loading.")
            T_ERA = xr.load_dataarray(save_path)
            
            #####################################################################
        #TENAX 
        if len(T_ERA) == 0: # dont do tenax if no T data saved
            print('skip')
            F_phats[i] = np.array([np.nan,np.nan,np.nan,np.nan])
            RL[i] = np.nan
        else:
            
            climate_type = val_info.kgb_group.iloc[i]
            
            b_set = parameter_kgb_means.b[climate_type]
            print(f"{climate_type} : {b_set}")
            
            data = G 
            data = S.remove_incomplete_years(data, name_col)
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
            
            
            # Number of threshold 
            thr = dict_ordinary["60"]["ordinary"].quantile(S.left_censoring[1])
            
            
            n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
            
            AMS_sort = AMS.sort_values(by=['AMS'])['AMS']
            plot_pos = np.arange(1,np.size(AMS_sort)+1)/(1+np.size(AMS_sort))
            
            eRP = 1/(1-plot_pos)
            S.return_period = eRP
            
            #TENAX MODEL HERE
            #magnitude model
            F_phats_norm, loglik, _, _ = S.magnitude_model(P, T, thr)
            F_phats[i], loglik, _, _ = S.magnitude_model(P, T, thr, b_set = b_set)
            #temperature model
            g_phat = S.temperature_model(T)
            
            T_min = g_phat[0] - 2.5 * g_phat[1]
            T_max = g_phat[0] + 2.5 * g_phat[1]
            Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
            
            RL[i], __, __ = S.model_inversion(F_phats[i], g_phat, n, Ts)
            
            
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (len(new_df)-i)*time_taken/60
            print(f"b set: {F_phats[i]}. normal {F_phats_norm}")
            print(RL[i])
            print(f"{i}/{len(new_df)}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
        
    
    
    df_parameters_kgb = pd.DataFrame({'station':new_df.station,'latitude':val_info.latitude.to_list(),'longitude':val_info.longitude.to_list(),
                                       'kappa':np.array(F_phats)[:,0],'b':np.array(F_phats)[:,1],'lambda':np.array(F_phats)[:,2],'a':np.array(F_phats)[:,3],
                                       'return_levels': RL,
                                       'kgb_group' : val_info.kgb_group.to_list()
                                       })
    df_parameters_kgb.to_csv(df_savename,index=False) #save calculated parameters
    

else:
    print('TENAX already done! reading in data')
    df_parameters_kgb = pd.read_csv(df_savename) 


lon_lims = [truncate_neg(np.min(df_parameters_kgb.longitude),2.5),np.ceil(np.max(df_parameters_kgb.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters_kgb.latitude),2.5),np.ceil(np.max(df_parameters_kgb.latitude/2.5))*2.5]

s=3
############################################
## PLOT ALL b 2
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

# Choosing cmap
if df_parameters_kgb.b.min() == 0:
    norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
else:
    norm = mcolors.TwoSlopeNorm(vmin=df_parameters_kgb.b.min(),vcenter =df_parameters_kgb.b.min()/2, vmax=0)

sc = ax1.scatter( #plot the negligable at 5% lvl points
    new_df.longitude,
    new_df.latitude,
    c = df_parameters_kgb.b,
    s = s,
    cmap = 'rainbow',
)



# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('b', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. b ', fontsize=16)
plt.show()
#THIS SHOWS THE LOCATION OF THE STATION, NOT THE ERA DATA

###############################################################
#scale param
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.TwoSlopeNorm(vmin=0,vcenter = (df_parameters_kgb["lambda"].max()/2) , vmax=df_parameters_kgb["lambda"].max())
sc = ax1.scatter(
    df_parameters_kgb.longitude,
    df_parameters_kgb.latitude,
    c=df_parameters_kgb["lambda"],
    s = s,
    cmap='YlGnBu',  
    norm=norm
)




# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('λ (mm/hr)', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. λ', fontsize=16)
plt.show()

#######################################################
#scale param
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.TwoSlopeNorm(vmin=0,vcenter = (df_parameters_kgb.kappa.max()/2) , vmax=df_parameters_kgb.kappa.max())
sc = ax1.scatter(
    df_parameters_kgb.longitude,
    df_parameters_kgb.latitude,
    c=df_parameters_kgb.kappa,
    s = s,
    cmap='hsv',
)




# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('κ', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. κ', fontsize=16)
plt.show()

#######################################################
fig = plt.figure(figsize=(10, 10))
proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

sc = ax1.scatter(
    df_parameters_kgb.longitude,
    df_parameters_kgb.latitude,
    c=df_parameters_kgb.a,
    s = s,
    cmap='hsv',  
)




# Add a colorbar at the bottom
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label('a', fontsize=14)  
cb.ax.tick_params(labelsize=12)

# Set x and y ticks
ax1.set_xticks(np.arange(lon_lims[0],lon_lims[1]+1,2.5), crs=proj)
ax1.set_yticks(np.arange(lat_lims[0],lat_lims[1]+1,2.5), crs=proj)
ax1.tick_params(labelsize=12)  

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)


plt.title(f'GSDR: {ERA_country}. a', fontsize=16)
plt.show()














