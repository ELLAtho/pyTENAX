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



drive = 'D'
alpha_set = 0.05


# country = 'Germany' 
# ERA_country = 'Germany'
# country_save = 'Germany'
# code_str = 'DE_'
# minlat,minlon,maxlat,maxlon = 47, 3, 55, 15 #GERMANY
# name_len = 5
# min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9


country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP_'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
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


save_name = f"{drive}:/outputs/{country_save}\\return_levels.csv"
output_files = glob.glob(f"{drive}:/outputs/{country_save}/*")

if save_name not in output_files:
    print("levels not calculated yet, doing now.")
    S = TENAX(
            return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
            durations = [60, 180, 360, 720, 1440],
            left_censoring = [0, censor_thr],
            alpha = alpha_set,
            min_ev_dur = 60,
            niter_smev = 1000, 
        )
    
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
        g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
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
    RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels.csv",index=False)
    
    FRMSE_df = pd.DataFrame({'station': df_parameters.station,
                             'FRMSE': FRMSE,
                             'FRMSE_5': FRMSE_5,
                             'FRMSE_0': FRMSE_0})
    
    FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE.csv",index=False)
else:
    print("Files already saved, reading")
    RL_df = pd.read_csv(f"{drive}:/outputs/{country_save}/return_levels.csv", dtype={'station': str})
    nan_locs = RL_df.return_levels[RL_df.return_levels.isna()].index
    replace_range = np.arange(0,len(RL_df))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
        RL_df.loc[j, "return_levels"] = np.fromstring(RL_df.return_levels.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        RL_df.loc[j, "return_levels_5"] = np.fromstring(RL_df["return_levels_5"].iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        RL_df.loc[j, "return_levels_b0"] = np.fromstring(RL_df.return_levels_b0.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        RL_df.loc[j, "obs_AMS"] = np.fromstring(RL_df.obs_AMS.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
    
       
    
    FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE.csv", dtype={'station': str})



###############################################################################
# Probs and stuff

mult_prob = [0] * len(new_df)
ave_prob = [0] * len(new_df)
mins = [0] * len(new_df)
maxes = [0] * len(new_df)
n_bad_RL = [0] * len(new_df)
start_time = [0] * len(new_df)

for i in np.arange(0,len(new_df)):
    start_time[i] = time.time()
    n_itn = 1000
    percentages = [0.05,0.95]
    n = round(df_parameters.n_events_per_yr.iloc[i])
    
    g_phat = [df_parameters.mu.iloc[i],df_parameters.sigma.iloc[i]]
    if np.any(np.isnan(g_phat[0])):
        print(f"no gphat. {g_phat}")
        
        mult_prob[i] = np.nan
        ave_prob[i] = np.nan
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
                             "n_bad_RL": n_bad_RL
                             })

# TODO: check format of mins and maxes, dont know if pandas will understand


#maps
significants = df_parameters[df_parameters.b != 0]
show_sig_locs = False


lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]
s = 5
cmap = 'magma_r'


fig = plt.figure(figsize=(20, 20))
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

fig = plt.figure(figsize=(20, 20))
norm = mcolors.Normalize(vmin=-0.7, vmax=0.7)
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







# CHECKS
j = 19
plot_pos = np.arange(1,np.size(RL_df.obs_AMS.iloc[j])+1)/(1+np.size(RL_df.obs_AMS.iloc[j]))

eRP = 1/(1-plot_pos)

TNX_FIG_valid(RL_df.obs_AMS.iloc[j],eRP,RL_df.return_levels_b0.iloc[j],TENAXlabel = "b = 0")
plt.plot(eRP,RL_df.return_levels.iloc[j],"r", alpha = 0.5,label = "b = free")
plt.plot(eRP,RL_df.return_levels_5.iloc[j],"g",alpha = 0.5, label = "b = 5% sig")
plt.legend()
plt.title(f"station {j}: {FRMSE_df.station.iloc[j]}. free FRMSE: {FRMSE_df.FRMSE.iloc[j]:.3f} \n 5% sig FRMSE: {FRMSE_df.FRMSE_5.iloc[j]:.3f} \n b always 0 FRMSE: {FRMSE_df.FRMSE_0.iloc[j]:.3f}")
plt.show()



