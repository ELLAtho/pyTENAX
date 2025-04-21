# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:18:16 2025

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

drive = "D"

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

# country = 'US' 
# ERA_country = 'US'
# country_save = 'US_main'
# code_str = 'US_'
# minlat,minlon,maxlat,maxlon = 24, -125, 56, -66  
# name_len = 6
# min_startdate = dt.datetime(1950,1,1) #this is for if havent read all ERA5 data yet
# censor_thr = 0.9



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
        alpha = 0,
        min_ev_dur = 60,
        niter_smev = 1000, 
        beta = 4
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
df_parameters_exp = pd.read_csv(f"{drive}:/outputs/{country_save}/parameters_exp.csv", dtype={'station': str})
# for some reason in germany there is one less row...
    



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
missing_rows = pd.merge(df_parameters.station, df_parameters_0.station, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
if len(missing_rows) != 0:
    print("miss-match, dropping")
    df_parameters = df_parameters.drop(missing_rows.index)
    new_df = new_df.drop(missing_rows.index)
else:
    pass



save_name = f"{drive}:/outputs/{country_save}\\return_levels.csv"

RL_df = pd.read_csv(save_name, dtype={'station': str})
nan_locs = RL_df.return_levels[RL_df.return_levels.isna()].index
replace_range = np.arange(0,len(RL_df))

RL_column_names = [col for col in RL_df.columns if "return_levels" in col]


for col in RL_column_names:
    nan_locs = RL_df[col][RL_df[col].isna()].index
    replace_range = np.arange(0,len(RL_df))
    for k in range(len(nan_locs)):
        replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
    for j in replace_range:
    
        RL_df.loc[j, col] = np.fromstring(RL_df[col].iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')
        
nan_locs = RL_df.obs_AMS[RL_df.obs_AMS.isna()].index
replace_range = np.arange(0,len(RL_df))
for k in range(len(nan_locs)):
    replace_range = np.delete(replace_range, np.where(replace_range == nan_locs[k]))
for j in replace_range:
    RL_df.loc[j, "obs_AMS"] = np.fromstring(RL_df.obs_AMS.iloc[j].replace('\n', ' ').strip().replace('  ', ' ').strip().strip('[]'), sep=' ')




if "return_levels_kernal" not in RL_df.columns:
    print("calculating return levels with temperature kernel")
    RL = [0] * len(new_df)
    RL_0 = [0] * len(new_df)
    RL_exp = [0] *  len(new_df)
    start_time = [0] * len(new_df)
    FRMSE = [0] * len(new_df)
    FRMSE_0 = [0] * len(new_df)
    FRMSE_exp = [0] * len(new_df)
    
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time()
        
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"):
            RL[i] = np.nan
            RL_0[i] = np.nan
            RL_exp[i] = np.nan
            FRMSE[i] = np.nan
            FRMSE_0[i] = np.nan
            FRMSE_exp[i] = np.nan
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv")
            times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{df_parameters.station.iloc[i]}.csv",parse_dates = ["oe_time"])
            kde  = gaussian_kde(T) #use kernel density to get probability
            
            F_phat = [new_df.kappa.iloc[i],new_df.b.iloc[i],
                      new_df["lambda"].iloc[i],new_df.a.iloc[i]]
            
            #b always 0
            F_phat_0 = [df_parameters_0.kappa.iloc[i],df_parameters_0.b.iloc[i],
                        df_parameters_0["lambda"].iloc[i],df_parameters_0.a.iloc[i]]
            
            F_phat_exp = [df_parameters_exp.kappa.iloc[i],df_parameters_exp.b.iloc[i],
                        df_parameters_exp["lambda"].iloc[i],df_parameters_exp.a.iloc[i]]
            
            
            T_min = np.min(T)
            T_max = np.max(T)
            Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
            
            n = df_parameters.n_events_per_yr.iloc[i]
            
            pdf_values = kde(Ts)
            df = np.vstack([pdf_values, Ts])
            
            T_mc = randdf(S.n_monte_carlo, df, 'pdf').T              
           
            wbl_phat = np.column_stack((
                                        F_phat[2] * np.exp(F_phat[3] * T_mc),
                                        F_phat[0] + F_phat[1] * T_mc
                                        ))
            wbl_phat_0 = np.column_stack((
                                        F_phat_0[2] * np.exp(F_phat_0[3] * T_mc),
                                        F_phat_0[0] + F_phat_0[1] * T_mc
                                        ))
            
            wbl_phat_exp = np.column_stack((
                                        F_phat_exp[2] * np.exp(F_phat_exp[3] * T_mc),
                                        F_phat_exp[0] * np.exp(F_phat_exp[1] * T_mc)
                                        ))
            
            AMS = RL_df.obs_AMS.iloc[i]
            plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
            
            eRP = 1/(1-plot_pos)
            S.return_period = eRP
            
            vguess = 10 ** np.arange(np.log10(0.05), np.log10(5e2), 0.05)
            RL[i] = SMEV_Mc_inversion(wbl_phat, n, S.return_period, vguess, method_root_scalar="brentq")
            
            RL_0[i] = SMEV_Mc_inversion(wbl_phat_0, n, S.return_period, vguess, method_root_scalar="brentq")
            
            
            RL_exp[i] = SMEV_Mc_inversion(wbl_phat_exp, n, S.return_period, vguess, method_root_scalar="brentq")
            
            diffs = RL[i] - AMS
            diffs_0 = RL_0[i] - AMS
            diffs_exp = RL_exp[i] - AMS
            
            FRMSE[i] = np.sqrt(np.sum(diffs**2)/len(diffs))/(np.sum(AMS)/len(diffs))
            FRMSE_0[i] = np.sqrt(np.sum(diffs_0**2)/len(diffs_0))/(np.sum(AMS)/len(diffs_0))
            FRMSE_exp[i] = np.sqrt(np.sum(diffs_exp**2)/len(diffs_exp))/(np.sum(AMS)/len(diffs_exp))
        
        print(f"Free {FRMSE[i]}, b always 0 {FRMSE_0[i]}")
        time_taken = (time.time()-start_time[i-9])/10
        time_left = (len(new_df)-i)*time_taken/60
        print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
    
    FRMSE_df = pd.DataFrame({'station': df_parameters.station,
                             'FRMSE': FRMSE,
                             'FRMSE_0': FRMSE_0,
                             'FRMSE_exp': FRMSE_exp
                             })
    
    FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE_kernel.csv",index=False)
    
    RL_df["return_levels_kernal"] = RL
    RL_df["return_levels_kernal_exp"] = RL_exp
    RL_df["return_levels_kernal_0"] = RL_0
    RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels.csv",index=False)

else:
    FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE_kernel.csv", dtype={'station': str})


FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE.csv", dtype={'station': str})

###############################################################################
# Plot FRMSE comparisons

lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]

fig = plt.figure(figsize=(15, 10))
norm = mcolors.Normalize(vmin=-0.2, vmax=0.2)
s = 3
cmap = 'seismic'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df.FRMSE - FRMSE_df.FRMSE_exp),
    cmap=cmap,
    norm = norm,
    s = s,
)

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

ax1.set_title("b free - exp")



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

gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

ax2.set_title("b free - b 0")


ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df.FRMSE_exp - FRMSE_df.FRMSE_0),
    cmap=cmap,
    norm = norm,  
    s = s,  
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

ax3.set_title("b exp - b 0")

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

gl = ax4.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

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
fig.suptitle(f'GSDR: {ERA_country}. FRMSE on RL with kernel density temperature', fontsize=16)
plt.show()



fig = plt.figure(figsize=(15, 10))
norm = mcolors.Normalize(vmin=-0.2, vmax=0.2)
s = 3
cmap = 'seismic'


proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(2, 2, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax1.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df4.FRMSE - FRMSE_df4.FRMSE_bexp),
    cmap=cmap,
    norm = norm,
    s = s,
)

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

ax1.set_title("b free - exp")



ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df4.FRMSE - FRMSE_df4.FRMSE_0),
    cmap=cmap,
    norm = norm,  
    s = s,
)

gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

ax2.set_title("b free - b 0")


ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=(FRMSE_df4.FRMSE_bexp - FRMSE_df4.FRMSE_0),
    cmap=cmap,
    norm = norm,  
    s = s,  
)


gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

ax3.set_title("b exp - b 0")

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

gl = ax4.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

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
fig.suptitle(f'GSDR: {ERA_country}. FRMSE on RL with beta 4 temperature', fontsize=16)
plt.show()
###############################################################################
# Plot plain FRMSEs


s = 3
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
gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b free")



ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=FRMSE_df.FRMSE_exp,
    cmap=cmap,
    norm = norm,  
    s = s,
)

#plot the locations of significant stations

gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b exp")


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
gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
 
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

gl = ax4.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

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

plt.suptitle("FRMSE with kernel density")

plt.show()



s = 3
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
    c=FRMSE_df4.FRMSE,
    cmap=cmap,
    norm = norm,
    s = s,
)
gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax1.set_title("b free")



ax2 = fig.add_subplot(2, 2, 2, projection=proj)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax2.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=FRMSE_df4.FRMSE_bexp,
    cmap=cmap,
    norm = norm,  
    s = s,
)

#plot the locations of significant stations

gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

  
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax2.set_title("b exp")


ax3 = fig.add_subplot(2, 2, 3, projection=proj)
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS, linestyle=':')


sc = ax3.scatter(
    df_parameters.longitude,
    df_parameters.latitude,
    c=FRMSE_df4.FRMSE_0,
    cmap=cmap,
    norm = norm,  
    s = s,  
)
gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
 
plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)
ax3.set_title("b = 0")

ax4 = fig.add_subplot(2, 2, 4, projection=proj)
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS, linestyle=':')

norm = mcolors.Normalize(vmin=-0.1, vmax=0.1)

sc4 = ax4.scatter(
    val_info.longitude,
    val_info.latitude,
    c=FRMSE_df4.FRMSE - FRMSE_df.FRMSE,
    cmap="seismic",
    s = s,  
    norm = norm
)

gl = ax4.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

plt.xlim(lon_lims[0]-1,lon_lims[1]+1)
plt.ylim(lat_lims[0]-1,lat_lims[1]+1)

fig.subplots_adjust(right=0.85)

cbar_ax4 = fig.add_axes([0.87, 0.12, 0.03, 0.32])  # Position for the colorbar outside
cb4 = plt.colorbar(sc4, cax=cbar_ax4)  # Colorbar for ax4
cb4.set_label(r'$\Delta$FRMSE', fontsize=14)
cb4.ax.tick_params(labelsize=12)

ax4.set_title("b = free. beta = 4 - kernal")





# Add a colorbar at the bottom
cbar_ax = fig.add_subplot([0.15, 0.02, 0.7, 0.03])  # Position for the colorbar
cb = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label('FRMSE', fontsize=14)
cb.ax.tick_params(labelsize=12)

plt.suptitle("FRMSE with beta = 4")

plt.show()

###############################################################################
# check some stations incl magnitude
val_info.index = range(len(val_info))


df_high = new_df[FRMSE_df.FRMSE > np.nanquantile(FRMSE_df.FRMSE,0.9)] #stations with top 10% FRMSE b free
FRMSE_df_high = FRMSE_df[FRMSE_df.FRMSE > np.nanquantile(FRMSE_df.FRMSE,0.9)]
info_high = val_info[FRMSE_df.FRMSE > np.nanquantile(FRMSE_df.FRMSE,0.9)]
RL_df_high = RL_df[FRMSE_df.FRMSE > np.nanquantile(FRMSE_df.FRMSE,0.9)]

df_high_north = df_high[df_high.latitude > 35]
FRMSE_df_high_north = FRMSE_df_high[df_high.latitude > 35]
info_high_north = info_high[df_high.latitude > 35]
RL_df_high_north = RL_df_high[df_high.latitude > 35]


df_high_north_long = df_high_north[info_high_north.cleaned_years > 30]
FRMSE_df_high_north_long = FRMSE_df_high_north[info_high_north.cleaned_years > 30]
info_high_north_long = info_high_north[info_high_north.cleaned_years > 30]
RL_df_high_north_long = RL_df_high_north[info_high_north.cleaned_years > 30]



qs = [.85,.95,.99,.999]


for i in range(6):
    station = df_high_north_long.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    
    F_phat = [df_high_north_long.kappa.iloc[i],df_high_north_long.b.iloc[i],df_high_north_long["lambda"].iloc[i],df_high_north_long.a.iloc[i]]
    thr = df_high_north_long.thr.iloc[i]
    
    RL = RL_df_high_north_long.return_levels.iloc[i]
    AMS = RL_df_high_north_long.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
    eT = np.arange(np.min(T),np.max(T)+4,1)
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"({info_high_north_long.latitude.iloc[i]},{info_high_north_long.longitude.iloc[i]}). FRMSE = {FRMSE_df_high_north_long.FRMSE.iloc[i]:.2f}")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_df_high_north_long.return_levels_kernal.iloc[i],label = f"b = free, temperature kernal")
    plt.plot(eRP,RL_df_high_north_long.return_levels_kernal_0.iloc[i],label = f"b = 0, temperature kernal")
    plt.plot(eRP,RL_df_high_north_long.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()
    
 
# as above but for stations with large bs
RL_df_high_north_long_sig = RL_df_high_north_long[df_high_north_long.b < -0.02]
df_high_north_long_sig = df_high_north_long[df_high_north_long.b < -0.02]
FRMSE_df_high_north_long_sig = FRMSE_df_high_north_long[df_high_north_long.b < -0.02]
info_high_north_long_sig = info_high_north_long[df_high_north_long.b < -0.02]
 
    
for i in range(6):
    station = df_high_north_long_sig.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    
    F_phat = [df_high_north_long_sig.kappa.iloc[i],df_high_north_long_sig.b.iloc[i],df_high_north_long_sig["lambda"].iloc[i],df_high_north_long_sig.a.iloc[i]]
    thr = df_high_north_long_sig.thr.iloc[i]
    
    RL = RL_df_high_north_long_sig.return_levels.iloc[i]
    AMS = RL_df_high_north_long_sig.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
    eT = np.arange(np.min(T),np.max(T)+4,1)
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"({info_high_north_long_sig.latitude.iloc[i]},{info_high_north_long_sig.longitude.iloc[i]}). FRMSE = {FRMSE_df_high_north_long_sig.FRMSE.iloc[i]:.2f}")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_df_high_north_long_sig.return_levels_kernal.iloc[i],label = f"b = free, temperature kernal")
    plt.plot(eRP,RL_df_high_north_long_sig.return_levels_kernal_0.iloc[i],label = f"b = 0, temperature kernal")
    plt.plot(eRP,RL_df_high_north_long_sig.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()
    



# low FRMSE so good fit
df_low = new_df[FRMSE_df.FRMSE < np.nanquantile(FRMSE_df.FRMSE,0.1)] #stations with top 10% FRMSE b free
FRMSE_df_low = FRMSE_df[FRMSE_df.FRMSE < np.nanquantile(FRMSE_df.FRMSE,0.1)]
info_low = val_info[FRMSE_df.FRMSE < np.nanquantile(FRMSE_df.FRMSE,0.1)]
RL_df_low = RL_df[FRMSE_df.FRMSE < np.nanquantile(FRMSE_df.FRMSE,0.1)]

df_low_north = df_low[df_low.latitude > 35]
FRMSE_df_low_north = FRMSE_df_low[df_low.latitude > 35]
info_low_north = info_low[df_low.latitude > 35]
RL_df_low_north = RL_df_low[df_low.latitude > 35]


df_low_north_long = df_low_north[info_low_north.cleaned_years > 30]
FRMSE_df_low_north_long = FRMSE_df_low_north[info_low_north.cleaned_years > 30]
info_low_north_long = info_low_north[info_low_north.cleaned_years > 30]
RL_df_low_north_long = RL_df_low_north[info_low_north.cleaned_years > 30]
    

    
for i in range(6):
    station = df_low_north_long.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    eT = np.arange(np.min(T),np.max(T)+4,1)
    
    T_min = np.min(T)
    T_max = np.max(T)
    Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
    
    
    g_phat = [df_low_north_long.mu.iloc[i],df_low_north_long.sigma.iloc[i]]
    g_phat6 = S.temperature_model(T,beta = 6)
    
    TNX_FIG_temp_model(T, g_phat, 4, eT)
    plt.plot(eT,gen_norm_pdf(eT,g_phat6[0],g_phat6[1],6))
    plt.show()
    
    F_phat = [df_low_north_long.kappa.iloc[i],df_low_north_long.b.iloc[i],df_low_north_long["lambda"].iloc[i],df_low_north_long.a.iloc[i]]
    thr = df_low_north_long.thr.iloc[i]
    n = df_low_north_long.n_events_per_yr.iloc[i]
    
    RL = RL_df_low_north_long.return_levels.iloc[i]
    AMS = RL_df_low_north_long.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
    S.beta = 6
    S.return_period = eRP
    RL6, __, __ = S.model_inversion(F_phat, g_phat6, n, Ts)
   
    
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"({station}. {info_low_north_long.latitude.iloc[i]},{info_low_north_long.longitude.iloc[i]}). FRMSE = {FRMSE_df_low_north_long.FRMSE.iloc[i]}")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_df_low_north_long.return_levels_kernal.iloc[i],label = f"b = free, temperature kernal")
    plt.plot(eRP,RL_df_low_north_long.return_levels_kernal_0.iloc[i],label = f"b = 0, temperature kernal")
    plt.plot(eRP,RL_df_low_north_long.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    plt.plot(eRP,RL6, label = "beta = 6, b free")
    
    plt.legend()
    plt.show()
    
    
###############################################################################
#location with good and bad fit for free vs 0

free_take_0 = FRMSE_df.FRMSE - FRMSE_df.FRMSE_0

mask = ((free_take_0 < np.nanquantile(free_take_0,0.1)) & # where free is better
        (val_info.cleaned_years > 30)
        )

df_free_better = new_df[mask]
FRMSE_df_free_better = FRMSE_df[mask]
info_free_better = val_info[mask]
RL_df_free_better = RL_df[mask]



for i in range(6):
    station = df_free_better.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    eT = np.arange(np.min(T),np.max(T)+4,1)
    
    T_min = np.min(T)
    T_max = np.max(T)
    Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
    
    
    g_phat = [df_free_better.mu.iloc[i],df_free_better.sigma.iloc[i]]
    g_phat6 = S.temperature_model(T,beta = 6)
    
    
    F_phat = [df_free_better.kappa.iloc[i],df_free_better.b.iloc[i],df_free_better["lambda"].iloc[i],df_free_better.a.iloc[i]]
    thr = df_free_better.thr.iloc[i]
    n = df_free_better.n_events_per_yr.iloc[i]
    
    RL = RL_df_free_better.return_levels.iloc[i]
    AMS = RL_df_free_better.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
       
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"({station}. {info_free_better.latitude.iloc[i]},{info_free_better.longitude.iloc[i]}). FRMSE = {FRMSE_df_free_better.FRMSE.iloc[i]}")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_df_free_better.return_levels_kernal.iloc[i],label = f"b = free, temperature kernal")
    plt.plot(eRP,RL_df_free_better.return_levels_kernal_0.iloc[i],label = f"b = 0, temperature kernal")
    plt.plot(eRP,RL_df_free_better.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()


# now the opposite
mask = ((free_take_0 > np.nanquantile(free_take_0,0.9)) & # where free is better
        (val_info.cleaned_years > 30)
        )

df_free_better = new_df[mask]
FRMSE_df_free_better = FRMSE_df[mask]
info_free_better = val_info[mask]
RL_df_free_better = RL_df[mask]


for i in range(6):
    station = df_free_better.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    eT = np.arange(np.min(T),np.max(T)+4,1)
    
    T_min = np.min(T)
    T_max = np.max(T)
    Ts = np.arange(T_min - S.temp_delta, T_max + S.temp_delta, S.temp_res_monte_carlo)
    
    
    g_phat = [df_free_better.mu.iloc[i],df_free_better.sigma.iloc[i]]
    g_phat6 = S.temperature_model(T,beta = 6)
    
    
    F_phat = [df_free_better.kappa.iloc[i],df_free_better.b.iloc[i],df_free_better["lambda"].iloc[i],df_free_better.a.iloc[i]]
    thr = df_free_better.thr.iloc[i]
    n = df_free_better.n_events_per_yr.iloc[i]
    
    RL = RL_df_free_better.return_levels.iloc[i]
    AMS = RL_df_free_better.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
       
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"({station}. {info_free_better.latitude.iloc[i]},{info_free_better.longitude.iloc[i]}). FRMSE = {FRMSE_df_free_better.FRMSE.iloc[i]}")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_df_free_better.return_levels_kernal.iloc[i],label = f"b = free, temperature kernal")
    plt.plot(eRP,RL_df_free_better.return_levels_kernal_0.iloc[i],label = f"b = 0, temperature kernal")
    plt.plot(eRP,RL_df_free_better.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()


