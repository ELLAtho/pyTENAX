# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 08:13:31 2025

@author: ellar
"""

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
    new_df = df_parameters[['station','latitude','longitude','b','kappa','lambda','a','thr']].copy()
    
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


#file with gphat skew
temp_skew_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_skew.csv", dtype={'station': str})


if "return_levels_skew" not in RL_df.columns:
    print("calculating return levels with skew fit")
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
            
            g_phat = [temp_skew_df.skewness.iloc[i],temp_skew_df.g_phat1.iloc[i],temp_skew_df.g_phat2.iloc[i]]
            
            
            AMS = RL_df.obs_AMS.iloc[i]
            plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
            
            eRP = 1/(1-plot_pos)
            S.return_period = eRP
            
            RL[i], __, __ = S.model_inversion(F_phat, g_phat, n, Ts,temp_method = "skewnorm")
            RL_0[i], __, __ = S.model_inversion(F_phat_0, g_phat, n, Ts,temp_method = "skewnorm")
            RL_exp[i], __, __ = S.model_inversion(F_phat_exp, g_phat, n, Ts,temp_method = "skewnorm")
    
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
    
    FRMSE_df.to_csv(f"{drive}:/outputs/{country_save}/FRMSE_skew.csv",index=False)
    
    RL_df["return_levels_skew"] = RL
    RL_df["return_levels_skew_exp"] = RL_exp
    RL_df["return_levels_skew_0"] = RL_0
    RL_df.to_csv(f"{drive}:/outputs/{country_save}/return_levels.csv",index=False)
else:
    FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE_skew.csv", dtype={'station': str})



FRMSE_df4 = pd.read_csv(f"{drive}:/outputs/{country_save}/FRMSE.csv", dtype={'station': str})

###############################################################################
#Plots
lon_lims = [truncate_neg(np.min(df_parameters.longitude),2.5),np.ceil(np.max(df_parameters.longitude/2.5))*2.5]
lat_lims = [truncate_neg(np.min(df_parameters.latitude),2.5),np.ceil(np.max(df_parameters.latitude/2.5))*2.5]


# Plot plain FRMSEs


s = 3
cmap = 'magma_r'


fig = plt.figure(figsize=(15, 7))
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

plt.suptitle("FRMSE with skewnorm")

plt.show()


##############################################################################
#Station plots
val_info.index = range(len(val_info))

qs = [.85,.95,.99,.999]

peaks_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\peaks.csv")

mask = ((temp_skew_df.skewness > 0) &
    (peaks_df.n_peaks01 == 1) &
    (df_parameters.longitude < -110) &
    (df_parameters.latitude > 40) &
    (val_info.cleaned_years > 20))

RL_westcoast = RL_df[mask]
FRMSE_westcoast = FRMSE_df[mask]
parameters_westcoast = df_parameters[mask]
df_westcoast = new_df[mask]
skew_westcoast = temp_skew_df[mask]

RL_westcoast_high = RL_westcoast[FRMSE_westcoast.FRMSE > np.nanquantile(FRMSE_westcoast.FRMSE,0.9)]
FRMSE_westcoast_high = FRMSE_westcoast[FRMSE_westcoast.FRMSE > np.nanquantile(FRMSE_westcoast.FRMSE,0.9)]
parameters_westcoast_high = parameters_westcoast[FRMSE_westcoast.FRMSE > np.nanquantile(FRMSE_westcoast.FRMSE,0.9)]
df_westcoast_high = df_westcoast[FRMSE_westcoast.FRMSE > np.nanquantile(FRMSE_westcoast.FRMSE,0.9)]
skew_westcoast_high = skew_westcoast[FRMSE_westcoast.FRMSE > np.nanquantile(FRMSE_westcoast.FRMSE,0.9)]


RL_westcoast_low = RL_westcoast[FRMSE_westcoast.FRMSE < np.nanquantile(FRMSE_westcoast.FRMSE,0.1)]
FRMSE_westcoast_low = FRMSE_westcoast[FRMSE_westcoast.FRMSE < np.nanquantile(FRMSE_westcoast.FRMSE,0.1)]
parameters_westcoast_low = parameters_westcoast[FRMSE_westcoast.FRMSE < np.nanquantile(FRMSE_westcoast.FRMSE,0.1)]
df_westcoast_low = df_westcoast[FRMSE_westcoast.FRMSE < np.nanquantile(FRMSE_westcoast.FRMSE,0.1)]
skew_westcoast_low = skew_westcoast[FRMSE_westcoast.FRMSE < np.nanquantile(FRMSE_westcoast.FRMSE,0.1)]
    
for i in range(6):
    station = RL_westcoast_low.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    
    eT = np.arange(np.min(T),np.max(T)+4,1)
    g_phat = [skew_westcoast_low.skewness.iloc[i],skew_westcoast_low.g_phat1.iloc[i],skew_westcoast_low.g_phat2.iloc[i]]
    
    TNX_FIG_temp_model(T, g_phat, 4, eT,method = "skewnorm")
    plt.xlim(np.min(T),np.max(T))
    plt.ylim(0,4/(np.max(T)-np.min(T)))
    plt.show()
    F_phat = [df_westcoast_low.kappa.iloc[i],df_westcoast_low.b.iloc[i],df_westcoast_low["lambda"].iloc[i],df_westcoast_low.a.iloc[i]]
    thr = df_westcoast_low.thr.iloc[i]
    
    RL = RL_westcoast_low.return_levels.iloc[i]
    AMS = RL_westcoast_low.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"station: {station}. westcoast, low FRMSE")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_westcoast_low.return_levels_skew.iloc[i],label = f"b = free, temperature skew")
    plt.plot(eRP,RL_westcoast_low.return_levels_skew_0.iloc[i],label = f"b = 0, temperature skew")
    # plt.plot(eRP,RL_westcoast_low.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()

for i in range(6):
    station = RL_westcoast_high.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    
    eT = np.arange(np.min(T),np.max(T)+4,1)
    g_phat = [skew_westcoast_high.skewness.iloc[i],skew_westcoast_high.g_phat1.iloc[i],skew_westcoast_high.g_phat2.iloc[i]]
    
    TNX_FIG_temp_model(T, g_phat, 4, eT,method = "skewnorm")
    plt.xlim(np.min(T),np.max(T))
    plt.ylim(0,4/(np.max(T)-np.min(T)))
    plt.show()
    F_phat = [df_westcoast_high.kappa.iloc[i],df_westcoast_high.b.iloc[i],df_westcoast_high["lambda"].iloc[i],df_westcoast_high.a.iloc[i]]
    thr = df_westcoast_high.thr.iloc[i]
    
    RL = RL_westcoast_high.return_levels.iloc[i]
    AMS = RL_westcoast_high.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"station: {station}. westcoast, high FRMSE")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_westcoast_high.return_levels_skew.iloc[i],label = f"b = free, temperature skew")
    plt.plot(eRP,RL_westcoast_high.return_levels_skew_0.iloc[i],label = f"b = 0, temperature skew")
    # plt.plot(eRP,RL_westcoast_high.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()


mask = ((temp_skew_df.skewness < 0) &
    (peaks_df.n_peaks01 == 1) &
    (df_parameters.longitude > -110) &
    (val_info.cleaned_years > 20)
    ) #negative skew, one peak

RL_southeast = RL_df[mask]
FRMSE_southeast = FRMSE_df[mask]
parameters_southeast = df_parameters[mask]
df_southeast = new_df[mask]
skew_southeast = temp_skew_df[mask]

RL_southeast_high = RL_southeast[FRMSE_southeast.FRMSE > np.nanquantile(FRMSE_southeast.FRMSE,0.9)]
FRMSE_southeast_high = FRMSE_southeast[FRMSE_southeast.FRMSE > np.nanquantile(FRMSE_southeast.FRMSE,0.9)]
parameters_southeast_high = parameters_southeast[FRMSE_southeast.FRMSE > np.nanquantile(FRMSE_southeast.FRMSE,0.9)]
df_southeast_high = df_southeast[FRMSE_southeast.FRMSE > np.nanquantile(FRMSE_southeast.FRMSE,0.9)]
skew_southeast_high = skew_southeast[FRMSE_southeast.FRMSE > np.nanquantile(FRMSE_southeast.FRMSE,0.9)]


RL_southeast_low = RL_southeast[FRMSE_southeast.FRMSE < np.nanquantile(FRMSE_southeast.FRMSE,0.1)]
FRMSE_southeast_low = FRMSE_southeast[FRMSE_southeast.FRMSE < np.nanquantile(FRMSE_southeast.FRMSE,0.1)]
parameters_southeast_low = parameters_southeast[FRMSE_southeast.FRMSE < np.nanquantile(FRMSE_southeast.FRMSE,0.1)]
df_southeast_low = df_southeast[FRMSE_southeast.FRMSE < np.nanquantile(FRMSE_southeast.FRMSE,0.1)]
skew_southeast_low = skew_southeast[FRMSE_southeast.FRMSE < np.nanquantile(FRMSE_southeast.FRMSE,0.1)]


    
for i in range(6):
    station = RL_southeast_low.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    
    eT = np.arange(np.min(T),np.max(T)+4,1)
    g_phat = [skew_southeast_low.skewness.iloc[i],skew_southeast_low.g_phat1.iloc[i],skew_southeast_low.g_phat2.iloc[i]]
    
    TNX_FIG_temp_model(T, g_phat, 4, eT,method = "skewnorm")
    plt.xlim(np.min(T),np.max(T))
    plt.ylim(0,4/(np.max(T)-np.min(T)))
    plt.show()
    F_phat = [df_southeast_low.kappa.iloc[i],df_southeast_low.b.iloc[i],df_southeast_low["lambda"].iloc[i],df_southeast_low.a.iloc[i]]
    thr = df_southeast_low.thr.iloc[i]
    
    RL = RL_southeast_low.return_levels.iloc[i]
    AMS = RL_southeast_low.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"station: {station}. southest, low FRMSE")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_southeast_low.return_levels_skew.iloc[i],label = f"b = free, temperature skew")
    plt.plot(eRP,RL_southeast_low.return_levels_skew_0.iloc[i],label = f"b = 0, temperature skew")
    # plt.plot(eRP,RL_southeast_low.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()

for i in range(6):
    station = RL_southeast_high.station.iloc[i]
    T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{station}.csv")
    P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{station}.csv")
    
    eT = np.arange(np.min(T),np.max(T)+4,1)
    g_phat = [skew_southeast_high.skewness.iloc[i],skew_southeast_high.g_phat1.iloc[i],skew_southeast_high.g_phat2.iloc[i]]
    
    TNX_FIG_temp_model(T, g_phat, 4, eT,method = "skewnorm")
    plt.xlim(np.min(T),np.max(T))
    plt.ylim(0,4/(np.max(T)-np.min(T)))
    plt.show()
    F_phat = [df_southeast_high.kappa.iloc[i],df_southeast_high.b.iloc[i],df_southeast_high["lambda"].iloc[i],df_southeast_high.a.iloc[i]]
    thr = df_southeast_high.thr.iloc[i]
    
    RL = RL_southeast_high.return_levels.iloc[i]
    AMS = RL_southeast_high.obs_AMS.iloc[i]
    
    plot_pos = np.arange(1,np.size(AMS)+1)/(1+np.size(AMS))
    
    eRP = 1/(1-plot_pos)
    
    TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
    plt.title(f"station: {station}. southest, high FRMSE")
    plt.show()
    
    
    TNX_FIG_valid(AMS,eRP,RL,TENAXlabel = f"b = free, beta = 4",obslabel='AMS',ylimits = [0,np.max(AMS)+1])
    plt.plot(eRP,RL_southeast_high.return_levels_skew.iloc[i],label = f"b = free, temperature skew")
    plt.plot(eRP,RL_southeast_high.return_levels_skew_0.iloc[i],label = f"b = 0, temperature skew")
    # plt.plot(eRP,RL_southeast_high.return_levels_kernal_exp.iloc[i],label = f"b = exp, temperature kernal")
    
    plt.legend()
    plt.show()


