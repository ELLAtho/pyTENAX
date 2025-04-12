# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:59:10 2025

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
from scipy.optimize import curve_fit

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




save_name = f"{drive}:/outputs/{country_save}/doubles\\temp_FRMSE_doubles.csv"
output_files = glob.glob(f"{drive}:/outputs/{country_save}/doubles/*")
GOF_perc = 0.8
S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],
        durations = [60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
        min_ev_dur = 60,
        beta = 4
    )

###############################################################TODO: maybe put these in a function 
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

#############################################################################


if save_name not in output_files:
    print("temp FRMSE not calculated yet for splits. here we gooooooo")
    

    FRMSE_upper_perc = [[0, 0, 0] for _ in range(len(new_df))]
    start_time = [0] * len(new_df)
    FRMSE_doub = [0] * len(new_df)
    FRMSE_summer_winter = [0] * len(new_df)
    FRMSE_summer_winter_skew = [0] * len(new_df)
    g_phat_double_free = [0] * len(new_df)
    g_phat_winter = [0] * len(new_df)
    g_phat_summer = [0] * len(new_df)
    g_phat_winter_skew = [0] * len(new_df)
    g_phat_summer_skew = [0] * len(new_df)
    
    log_liks = [[0, 0, 0] for _ in range(len(new_df))]
    
    
    for i in np.arange(0, len(new_df)):
        start_time[i] = time.time()
    
        oe_save = f"{drive}:/ordinary_events/{country_save}\\T_{df_parameters.station.iloc[i]}.csv"
        if oe_save not in glob.glob(f"{drive}:/ordinary_events/{country_save}/*"): #save as nans if not in
            
            FRMSE_upper_perc[i] = [np.nan,np.nan,np.nan]
            FRMSE_doub[i] = np.nan
            FRMSE_summer_winter[i] = np.nan
            FRMSE_summer_winter_skew[i] = np.nan
            g_phat_double_free[i] = [np.nan] * 6
            g_phat_winter[i] = [np.nan]*2
            g_phat_summer[i] = [np.nan]*2
            g_phat_winter_skew[i] = [np.nan]*3
            g_phat_summer_skew[i] = [np.nan]*3
            
            log_liks[i] = [np.nan,np.nan,np.nan]
            
        else:
            T = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/T_{df_parameters.station.iloc[i]}.csv")
            P = np.genfromtxt(f"{drive}:/ordinary_events/{country_save}/P_{df_parameters.station.iloc[i]}.csv")
            times = pd.read_csv(f"{drive}:/ordinary_events/{country_save}/time_{df_parameters.station.iloc[i]}.csv",parse_dates = ["oe_time"])
            
            
            eT = np.arange(np.min(T),np.max(T)+4,0.1)
            season_separations = [5, 10]
            months = times["oe_time"].dt.month
            winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
            summer_inds = months.index[(months<season_separations[1]+1)&(months>season_separations[0]-1)]
            
            T_winter = T[winter_inds]
            T_summer = T[summer_inds]


            g_phat_winter[i] = S.temperature_model(T_winter,beta = 2)
            g_phat_summer[i] = S.temperature_model(T_summer,beta = 2)
            
            
            g_phat_winter_skew[i] = S.temperature_model(T_winter,method = "skewnorm")
            g_phat_summer_skew[i] = S.temperature_model(T_summer,method = "skewnorm")

            # Actual values kde
            kde  = gaussian_kde(T) #use kernel density to get probability
            prob = kde(eT)
            
            
            #Double fit
            hist, bin_edges = np.histogram(T, bins=100, density=True)
            xdata = (bin_edges[:-1] + bin_edges[1:]) / 2
            exp_sds = [g_phat_summer[i][1],g_phat_winter[i][1]]
            exp_means = [g_phat_summer[i][0],g_phat_winter[i][0]]
            expected=[exp_means[0],exp_sds[0],0.05,exp_means[1],exp_sds[0],0.05]
            g_phat_double_free[i],cov=curve_fit(bimodal,xdata,hist,expected)
            double_pdf = bimodal(eT,*g_phat_double_free[i])
            
            
            #basic summer/winter
            
            winter_pdf = gen_norm_pdf(eT, g_phat_winter[i][0], g_phat_winter[i][1], 2)
            summer_pdf = gen_norm_pdf(eT, g_phat_summer[i][0], g_phat_summer[i][1], 2)
            
            winter_pdf_skew = skewnorm.pdf(eT, *g_phat_winter_skew[i])
            summer_pdf_skew = skewnorm.pdf(eT, *g_phat_summer_skew[i])

            combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))
            combined_pdf_skew = (winter_pdf_skew*np.size(T_winter)+summer_pdf_skew*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))
            
            
            
            ## all the FRMSEs
            diff = double_pdf - prob
            FRMSE_doub[i] = np.sqrt(
                np.sum(diff**2)/len(diff))/(np.sum(prob)/len(diff))
            
            
            diff = combined_pdf - prob
            FRMSE_summer_winter[i] = np.sqrt(
                np.sum(diff**2)/len(diff))/(np.sum(prob)/len(diff))
            
            diff = combined_pdf_skew - prob
            FRMSE_summer_winter_skew[i] = np.sqrt(
                np.sum(diff**2)/len(diff))/(np.sum(prob)/len(diff))
            
            
            # now the upper 20%%
            min_T_upper = np.quantile(T,GOF_perc)
            
            eT_upper_perc = eT[eT>=min_T_upper]
            
            prob_upper_perc = prob[eT>=min_T_upper]
            
            
            pdf_values_upper_perc = double_pdf[eT>=min_T_upper]
            diff_upper_perc = pdf_values_upper_perc - prob_upper_perc
            FRMSE_upper_perc[i][0] = np.sqrt(
                np.sum(diff_upper_perc**2)/len(diff_upper_perc))/(np.sum(prob_upper_perc)/len(diff_upper_perc))
            
            pdf_values_upper_perc = combined_pdf[eT>=min_T_upper]
            diff_upper_perc = pdf_values_upper_perc - prob_upper_perc
            FRMSE_upper_perc[i][1] = np.sqrt(
                np.sum(diff_upper_perc**2)/len(diff_upper_perc))/(np.sum(prob_upper_perc)/len(diff_upper_perc))
            
            pdf_values_upper_perc = combined_pdf_skew[eT>=min_T_upper]
            diff_upper_perc = pdf_values_upper_perc - prob_upper_perc
            FRMSE_upper_perc[i][2] = np.sqrt(
                np.sum(diff_upper_perc**2)/len(diff_upper_perc))/(np.sum(prob_upper_perc)/len(diff_upper_perc))
            
            
            #log liklihoods
            log_liks[i][0] = np.sum(np.log(double_pdf))
            log_liks[i][1] = np.sum(np.log(combined_pdf))
            log_liks[i][2] = np.sum(np.log(combined_pdf_skew))
            
            if i%50 == 0:    
                print(log_liks[i])
                print(FRMSE_doub[i])
                time_taken = (time.time()-start_time[i-9])/10
                time_left = (len(new_df)-i)*time_taken/60
                print(f"{i}/{len(new_df)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
            else:
                pass
            
    FRMSE_double_df = pd.DataFrame({
        "station" : df_parameters.station,
        "FRMSE_double" : FRMSE_doub,
        "FRMSE_summer_winter" : FRMSE_summer_winter,
        "FRMSE_summer_winter_skew" : FRMSE_summer_winter_skew,
        "FRMSE_double_upper" : np.array(FRMSE_upper_perc)[:,0],
        "FRMSE_summer_winter_upper" : np.array(FRMSE_upper_perc)[:,1],
        "FRMSE_summer_winter_skew_upper" : np.array(FRMSE_upper_perc)[:,2],
        })
    
    log_liks_double_df = pd.DataFrame({
        "station" : df_parameters.station,
        "double" : np.array(log_liks)[:,0],
        "summer_winter" : np.array(log_liks)[:,1],
        "summer_winter_skew" : np.array(log_liks)[:,2],
        })
    
    g_phats_double_df = pd.DataFrame({
        "station" : df_parameters.station,
        "mu1" : np.array(g_phat_double_free)[:,0],
        "sigma1" : np.array(g_phat_double_free)[:,1],
        "A1" : np.array(g_phat_double_free)[:,2],
        "mu2" : np.array(g_phat_double_free)[:,3],
        "sigma2" : np.array(g_phat_double_free)[:,4],
        "A2" : np.array(g_phat_double_free)[:,5]
        })
    
    g_phats_summer_winter_df = pd.DataFrame({
        "station" : df_parameters.station,
        "mu_w" : np.array(g_phat_winter)[:,0],
        "sigma_w" : np.array(g_phat_winter)[:,1],
        "mu_s" : np.array(g_phat_summer)[:,0],
        "sigma_s" : np.array(g_phat_summer)[:,1],
        })
            
    g_phats_summer_winter_skew_df = pd.DataFrame({
        "station" : df_parameters.station,
        "skew_w" : np.array(g_phat_winter_skew)[:,0],
        "mu_w" : np.array(g_phat_winter_skew)[:,1],
        "sigma_w" : np.array(g_phat_winter_skew)[:,2],
        "skew_s" : np.array(g_phat_winter_skew)[:,0],
        "mu_s" : np.array(g_phat_summer_skew)[:,1],
        "sigma_s" : np.array(g_phat_summer_skew)[:,2],
        })

    FRMSE_double_df.to_csv(save_name,index = False)
    log_liks_double_df.to_csv('D:/outputs/Japan/doubles\\log_likelihood.csv',index = False)
    
    g_phats_double_df.to_csv('D:/outputs/Japan/doubles\\g_phat_double.csv',index = False)
    g_phats_summer_winter_df.to_csv('D:/outputs/Japan/doubles\\g_phat_summer_winter.csv',index = False)
    g_phats_summer_winter_skew_df.to_csv('D:/outputs/Japan/doubles\\g_phat_summer_winter_skew.csv',index = False)

else:
    FRMSE_double_df = pd.read_csv(save_name,dtype = {"station":str})
    log_liks_double_df = pd.read_csv('D:/outputs/Japan/doubles\\log_likelihood.csv',dtype = {"station":str})
    
    g_phats_double_df = pd.read_csv('D:/outputs/Japan/doubles\\g_phat_double.csv',dtype = {"station":str})
    g_phats_summer_winter_df = pd.read_csv('D:/outputs/Japan/doubles\\g_phat_summer_winter.csv',dtype = {"station":str})
    g_phats_summer_winter_skew_df = pd.read_csv('D:/outputs/Japan/doubles\\g_phat_summer_winter_skew.csv',dtype = {"station":str})




###############################################################################
# read other FRMSEs and AICs

skew_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE_skew.csv",dtype = {"station":str})
AIC_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\AIC.csv",dtype = {"station":str})


temp_FRMSE_df = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE.csv",dtype = {"station":str})
temp_FRMSE_df6 = pd.read_csv(f"{drive}:/outputs/{country_save}\\temp_FRMSE6.csv",dtype = {"station":str})

###############################################################################
# convert log_lik to AIC
n_params = {
    "double" : 6, #might actually be 5 because A1 and A2 depend on eachother
    "summer_winter" : 4,
    "summer_winter_skew" : 6,
    }


AIC_double = pd.concat([2 * n_params[name] - 2 * log_liks_double_df[name] for name in log_liks_double_df.columns[1:]],axis = 1)



###############################################################################
#plots

#AIC
AIC_df_sm = AIC_df.drop(columns = "station")
AIC_df_sm = pd.concat([AIC_double,AIC_df_sm],axis = 1) #combine all AIC dataframes
AIC_df_sm = AIC_df_sm.drop(columns = [string for string in AIC_df_sm.columns if "_20" in string ]) #drop the top 20% ones because they're bs
number_AIC = int(len(AIC_df_sm.columns))



skew_FRMSE_df.rename(columns={"FRMSE_upper_perc": "skew_upper", "FRMSE": "skew"},inplace = True)
temp_FRMSE_df.rename(columns={"FRMSE_upper_perc": "beta4_upper", "FRMSE": "beta4"},inplace = True)
temp_FRMSE_df6.rename(columns={"FRMSE_upper_perc": "beta6_upper", "FRMSE": "beta6"},inplace = True)


FRMSE_df_sm = pd.concat([FRMSE_double_df,skew_FRMSE_df,temp_FRMSE_df6,temp_FRMSE_df],axis = 1).drop(columns = ["station","difference"])
FRMSE_df_sm = FRMSE_df_sm.drop(columns = [string for string in FRMSE_df_sm.columns if "_upper" in string ]) #drop the top 20% ones because they're bs


proj = ccrs.PlateCarree()
cmap = "seismic"
norm = mcolors.Normalize(vmin=-200, vmax=200)
norm2 = mcolors.Normalize(vmin=-0.3, vmax=0.3)

AIC_headers = AIC_df_sm.columns.drop("AIC")
FRMSE_headers = FRMSE_df_sm.columns.drop("beta4")

#compare to beta = 4
fig,axs = plt.subplots(number_AIC - 1,2,figsize = (12,(number_AIC - 1)*6), subplot_kw={'projection': proj})

for n in range(number_AIC - 1):
    axs[n,0].coastlines()
    axs[n,0].add_feature(cfeature.BORDERS, linestyle=':')
    
    sc = axs[n,0].scatter(df_parameters.longitude,
                df_parameters.latitude,
                c = AIC_df_sm[AIC_headers[n]] - AIC_df_sm["AIC"],
                cmap=cmap,
                norm = norm,
                s = 3,
                )
    plt.colorbar(sc,extend = "both")
    axs[n,0].set_title(f"{AIC_headers[n]} - 4")
    
    axs[n,1].coastlines()
    axs[n,1].add_feature(cfeature.BORDERS, linestyle=':')
    
    sc = axs[n,1].scatter(df_parameters.longitude,
                df_parameters.latitude,
                c = FRMSE_df_sm[FRMSE_headers[n]] - FRMSE_df_sm["beta4"],
                cmap=cmap,
                norm = norm2,
                s = 3,
                )
    plt.colorbar(sc,extend = "both")
    axs[n,1].set_title(f"{FRMSE_headers[n]} - 4  FRMSE")
    
plt.show()


###############################################################################
# Doubles intercomparison

FRMSE_double_sm = FRMSE_double_df.drop(columns = ["station"])


AIC_headers = AIC_df_sm.columns.drop("AIC")
FRMSE_headers = FRMSE_df_sm.columns.drop("beta4")


n = len(AIC_double.columns)
num_plots = int((n * (n - 1)) / 2)
fig, axs = plt.subplots(num_plots, 1, figsize=(6, num_plots * 6), subplot_kw={'projection': proj})

axs = axs if num_plots > 1 else [axs]  # ensure axs is always iterable

plot_idx = 0
for i in range(n):
    for j in range(n):
        if i >= j:  # skip self-comparisons and duplicates
            continue

        ax = axs[plot_idx]
        plot_idx += 1

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        sc = ax.scatter(df_parameters.longitude,
                        df_parameters.latitude,
                        c=AIC_double[AIC_double.columns[i]] - AIC_double[AIC_double.columns[j]],
                        cmap=cmap,
                        norm=norm,
                        s=3)

        plt.colorbar(sc, extend="both")
        ax.set_title(f"{AIC_double.columns[i]} - {AIC_double.columns[j]}")

plt.show()




