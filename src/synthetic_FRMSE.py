# -*- coding: utf-8 -*-
"""
Created on Mon May 19 11:45:28 2025

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
import seaborn

from scipy.stats import norm

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob


F_phat_typical = [ 1.1, 0, 1, 0.07] # basically Germany
g_phat_typical = [10, 13] # also basically Germany
attempt = 5 # IF YOU REPEAT WITH DIFFERENT PARAMETERS CHANGE THIS NUMBER

n_years = 10
n = 80. # typical number of events per year

n_its = 1000

plot_pos = np.arange(1,n_years+1)/(1+n_years)

eRP = 1/(1-plot_pos)

S = TENAX(
        return_period = eRP,  
        durations = [60, 180],
        left_censoring = [0, 0.90],
        alpha = 0,
        n_monte_carlo = int(n * n_years), # total number of events (on average)
        
    )

Ts = np.arange(g_phat_typical[0]-2*g_phat_typical[1] - S.temp_delta, g_phat_typical[0]+2*g_phat_typical[1] + S.temp_delta, S.temp_res_monte_carlo)


save_name = f"D:/outputs/synthetic\\gen_F_phat{attempt}.csv"

if save_name not in glob.glob("D:/outputs/synthetic/*"):
    print("making linear")
    # linear b
    F_phat_new_free = [0]*n_its
    F_phat_new_0 = [0]*n_its
    F_phat_new_set = [0]*n_its
    
    
    generated_AMS = [0]*n_its
    
    
    RL_free = [0]*n_its
    RL_0 = [0]*n_its
    RL_set = [0]*n_its
    
    
    start_time = [0]*n_its
    
    for i in range(n_its):
        start_time[i] = time.time()
        
        _, T_mc, P_mc = S.model_inversion(F_phat_typical, g_phat_typical, n, Ts, gen_P_mc = True,gen_RL=False) 
        T_mc = T_mc.reshape(-1)
        thr_gen = np.nanquantile(P_mc,S.left_censoring[1])
        generated_AMS[i] = [np.max(P_mc[j:j+int(n)]) for j in np.arange(0,int(n_years*n),int(n))]
        
        
        # free b
        S.alpha = 0
        F_phat_new_free[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen)
        RL_free[i], __, __ = S.model_inversion(F_phat_new_free[i], g_phat_typical, n, Ts)
        
        # b = 0
        S.alpha = 1
        F_phat_new_0[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen)
        RL_0[i], __, __ = S.model_inversion(F_phat_new_0[i], g_phat_typical, n, Ts)
        
        # set b
        F_phat_new_set[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen, b_set = F_phat_typical[1])
        RL_set[i], __, __ = S.model_inversion(F_phat_new_set[i], g_phat_typical, n, Ts)
        
        if i%50 == 0:
            print(f"free b {F_phat_new_free[i]}")
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (n_its-i)*time_taken/60
            print(f"{i}/{n_its}. Approx time left: {time_left:.0f} mins")
    
    
    gen_F_phat_df = pd.DataFrame({
        'kappa_free':np.array(F_phat_new_free)[:,0],
        'b_free':np.array(F_phat_new_free)[:,1],
        'lambda_free':np.array(F_phat_new_free)[:,2],
        'a_free':np.array(F_phat_new_free)[:,3],
        'kappa_0':np.array(F_phat_new_0)[:,0],
        'b_0':np.array(F_phat_new_0)[:,1],
        'lambda_0':np.array(F_phat_new_0)[:,2],
        'a_0':np.array(F_phat_new_0)[:,3],
        'kappa_set':np.array(F_phat_new_set)[:,0],
        'b_set':np.array(F_phat_new_set)[:,1],
        'lambda_set':np.array(F_phat_new_set)[:,2],
        'a_set':np.array(F_phat_new_set)[:,3],
        })
    gen_F_phat_df.to_csv(f"D:/outputs/synthetic/gen_F_phat{attempt}.csv", index = False)
    
    generated_AMS_sort = [np.sort(generated_AMS[j]) for j in range(n_its)]
    
    AMS_df = pd.DataFrame(
        generated_AMS_sort, columns = eRP
        )
    AMS_df.to_csv(f"D:/outputs/synthetic/gen_AMS{attempt}.csv", index = False)
    
    
    RL_free_df = pd.DataFrame(
        RL_free, columns = eRP
        )
    RL_free_df.to_csv(f"D:/outputs/synthetic/RL_free{attempt}.csv", index = False)
    
    
    RL_0_df = pd.DataFrame(
        RL_0, columns = eRP
        )
    RL_0_df.to_csv(f"D:/outputs/synthetic/RL_0{attempt}.csv", index = False)
    
    
    RL_set_df = pd.DataFrame(
        RL_set, columns = eRP
        )
    RL_set_df.to_csv(f"D:/outputs/synthetic/RL_set{attempt}.csv", index = False)
    
    use_df = pd.DataFrame({
        "mu": [g_phat_typical[0]],
        "sigma": [g_phat_typical[1]],
        'kappa': [F_phat_typical[0]],
        'b': [F_phat_typical[1]],
        'lambda': [F_phat_typical[2]],
        'a': [F_phat_typical[3]],
        'n_years': [n_years],
        'n': [n]
    })
    use_df.to_csv(f"D:/outputs/synthetic/parameters_set{attempt}.csv", index = False)
else:
    gen_F_phat_df = pd.read_csv(f"D:/outputs/synthetic/gen_F_phat{attempt}.csv")
    
    AMS_df = pd.read_csv(f"D:/outputs/synthetic/gen_AMS{attempt}.csv")
    
    RL_free_df = pd.read_csv(f"D:/outputs/synthetic/RL_free{attempt}.csv")
    
    RL_0_df = pd.read_csv(f"D:/outputs/synthetic/RL_0{attempt}.csv")
    
    RL_set_df = pd.read_csv(f"D:/outputs/synthetic/RL_set{attempt}.csv")
    
    use_df = pd.read_csv(f"D:/outputs/synthetic/parameters_set{attempt}.csv")




# exponential b


save_name_exp = f"D:/outputs/synthetic\\gen_F_phat_exp{attempt}.csv"

if save_name_exp not in glob.glob("D:/outputs/synthetic/*"):
    print("making exp")
    
    F_phat_new_free_exp = [0]*n_its
    F_phat_new_0_exp = [0]*n_its
    F_phat_new_set_exp = [0]*n_its
    
    
    generated_AMS_exp = [0]*n_its
    
    
    RL_free_exp = [0]*n_its
    RL_0_exp = [0]*n_its
    RL_set_exp = [0]*n_its
    
    
    start_time = [0]*n_its
    
    for i in range(n_its):
        start_time[i] = time.time()
        
        _, T_mc, P_mc = S.model_inversion(F_phat_typical, g_phat_typical, n, Ts, gen_P_mc = True,gen_RL=False,b_exp = True) 
        T_mc = T_mc.reshape(-1)
        thr_gen = np.nanquantile(P_mc,S.left_censoring[1])
        generated_AMS_exp[i] = [np.max(P_mc[j:j+int(n)]) for j in np.arange(0,int(n_years*n),int(n))]
        
        
        # free b
        S.alpha = 0
        F_phat_new_free_exp[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen,b_exp = True)
        RL_free_exp[i], __, __ = S.model_inversion(F_phat_new_free_exp[i], g_phat_typical, n, Ts,b_exp = True)
        
        # b = 0
        S.alpha = 1
        F_phat_new_0_exp[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen)
        RL_0_exp[i], __, __ = S.model_inversion(F_phat_new_0_exp[i], g_phat_typical, n, Ts)
        
        # set b
        S.alpha = 0
        F_phat_new_set_exp[i], loglik, _, _ = S.magnitude_model(P_mc, T_mc, thr_gen, b_set = F_phat_typical[1],b_exp = True)
        RL_set_exp[i], __, __ = S.model_inversion(F_phat_new_set_exp[i], g_phat_typical, n, Ts,b_exp = True)
        
        if i%50 == 0:
            print(f"free b {F_phat_new_free_exp[i]}")
            print(f"set b {F_phat_new_set_exp[i]}")
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (n_its-i)*time_taken/60
            print(f"{i}/{n_its}. Approx time left: {time_left:.0f} mins")
    
    
    gen_F_phat_df_exp = pd.DataFrame({
        'kappa_free':np.array(F_phat_new_free_exp)[:,0],
        'b_free':np.array(F_phat_new_free_exp)[:,1],
        'lambda_free':np.array(F_phat_new_free_exp)[:,2],
        'a_free':np.array(F_phat_new_free_exp)[:,3],
        'kappa_0':np.array(F_phat_new_0_exp)[:,0],
        'b_0':np.array(F_phat_new_0_exp)[:,1],
        'lambda_0':np.array(F_phat_new_0_exp)[:,2],
        'a_0':np.array(F_phat_new_0_exp)[:,3],
        'kappa_set':np.array(F_phat_new_set_exp)[:,0],
        'b_set':np.array(F_phat_new_set_exp)[:,1],
        'lambda_set':np.array(F_phat_new_set_exp)[:,2],
        'a_set':np.array(F_phat_new_set_exp)[:,3],
        })
    gen_F_phat_df_exp.to_csv(f"D:/outputs/synthetic/gen_F_phat_exp{attempt}.csv", index = False)
    
    generated_AMS_sort_exp = [np.sort(generated_AMS_exp[j]) for j in range(n_its)]
    
    AMS_df_exp = pd.DataFrame(
        generated_AMS_sort_exp, columns = eRP
        )
    AMS_df_exp.to_csv(f"D:/outputs/synthetic/gen_AMS_exp{attempt}.csv", index = False)
    
    
    RL_free_df_exp = pd.DataFrame(
        RL_free_exp, columns = eRP
        )
    RL_free_df_exp.to_csv(f"D:/outputs/synthetic/RL_free_exp{attempt}.csv", index = False)
    
    
    RL_0_df_exp = pd.DataFrame(
        RL_0_exp, columns = eRP
        )
    RL_0_df.to_csv(f"D:/outputs/synthetic/RL_0_exp{attempt}.csv", index = False)
    
    
    RL_set_df_exp = pd.DataFrame(
        RL_set_exp, columns = eRP
        )
    RL_set_df_exp.to_csv(f"D:/outputs/synthetic/RL_set_exp{attempt}.csv", index = False)
    
    # use_df_exp = pd.DataFrame({
    #     "mu": [g_phat_typical[0]],
    #     "sigma": [g_phat_typical[1]],
    #     'kappa': [F_phat_typical[0]],
    #     'b': [F_phat_typical[1]],
    #     'lambda': [F_phat_typical[2]],
    #     'a': [F_phat_typical[3]],
    #     'n_years': [n_years],
    #     'n': [n]
    # })
    # use_df_exp.to_csv(f"D:/outputs/synthetic/parameters_set_exp{attempt}.csv", index = False)
else:    
    gen_F_phat_df_exp = pd.read_csv(f"D:/outputs/synthetic/gen_F_phat_exp{attempt}.csv")
    
    AMS_df_exp = pd.read_csv(f"D:/outputs/synthetic/gen_AMS_exp{attempt}.csv")
    
    RL_free_df_exp = pd.read_csv(f"D:/outputs/synthetic/RL_free_exp{attempt}.csv")
    
    RL_0_df_exp = pd.read_csv(f"D:/outputs/synthetic/RL_0_exp{attempt}.csv")
    
    RL_set_df_exp = pd.read_csv(f"D:/outputs/synthetic/RL_set_exp{attempt}.csv")
    
    # use_df_exp = pd.read_csv(f"D:/outputs/synthetic/parameters_set_exp{attempt}.csv")
    
#calculate specific RLs and compare to expected from given F_phat etc.
  
S.return_period = [10,20,50,100]
S.n_monte_carlo = 20000
#calculate expected from input data

RL_typical, _, _ = S.model_inversion(F_phat_typical, g_phat_typical, n, Ts)

save_name_RL = f"D:/outputs/synthetic\\RL_specific{attempt}.csv"

if save_name_RL not in glob.glob("D:/outputs/synthetic/*"):
    print("calculating specific RLs")
    
    
    RL_10 = [0]*n_its
    RL_20 = [0]*n_its
    RL_50 = [0]*n_its
    RL_100 = [0]*n_its
    
    RL_10_exp = [0]*n_its
    RL_20_exp = [0]*n_its
    RL_50_exp = [0]*n_its
    RL_100_exp = [0]*n_its
    
    start_time = [0]*n_its
    
    for i in range(n_its):
        start_time[i] = time.time()
        
        RL_10_now = []
        RL_20_now = []
        RL_50_now = []
        RL_100_now = []
        for b_type in ["free","0","set"]:
            F_phat_now = [gen_F_phat_df[f"kappa_{b_type}"].iloc[i],
                          gen_F_phat_df[f"b_{b_type}"].iloc[i],
                          gen_F_phat_df[f"lambda_{b_type}"].iloc[i],
                          gen_F_phat_df[f"a_{b_type}"].iloc[i]]
            RL_full,_,_ = S.model_inversion(F_phat_now, g_phat_typical, n, Ts)
            
            RL_10_now.append(RL_full[0])
            RL_20_now.append(RL_full[1])
            RL_50_now.append(RL_full[2])
            RL_100_now.append(RL_full[3])
        
        
        RL_10[i] = RL_10_now
        RL_20[i] = RL_20_now
        RL_50[i] = RL_50_now
        RL_100[i] = RL_100_now
        
        #exp
        RL_10_now_exp = []
        RL_20_now_exp = []
        RL_50_now_exp = []
        RL_100_now_exp = []
        for b_type in ["free","0","set"]:
            F_phat_now = [gen_F_phat_df_exp[f"kappa_{b_type}"].iloc[i],
                          gen_F_phat_df_exp[f"b_{b_type}"].iloc[i],
                          gen_F_phat_df_exp[f"lambda_{b_type}"].iloc[i],
                          gen_F_phat_df_exp[f"a_{b_type}"].iloc[i]]
            RL_full_exp,_,_ = S.model_inversion(F_phat_now, g_phat_typical, n, Ts,b_exp=True)
            
            RL_10_now_exp.append(RL_full_exp[0])
            RL_20_now_exp.append(RL_full_exp[1])
            RL_50_now_exp.append(RL_full_exp[2])
            RL_100_now_exp.append(RL_full_exp[3])
        
        
        RL_10_exp[i] = RL_10_now_exp
        RL_20_exp[i] = RL_20_now_exp
        RL_50_exp[i] = RL_50_now_exp
        RL_100_exp[i] = RL_100_now_exp
        
        if i%50 == 0:
            print(f"RL_10  {RL_10[i]}")
            print(f"RL_10 exp {RL_10_exp[i]}")
            time_taken = (time.time()-start_time[i-9])/10
            time_left = (n_its-i)*time_taken/60
            print(f"{i}/{n_its}. Approx time left: {time_left:.0f} mins")
        
        
    RL_spec = pd.DataFrame({
        "free_10": np.array(RL_10)[:,0],
        "b0_10": np.array(RL_10)[:,1],
        "set_10": np.array(RL_10)[:,2],
        
        "free_20": np.array(RL_20)[:,0],
        "b0_20": np.array(RL_20)[:,1],
        "set_20": np.array(RL_20)[:,2],
        
        "free_50": np.array(RL_50)[:,0],
        "b0_50": np.array(RL_50)[:,1],
        "set_50": np.array(RL_50)[:,2],
        
        "free_100": np.array(RL_100)[:,0],
        "b0_100": np.array(RL_100)[:,1],
        "set_100": np.array(RL_100)[:,2],
        
        
        "free_10_exp": np.array(RL_10_exp)[:,0],
        "b0_10_exp": np.array(RL_10_exp)[:,1],
        "set_10_exp": np.array(RL_10_exp)[:,2],
        
        "free_20_exp": np.array(RL_20_exp)[:,0],
        "b0_20_exp": np.array(RL_20_exp)[:,1],
        "set_20_exp": np.array(RL_20_exp)[:,2],
        
        "free_50_exp": np.array(RL_50_exp)[:,0],
        "b0_50_exp": np.array(RL_50_exp)[:,1],
        "set_50_exp": np.array(RL_50_exp)[:,2],
        
        "free_100_exp": np.array(RL_100_exp)[:,0],
        "b0_100_exp": np.array(RL_100_exp)[:,1],
        "set_100_exp": np.array(RL_100_exp)[:,2],
        
        })
    RL_spec.to_csv(save_name_RL,index = False)
else:
    RL_spec = pd.read_csv(save_name_RL)
        
           




#FRMSE

#linear
diffs = RL_free_df - AMS_df
diffs_0 = RL_0_df - AMS_df  
diffs_set = RL_set_df - AMS_df


gen_FRMSE_df = pd.DataFrame({
    "free": np.sqrt(np.sum(diffs**2,axis = 1)/n_years)/(np.sum(AMS_df,axis = 1)/n_years),
    "b0": np.sqrt(np.sum(diffs_0**2,axis = 1)/n_years)/(np.sum(AMS_df,axis = 1)/n_years),
    "set": np.sqrt(np.sum(diffs_set**2,axis = 1)/n_years)/(np.sum(AMS_df,axis = 1)/n_years)
    })


gen_MAE_df = pd.DataFrame({
    "free": np.sum(diffs,axis = 1)/n_years,
    "b0": np.sum(diffs_0,axis = 1)/n_years,
    "set": np.sum(diffs_set,axis = 1)/n_years
    })

plt.violinplot([gen_FRMSE_df.free,gen_FRMSE_df.b0,gen_FRMSE_df.set],vert = False,showmeans=True)
plt.yticks(np.arange(1,4),["free","b = 0","set"])
plt.xlim(0,1)
plt.title("FRMSE between simulated annual maxima and calculated return levels. linear")
plt.show()

plt.violinplot([gen_MAE_df.free,gen_MAE_df.b0,gen_MAE_df.set],vert = False,showmeans=True)
plt.yticks(np.arange(1,4),["free","b = 0","set"])
plt.xlim(-9,7)
plt.title("Mean absolute error between simulated annual maxima and calculated return levels. linear")
plt.show()

#exp
diffs_exp = RL_free_df_exp - AMS_df_exp
diffs_0_exp = RL_0_df_exp - AMS_df_exp  
diffs_set_exp = RL_set_df_exp - AMS_df_exp


gen_FRMSE_df_exp = pd.DataFrame({
    "free": np.sqrt(np.sum(diffs_exp**2,axis = 1)/n_years)/(np.sum(AMS_df_exp,axis = 1)/n_years),
    "b0": np.sqrt(np.sum(diffs_0_exp**2,axis = 1)/n_years)/(np.sum(AMS_df_exp,axis = 1)/n_years),
    "set": np.sqrt(np.sum(diffs_set_exp**2,axis = 1)/n_years)/(np.sum(AMS_df_exp,axis = 1)/n_years)
    })


gen_MAE_df_exp = pd.DataFrame({
    "free": np.sum(diffs_exp,axis = 1)/n_years,
    "b0": np.sum(diffs_0_exp,axis = 1)/n_years,
    "set": np.sum(diffs_set_exp,axis = 1)/n_years
    })


plt.violinplot([gen_FRMSE_df_exp.free,gen_FRMSE_df_exp.b0,gen_FRMSE_df_exp.set],vert = False,showmeans=True)
plt.yticks(np.arange(1,4),["free","b = 0","set"])
plt.title("FRMSE between simulated annual maxima and calculated return levels. exponential")
plt.xlim(0,1)
plt.show()

plt.violinplot([gen_MAE_df_exp.free,gen_MAE_df_exp.b0,gen_MAE_df_exp.set],vert = False,showmeans=True)
plt.yticks(np.arange(1,4),["free","b = 0","set"])
plt.title("Mean absolute error between simulated annual maxima and calculated return levels. exponential")
plt.xlim(-9,7)
plt.show()



# load non-sythnetic to compare

country_save = "Japan"

FRMSE_df = pd.read_csv(f"D:/outputs/{country_save}/FRMSE.csv", dtype={'station': str})

RL_df = pd.read_csv(f"D:/outputs/{country_save}\\return_levels.csv", dtype={'station': str})


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


RL_df.dropna(axis = 0,inplace = True)
FRMSE_df.dropna(axis = 0,inplace = True)



lens = [len(RL_df.obs_AMS.iloc[j]) for j in range(len(RL_df))]





#RL_df.drop(464,inplace = True)


diffs = RL_df.return_levels_kernal - RL_df.obs_AMS
diffs_0 = RL_df.return_levels_kernal_0 - RL_df.obs_AMS
diffs_exp = RL_df.return_levels_kernal_exp - RL_df.obs_AMS




MAE_df = pd.DataFrame({
    "free": [np.sum(diffs.iloc[j])/lens[j] for j in range(len(RL_df))],
    "b0": [np.sum(diffs_0.iloc[j])/lens[j] for j in range(len(RL_df))],
    "exp": [np.sum(diffs_exp.iloc[j])/lens[j] for j in range(len(RL_df))]
    })

plt.violinplot([MAE_df.free.dropna(),MAE_df.b0,MAE_df.exp],vert = False,showmeans=True)
plt.yticks(np.arange(1,4),["free","b = 0","exp"])
plt.title(f"MAE {country_save}")
plt.xlim(-9,7)
plt.show()


plt.violinplot([FRMSE_df.FRMSE,FRMSE_df.FRMSE_0,FRMSE_df.FRMSE_bexp],vert = False,showmeans=True)
plt.yticks(np.arange(1,4),["free","b = 0","exp"])
plt.xlim(0,1)
plt.title(f"FRMSE {country_save}")
plt.show()




seaborn.stripplot([0,gen_FRMSE_df_exp.free,gen_FRMSE_df_exp.b0,gen_FRMSE_df_exp.set],alpha = 0.2,s = 2, color = "r")
plt.violinplot([gen_FRMSE_df_exp.free,gen_FRMSE_df_exp.b0,gen_FRMSE_df_exp.set],showmeans=True)
plt.xticks(np.arange(1,4),["free","b = 0","set"])
plt.title("FRMSE between simulated annual maxima and calculated return levels. exponential")
plt.show()


print("SIMULATED")
print("free linear")
print(f"mean FRMSE: {np.mean(gen_FRMSE_df.free):.3f}. Standard deviation FRMSE: {np.std(gen_FRMSE_df.free):.3f}")
print(f"mean MAE: {np.mean(gen_MAE_df.free):.3f}. Standard deviation MAE: {np.std(gen_MAE_df.free):.3f}")
print("set linear")
print(f"mean FRMSE: {np.mean(gen_FRMSE_df.set):.3f}. Standard deviation FRMSE: {np.std(gen_FRMSE_df.set):.3f}")
print(f"mean MAE: {np.mean(gen_MAE_df.set):.3f}. Standard deviation MAE: {np.std(gen_MAE_df.set):.3f}")
print("0")
print(f"mean FRMSE: {np.mean(gen_FRMSE_df.b0):.3f}. Standard deviation FRMSE: {np.std(gen_FRMSE_df.b0):.3f}")
print(f"mean MAE: {np.mean(gen_MAE_df.b0):.3f}. Standard deviation MAE: {np.std(gen_MAE_df.b0):.3f}")


print(f"OBSERVED {country_save}")
print("free linear")
print(f"mean FRMSE: {np.mean(FRMSE_df.FRMSE):.3f}. Standard deviation FRMSE: {np.std(FRMSE_df.FRMSE):.3f}")
print(f"mean MAE: {np.mean(MAE_df.free):.3f}. Standard deviation MAE: {np.std(MAE_df.free):.3f}")

print("0")
print(f"mean FRMSE: {np.mean(FRMSE_df.FRMSE_0):.3f}. Standard deviation FRMSE: {np.std(FRMSE_df.FRMSE_0):.3f}")
print(f"mean MAE: {np.mean(MAE_df.b0):.3f}. Standard deviation MAE: {np.std(MAE_df.b0):.3f}")


