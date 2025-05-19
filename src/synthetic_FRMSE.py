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

from scipy.stats import norm

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
import glob


F_phat_typical = [ 1.1, -0.015, 1, 0.07] # basically Germany
g_phat_typical = [10, 13] # also basically Germany
attempt = 1 # IF YOU REPEAT WITH DIFFERENT PARAMETERS CHANGE THIS NUMBER

n_years = 30
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
        "mu": g_phat_typical[0],
        "sigma": g_phat_typical[1],
        'kappa':F_phat_typical[0],
        'b':F_phat_typical[1],
        'lambda':F_phat_typical[2],
        'a':F_phat_typical[3],
        'n_years':n_years,
        'n':n
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



F_phat_typical = [ 1.1, -0.015, 1, 0.07] # basically Germany
g_phat_typical = [10, 13] # also basically Germany
attempt = 1 # IF YOU REPEAT WITH DIFFERENT PARAMETERS CHANGE THIS NUMBER


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
    
    use_df_exp = pd.DataFrame({
        "mu": g_phat_typical[0],
        "sigma": g_phat_typical[1],
        'kappa':F_phat_typical[0],
        'b':F_phat_typical[1],
        'lambda':F_phat_typical[2],
        'a':F_phat_typical[3],
        'n_years':n_years,
        'n':n
        })
    use_df_exp.to_csv(f"D:/outputs/synthetic/parameters_set_exp{attempt}.csv", index = False)
else:    
    gen_F_phat_df_exp = pd.read_csv(f"D:/outputs/synthetic/gen_F_phat_exp{attempt}.csv")
    
    AMS_df_exp = pd.read_csv(f"D:/outputs/synthetic/gen_AMS_exp{attempt}.csv")
    
    RL_free_df_exp = pd.read_csv(f"D:/outputs/synthetic/RL_free_exp{attempt}.csv")
    
    RL_0_df_exp = pd.read_csv(f"D:/outputs/synthetic/RL_0_exp{attempt}.csv")
    
    RL_set_df_exp = pd.read_csv(f"D:/outputs/synthetic/RL_set_exp{attempt}.csv")
    
    use_df_exp = pd.read_csv(f"D:/outputs/synthetic/parameters_set_exp{attempt}.csv")
    
    
    
    
    
    
    
    
    
    
    
