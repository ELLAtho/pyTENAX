# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:26:02 2025

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

import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *
from pyTENAX.smev_class import *
import xarray as xr
import time

from scipy.stats import linregress

# def wbl_tail_test_func(samples, oe_times, left_cens_prctile_thr, p_test, niter, censorams):
    
#     years = oe_times.dt.year
#     df = pd.DataFrame({"year": years, "P": samples})
#     df = df.sort_values(by = "P")
    
#     is_block_maximum = np.zeros_like(samples, dtype=bool) #make a bool list of the locations of annual maxima
#     for i in np.unique(years):
#         df_block = df.P[df.year == i]
#         max_index = df_block.idxmax()
#         is_block_maximum[max_index] = True
    
#     thr = df.P.quantile(left_cens_prctile_thr)
#     df_use = df[df.P >= thr]
#     is_block_maximum_use = is_block_maximum[df.P >= thr]
    
#     if censorams:
#         df_use = df_use[~is_block_maximum_use]
    
#     ECDF = np.arange(0,len(samples))/(len(samples)+1)
#     X = np.log(np.log(1./(1-ECDF[df_use.index])))   # Weibull-tranformation for probabilities
#     Y = np.log(samples[df_use.index])  # Weibull-tranformation for samples
#     slope, intercept, r_value, p_value, std_err = linregress(X, Y) #linear regression
#     scale = np.exp(intercept);         #Weibull scale parameter
#     shape = 1/slope
    
    
#     wblinv = scale * (- np.log(1-np.random.rand(niter,len(samples))))**(1 / shape)
#     randy = np.sort(wblinv)    #/this is wrong      #weibull-distributed stochastic samples
#     p_lo = nanmean( samples(istest)<quantile(randy(:,istest),p_test/2,1) );    % fraction of block maxima below the (1-p) CI
#                    p_hi = nanmean( samples(istest)>quantile(randy(:,istest),1-p_test/2,1)' );  % fraction of block maxima above the (1-p) CI

#     p_out = p_hi + p_lo; % fraction of block maxima out of the (1-p) CI
 
    
    
#     return is_rejected, p_out, p_hi, p_lo, scale, shape

def create_syntethic_records(seed_random, synthetic_records_amount, record_size, shape, scale):
    '''--------------------------------------------------------------------------
    Function that generates synthetic records using the Weibull parameters which were
    estimated based on original record (without AM).
    The synthetic records contain random ordinary events sampled uniformly from the Weibull distribution.  
    These synthetic records use as the basis for extracting the confidence interval.
    
    
    Arguments:
    - seed_random (int): Value that determines the starting point for the pseudorandom number generator
    - synthetic_records_amount (int): Value that determines how many synthetic records to generate
    - record_size (int): The number of ordinary events in the record
    - shape (float): Weibull distribution parameter
    - scale (float): Weibull distribution parameter
    
    Returns:
    - records_df (dataframe): df with all the synthetic records. 
      Each row represents separate synthetic record.
    -----------------------------------------------------------------------------'''
    
    # Generate random array of probability values between 0 to 1, uniformly sampled
    np.random.seed(seed_random)  # Set the seed
    random_array = np.random.uniform(0, 1, synthetic_records_amount * record_size) 

    # Calaulate quantiles & create records matrix
    random_ordinary_events = []  
    for p in random_array:

        intensity = scale*((-1)*(np.log(1-p)))**(1/shape) 

        random_ordinary_events.append(intensity)

    records_matrix = np.array(random_ordinary_events).reshape(synthetic_records_amount, record_size) # convert vector to matrix

    records_matrix = np.sort(records_matrix, axis=1) # sort each row

    records_df = pd.DataFrame(records_matrix) 
    
    return records_df


def check_confidence_interval(annual_max_indexes, records_df, p_confidence, annual_max, censor_value, p_out_dicts_lst):
    '''--------------------------------------------------------------------------
    Function that checks the fraction of the annual/block maxima that are out of the confidence interval.
    
    Arguments:
    - annual_max_indexes (list): List of indexes in the record of the annual/block maxima
    - records_df (dataframe): df with all the synthetic records
    - p_confidence (float): Probability to be used for the test. confidence interval = 1-p_confidence 
    - annual_max (list): List of values over which the hypothesis is tested, i.e. block maxima
    - censor_value (float): The threshold for left censoring the record 
    - p_out_dicts_lst (list): List of dicts - Each censor value tested gets a dict as follow: {censor_value:p_out}
    
    Returns:
    - p_out_dicts_lst (list): Same list as in the arguments, after appending dict fot the tested censor_value
    -----------------------------------------------------------------------------'''
    
    p_lo = 0
    p_hi = 0

    counter_index = 0
    # Iterate over AM values
    for index in annual_max_indexes: 
        column =  records_df.iloc[:,index] # Select from each synthetic record the value in the position of the AM tested value
        
        # Create confidence interval
        lower = p_confidence/2
        upper = 1-(p_confidence/2)
        quantiles = column.quantile([lower, upper])
        quantile_lower = quantiles.iloc[0] # Lower value of confidence interval
        quantile_upper = quantiles.iloc[1] # Upper value of confidence interval
        
        # Select AM value to test
        annual_max_value = annual_max[counter_index]

        counter_index +=1 
        
        # Test if AM is within the confidence interval - i.e count how many times AM value is out of confidence interval  
        if annual_max_value<quantile_lower :
            p_lo += 1
        elif annual_max_value>quantile_upper :
            p_hi += 1

    p_out = p_hi/len(annual_max) + p_lo/len(annual_max) # fraction of block maxima out of the (1-p) CI        

    p_out_dict = {round(censor_value,2):round(p_out,2)}

    p_out_dicts_lst.append(p_out_dict)
    
    return p_out_dicts_lst


def find_optimal_threshold(p_out_dicts_lst, p_confidence):
    '''--------------------------------------------------------------------------
    Function that finds the optimal threshold out of the list of dicts.
    The function returns the minimial threshold from which p_out <= p_confidence for all bigger thresholds.
    If all threshold rejected - it will return 0.95 
    
    Arguments:
    - p_out_dicts_lst (list): List of dicts for each of the censor values tested, as follow: {censor_value:p_out}
    - p_confidence (float): Probability to be used for the test. confidence interval = 1-p_confidence 
    
    Returns:
    - optimal_threshold (float): The minimal threshold from which p_out <= p_confidence for all bigger thresholds.
                                 If all threshold rejected - it will return 0.95. If not all thresholds rejected,
                                 (1-optimal_threshold) is the portion of the record that can be assumed to be 
                                 distributed Weibull.
      
    -----------------------------------------------------------------------------'''
    
    p_out_lst = []
    thresholds_lst = []
    
    # Get values from p_out_dicts - thresholds and their corresponding p_out  
    for dic in p_out_dicts_lst:
        p_out_lst.append(list(dic.values())[0])
        thresholds_lst.append(list(dic.keys())[0])
    
    # Get indexes of all thresholds that are rejected
    indexes_rejected = [index for index, p_out in enumerate(p_out_lst) if p_out > p_confidence]
    
    if len(indexes_rejected)>0 : 
        if len(indexes_rejected)==len(thresholds_lst): 
            optimal_threshold = 1
            #All thresholds rejected    
        else:
            #some thresholds rejected and some not
            index_to_use = indexes_rejected[-1]+1 # Select the next threshold after the biggest one that was rejected
            
            if index_to_use<len(thresholds_lst):
                optimal_threshold = thresholds_lst[index_to_use]
            else:
                optimal_threshold = 1

    else:
        optimal_threshold = thresholds_lst[0] 
        # No threshold rejected
        
    return optimal_threshold 

country = 'Japan'
ERA_country = 'Japan'
country_save = 'Japan'
code_str = 'JP'
minlat,minlon,maxlat,maxlon = 24, 122.9, 45.6, 145.8 #JAPAN
name_len = 5
min_startdate = dt.datetime(1900,1,1) #this is for if havent read all ERA5 data yet
name_col = 'ppt'
temp_name_col = "t2m"
min_yrs = 10


# country = 'Japan'
# country_save = 'Japan'
# code_str = 'JP' 
# n_stations = 10 #number of stations to sample
# min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
# max_yrs = 1000 #if no max, set to very high
chosen_station = "12441"

# country = 'germany'
# country_save = 'germany'
# code_str = 'DE' 
# n_stations = 10 #number of stations to sample
# min_yrs = 15 #atm this probably introduces a bug... need to put in if statement or something
# max_yrs = 1000 #if no max, set to very high
# name_col = 'ppt'
# temp_name_col = "t2m"
# chosen_station = "00020"

T = np.genfromtxt(f"D:/ordinary_events/{country_save}/T_{chosen_station}.csv")
P = np.genfromtxt(f"D:/ordinary_events/{country_save}/P_{chosen_station}.csv")
times = pd.read_csv(f"D:/ordinary_events/{country_save}/time_{chosen_station}.csv",parse_dates = ["oe_time"])

oe_df = pd.DataFrame({"year":times.oe_time.dt.year, "P": P, "T": T,})
AMS = oe_df.groupby(oe_df.year).P.max()

AMS_indices = oe_df.groupby("year").P.idxmax()

S = TENAX(
        return_period = [2,5,10,20,50,100, 200],  #for some reason it doesnt like calculating RP =<1
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
    )

S_SMEV = SMEV(threshold=0.1,
              separation = 24,
              return_period = S.return_period,
              durations = S.durations,
              time_resolution = 5, #time resolution in minutes
              min_duration = 30 ,
              left_censoring = [S.left_censoring[1],1])      

#estimate shape and  scale parameters of weibull distribution
shape,scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)


oe_sort_df = oe_df.sort_values(by="P")
oe_sort_df = oe_sort_df.reset_index()

AMS_indices = oe_sort_df.groupby("year").P.idxmax()

records_df = create_syntethic_records(seed_random = 0, synthetic_records_amount = 100, record_size = len(oe_sort_df), shape = shape, scale = scale)

p_out_dicts_lst = []
for thresh in np.arange(0.8,1,0.01):
    p_out_dicts_lst = check_confidence_interval(AMS_indices, records_df, 0.1, AMS.to_numpy(), thresh, p_out_dicts_lst)

optimal_threshold = find_optimal_threshold(p_out_dicts_lst, 0.1)

## loop to get the optimal threshold for all

info = pd.read_csv(f"D:/metadata/{country}_fulldata.csv", dtype={'station': str})

info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)
val_info = info[info['cleaned_years']>=min_yrs] #filter out stations that are less than min

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


min_thr_savename = f"D://outputs/{country_save}/weibull_threshold.csv"
output_files = glob.glob(f"D:/outputs/{country_save}/*")

if min_thr_savename not in output_files:
    print("test not yet run")
    
    optimal_thresholds = np.zeros(len(val_info))
    all_P = [0]*len(val_info)
    start_time = [0]*len(val_info)
    for i in range(len(val_info)):
        station = val_info.iloc[i].station
        start_time[i] = time.time()
        
        oe_save = f"D:/ordinary_events/{country_save}\\T_{station}.csv"
        if oe_save not in glob.glob(f"D:/ordinary_events/{country_save}/*"):
            optimal_thresholds[i] = 0
        else:
            T = np.genfromtxt(f"D:/ordinary_events/{country_save}/T_{station}.csv")
            P = np.genfromtxt(f"D:/ordinary_events/{country_save}/P_{station}.csv")
            times = pd.read_csv(f"D:/ordinary_events/{country_save}/time_{station}.csv",parse_dates = ["oe_time"])
            
            oe_df = pd.DataFrame({"year":times.oe_time.dt.year, "P": P, "T": T,})
            AMS = oe_df.groupby(oe_df.year).P.max()
            
            
    
            oe_sort_df = oe_df.sort_values(by="P")
            oe_sort_df = oe_sort_df.reset_index()
    
            AMS_indices = oe_sort_df.groupby("year").P.idxmax()
    
            
            p_out_dicts_lst = []
            for thresh in np.concatenate([np.arange(0,0.8,0.1),np.arange(0.8,1,0.01)]):
                shape,scale = S_SMEV.estimate_smev_parameters(P, [thresh,1])
        
                records_df = create_syntethic_records(seed_random = 0, synthetic_records_amount = 1000, record_size = len(oe_sort_df), shape = shape, scale = scale)
                
                p_out_dicts_lst = check_confidence_interval(AMS_indices, records_df, 0.1, AMS.to_numpy(), thresh, p_out_dicts_lst)
    
        all_P[i] = p_out_dicts_lst
        optimal_thresholds[i] = find_optimal_threshold(p_out_dicts_lst, 0.1)
        
        
    
        time_taken = (time.time()-start_time[i-9])/10
        time_left = (len(val_info)-i)*time_taken/60
        print(f"{i}/{len(val_info)}. Current average time to complete one {time_taken:.0f}s. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
    
    
    thresh_df = pd.DataFrame({"station":val_info.station,
                              "optimal_threshold":optimal_thresholds
        })
    thresh_df.to_csv(min_thr_savename,index=False)



