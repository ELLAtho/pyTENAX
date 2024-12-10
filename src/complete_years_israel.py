# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:30:07 2024

@author: User
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
from pyTENAX.pyTENAX import *
import time 

from pyTENAX.intense import *
import glob
import datetime as dt
import matplotlib.pyplot as plt
import scipy.io

drive = 'D'

S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        tolerance=0.1,
    )
#TODO: make this so it creates pandas immediately with stations as strings and checks if file already made

country = 'Israel' #folder name that contains country data
name_col = "ppt"
year_size_limit = 3 # won't bother getting clean years if file smaller than this times percent missing

mat = scipy.io.loadmat('D:/gaugeData_10min')

def matlab_to_datetime(matlab_datenum):
    # MATLAB's reference date: January 0, 0000
    # Python's datetime reference date: January 1, 0001
    # The difference between MATLAB and Python reference dates is 366 days.
    
    py_date = dt.datetime.fromordinal(int(matlab_datenum)) + dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    py_date = (py_date+dt.timedelta(seconds=1)).replace(second=0, microsecond=0) #this rounds to nearest minute effectively
    
    return py_date

#datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)

names = [mat['R'][0,i][0][0] for i in np.arange(0,np.shape(mat['R'])[1])]
lats = [mat['R'][0,i][7][0][0] for i in np.arange(0,np.shape(mat['R'])[1])]
lons = [mat['R'][0,i][8][0][0] for i in np.arange(0,np.shape(mat['R'])[1])]
starts = [matlab_to_datetime(mat['R'][0,i][1][0][0]) for i in np.arange(0,np.shape(mat['R'])[1])]
ends = [matlab_to_datetime(mat['R'][0,i][1][-1][0]) for i in np.arange(0,np.shape(mat['R'])[1])]
total_years = [(mat['R'][0,i][1][-1][0]-mat['R'][0,i][1][0][0])/365.25 for i in np.arange(0,np.shape(mat['R'])[1])]


info = pd.DataFrame({'station':names, 'total_years':total_years,'latitude': lats,'longitude': lons,'startdate': starts,'enddate': ends})



 
start_time = time.time()
clean = np.zeros(np.shape(mat['R'])[1])

i=0 

for i in np.arange(0,np.shape(mat['R'])[1]): 

    G =  pd.DataFrame({'ppt':[mat['R'][0,i][2][n][0] for n in np.arange(0,len(mat['R'][0,i][2]))]})
                
    #extract start and end dates from metadata
    start_date_G = info.startdate[i]
    end_date_G= info.enddate[i]
 
    # replace -999 with nan
    G[G == -999] = np.nan
     
    G['prec_time'] = [matlab_to_datetime(mat['R'][0,i][1][n][0]) for n in np.arange(0,len(mat['R'][0,i][1]))]
    G = G.set_index('prec_time')
     
 
    data_clean = S.remove_incomplete_years(G, name_col) #remove incomplete years (below tolerance)
    clean[i] = np.size(np.unique(data_clean.index.year))
    
    file_name = f'{drive}:/Israel/10_min/{info.station[i]}.csv'
    G.to_csv(file_name)
        
    if (i-1) % 10 == 0: #i-1 to skip 0 
        print(i)
        print(info.station[i])
        print('Estimated time to complete: '+str((np.shape(mat['R'])[1]-i)*(time.time()-start_time)/(i*60))+' mins')
    i=i+1

info.cleaned_years = clean
    
print('time to loop through '+country,str(time.time()-start_time))


plt.hist(info.cleaned_years)
plt.title('Number of complete years '+country)
plt.show()

yrs_above_10 =  info.cleaned_years[info.cleaned_years>10]
yrs_above_20 =  info.cleaned_years[info.cleaned_years>20]

print('files longer than 20 years: '+str(np.size(yrs_above_20)))
print('files longer than 10 years: '+str(np.size(yrs_above_10)))
print('total files: '+str(np.size(info.cleaned_years)))


# num_above_10 = np.size(info[3][info[3]>10])
#filename = DE_+info[0,:]+.txt
    
    
#info_g = np.load('D:/germany_data.npy')







    