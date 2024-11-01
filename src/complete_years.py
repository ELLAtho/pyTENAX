# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:19:29 2024

@author: ellar
"""


## NEED TO RUN THE IMPORTS IN /SRC FOLDER THEN GO INTO DRIVE FOLDER

import os
# os.environ['USE_PYGEOS'] = '0'
from os.path import dirname, abspath, join
from os import getcwd
import sys
#run this fro src folder, otherwise it doesn't work
THIS_DIR = dirname(getcwd())
CODE_DIR = join(THIS_DIR, 'src')
RES_DIR =  join(THIS_DIR, 'res')
sys.path.append(CODE_DIR)
sys.path.append(RES_DIR)
import numpy as np
import pandas as pd
from pyTENAX.pyTENAX import *
import time 
import sys

from pyTENAX.intense import *
import glob
import datetime as dt


S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        tolerance=0.1,
    )

start_time = time.time()
files = glob.glob('Germany/*') #change Germany to folder name if looking through a different folder
name_col = "ppt"

data_meta = readIntense(files[0], only_metadata=True, opened=False)

perc = data_meta.percent_missing_data
yrs_no = data_meta.number_of_records/24/365
G = np.genfromtxt(files[0],skip_header=21)


#extract start and end dates from metadata
start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
end_date_G= dt.datetime.strptime(data_meta.end_datetime, "%Y%m%d%H")

time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
# replace -999 with nan
G[G == -999] = np.nan

df = pd.DataFrame({'prec_time':time_list_G,name_col:G}).set_index('prec_time')

data_clean = S.remove_incomplete_years(df, name_col) #remove incomplete years (below tolerance)
full_yrs_no = np.size(data_clean)/24/365

print('time for one file',str(time.time()-start_time))

info = np.zeros([4,np.size(files)]) # filename, perc, years, cleaned_yrs
start_time = time.time()

i=0
while i<np.size(files):
    
    data_meta = readIntense(files[i], only_metadata=True, opened=False)
    info[0,i] = files[i][11:16] #filename
    info[1,i] = data_meta.percent_missing_data #percent missing
    info[2,i] = data_meta.number_of_records/24/365 #years uncleaned
    
    
    if data_meta.number_of_records>365:
        G = np.genfromtxt(files[i],skip_header=21)
                   
        #extract start and end dates from metadata
        start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
        end_date_G= dt.datetime.strptime(data_meta.end_datetime, "%Y%m%d%H")
    
        time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
        # replace -999 with nan
        G[G == -999] = np.nan
        
        df = pd.DataFrame({'prec_time':time_list_G,name_col:G}).set_index('prec_time')
    
        data_clean = S.remove_incomplete_years(df, name_col) #remove incomplete years (below tolerance)
        info[3,i] = np.size(np.unique(data_clean.index.year))
    else:
        print('tiny!')
        print(i)
        print(str(time.time()-start_time))

    i=i+1
    
    
    
print('time to loop thorugh germany',str(time.time()-start_time))
np.save('germany_data.npy',info)


#filename = DE_+info[0,:]+.txt
    
    
    
    
    
    
    
    