# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:07:28 2024

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
import glob
import time


from pyTENAX.intense import *
from pyTENAX.pyTENAX import *
from pyTENAX.globalTENAX import *

drive = 'D'
country = 'US' #folder name that contains country data
country_code = 'US_'
name_col = "ppt"
year_size_limit = 3 # won't bother getting clean years if file smaller than this times percent missing


#done: germany, belgium, finland, Ireland, Japan, Norway, Portugal, UK


S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        tolerance=0.3,
    )



start_time = time.time()

metadata = pd.read_csv(f'{drive}:/metadata/{country}_fulldata.csv', dtype={'station': str})
metadata['cleaned_years_70'] = np.zeros(len(metadata))
data_clean = np.zeros(len(metadata))


for i in np.arange(0,len(metadata)):
    code = metadata.station[i]
    path = f'{drive}:/{country}/{country_code}{code}.txt'
    if metadata.cleaned_years[i] != 0:
        G,data_meta = read_GSDR_file(path,name_col)
        
        yrs_gone = S.remove_incomplete_years(G, name_col) #remove incomplete years (below tolerance)
        
        data_clean[i] = np.size(np.unique(yrs_gone.index.year))
    else:
        data_clean[i] = 0
        
    if (i-1) % 100 == 0: #i-1 to skip 0 
        print(i)
        print(f'Estimated time to complete: {(len(metadata)-i)*(time.time()-start_time)/(i*60):.0f} mins')
        
        
metadata['cleaned_years_70'] = data_clean

metadata.to_csv(f'{drive}:/metadata/{country}_fulldata.csv',index = False)



#FOR ISD
country = 'ISD' #folder name that contains country data
name_col = "ppt"
year_size_limit = 3 # won't bother getting clean years if file smaller than this times percent missing


metadata = pd.read_csv(f'{drive}:/metadata/{country}_fulldata.csv', dtype={'station': str})
metadata['cleaned_years_70'] = np.zeros(len(metadata))
data_clean = np.zeros(len(metadata))



files = glob.glob('D:/ISD/*') #list of filenames in folder
for i in np.arange(0,len(files)):
    if metadata.cleaned_years[i] != 0:
        G,data_meta = read_GSDR_file(files[i],name_col)
        print(files[i])
        print(metadata.station)
        yrs_gone = S.remove_incomplete_years(G, name_col) #remove incomplete years (below tolerance)
        
        data_clean[i] = np.size(np.unique(yrs_gone.index.year))
    else:
        data_clean[i] = 0
    

metadata['cleaned_years_70'] = data_clean

#metadata.to_csv(f'{drive}:/metadata/{country}_fulldata.csv',index = False)
