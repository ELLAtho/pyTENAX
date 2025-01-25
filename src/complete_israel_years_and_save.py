# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:15:47 2025

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

info = pd.read_csv(drive+':/metadata/'+country+'_fulldata.csv', dtype={'station': str})
info.startdate = pd.to_datetime(info.startdate)
info.enddate = pd.to_datetime(info.enddate)



files = glob.glob('D:/'+country+'/60_min/*') #list of filenames in folder
folder_size = np.size(files)
clean_count = [0]*np.size(files)

start_time = time.time()

i=0 
while i<np.size(files): 
    G = pd.read_csv(files[i])
    
    G.prec_time = pd.to_datetime(G.prec_time)
    
    G = G.set_index('prec_time')
    

    data_clean = S.remove_incomplete_years(G, name_col) #remove incomplete years (below tolerance)
    clean_count[i] = np.size(np.unique(data_clean.index.year))
    
    
        
    if (i-1) % 100 == 0: #i-1 to skip 0 
        print(i)
        print('Estimated time to complete: '+str((folder_size-i)*(time.time()-start_time)/(i*60))+' mins')
    i=i+1

info['cleaned_years'] = clean_count

info.to_csv(drive+':/metadata/'+country+'_fulldata.csv',index = False)