# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:19:29 2024

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

S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        tolerance=0.1,
    )


country = 'germany' #folder name that contains country data
name_col = "ppt"
year_size_limit = 3 # won't bother getting clean years if file smaller than this times percent missing


files = glob.glob('D:/'+country+'/*') #list of filenames in folder
str_start = 7+len(country) # for getting filename number in loop

# initialize info table IT WOULD PROBABLY BE BETTER TO DO THIS AS A DF
info = np.zeros([4,np.size(files)]) # filename, perc, years, cleaned_yrs
times = np.zeros([2,np.size(files)]) # times table to check whats slow.
latslons = np.zeros([2,np.size(files)]) #get location data of stations [lat, lon]

start_time = time.time()


print('folder size: '+str(np.size(files))+' files')
print('first file: '+files[0])

#quick bit
i=0 
while i<np.size(files):
    data_meta = readIntense(files[i], only_metadata=True, opened=False)
    info[0,i] = files[i][str_start:str_start+6] #filename number only, 6 digits
    info[1,i] = data_meta.percent_missing_data #percent missing
    info[2,i] = data_meta.number_of_records/24/365 #years uncleaned
    latslons[0,i] = data_meta.latitude
    latslons[1,i] = data_meta.longitude
    
    if i % 100 == 0:
        print(i)
    i=i+1
    
print('time to do quick: '+str(time.time()-start_time))
np.save('D:/'+country+'_latslons.npy',latslons)
 

# THIS TAKES AGESSSSS (US=7.5HRS)   
start_time = time.time()
  
    
i=0 
while i<np.size(files):  
    data_meta = readIntense(files[i], only_metadata=True, opened=False) 
    if data_meta.number_of_records*(1-data_meta.percent_missing_data/100)>365*24*year_size_limit: #don't bother reading ones less than x years incl missing data
        G = pd.read_csv(files[i],skiprows=21,names=[name_col])
                   
        #extract start and end dates from metadata
        start_date_G= dt.datetime.strptime(data_meta.start_datetime, "%Y%m%d%H")
        end_date_G= dt.datetime.strptime(data_meta.end_datetime, "%Y%m%d%H")
    
        time_list_G= [start_date_G+ dt.timedelta(hours=x) for x in range(0, G.size)] #make timelist of size of FI
        # replace -999 with nan
        G[G == -999] = np.nan
        
        G['prec_time'] = time_list_G
        G = G.set_index('prec_time')
        
    
        data_clean = S.remove_incomplete_years(G, name_col) #remove incomplete years (below tolerance)
        info[3,i] = np.size(np.unique(data_clean.index.year))
        
    else:
        print('less than '+str(year_size_limit)+' year')
        print(i)
        print(str(time.time()-start_time))
        
    if i % 100 == 0:
        print(i)
    i=i+1

  
    
print('time to loop through '+country,str(time.time()-start_time))
np.save('D:/'+country+'_data.npy',info)



plt.hist(info[3])
plt.title('Number of complete years '+country)
plt.show()

# num_above_10 = np.size(info[3][info[3]>10])
#filename = DE_+info[0,:]+.txt
    
    
#info_g = np.load('D:/germany_data.npy')







    