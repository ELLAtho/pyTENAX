# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:56:20 2024

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



drive='F' #name of drive
countries = ['Belgium','Germany','Japan','US']
name_lens = [8,5,5,6]


info = []


for n in np.arange(0,4):
    dd = pd.read_csv(drive+':/metadata/'+countries[n]+'_fulldata.csv')
    dd.station = dd['station'].apply(lambda x: f'{int(x):0{name_lens[n]}}')
    dd.station = dd.station.astype(str)
    dd.startdate = pd.to_datetime(dd.startdate)
    dd.enddate = pd.to_datetime(dd.enddate)
    dd.to_csv(drive+':/metadata/'+countries[n]+'_fulldata.csv',index=False)
    info.append(dd)
    
#for Finland, Ireland, Norway,Portugal number = number


