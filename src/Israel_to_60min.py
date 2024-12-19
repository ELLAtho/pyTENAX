# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:00:39 2024

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

files10 = glob.glob(f'{drive}:/Israel/10_min/*')
for n in np.arange(0,len(files10)):
    file = files10[n]
    






