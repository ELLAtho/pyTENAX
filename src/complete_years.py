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



S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
    )


files = glob.glob('Germany/*') #change Germany to folder name if looking through a different folder


data_meta = readIntense(files[0], only_metadata=True, opened=False)

perc = data_meta.percent_missing_data
yrs_no = data_meta.number_of_records/24/365

# data=pd.read_parquet(file_path_input)
# # Convert 'prec_time' column to datetime, if it's not already
# data['prec_time'] = pd.to_datetime(data['prec_time'])
# # Set 'prec_time' as the index
# data.set_index('prec_time', inplace=True)
# name_col = "prec_values" #name of column containing data to extract




data_clean = S.remove_incomplete_years(data, name_col)
