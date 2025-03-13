# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:09:45 2025

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pyTENAX.intense import *
import glob
import datetime as dt
import matplotlib.pyplot as plt


#['total_years', 'cleaned_years', 'latitude', 'longitude', 'startdate',
#       'enddate', 'cleaned_years_70', 'kgb_zone', 'kgb_group', 'station']



S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        tolerance=0.1,
    )

country = "Brazil"
name_col = "ppt"


meta_auto = pd.read_csv("D:/Brazil/auto_meta.csv")
meta_pluvio = pd.read_csv("D:/Brazil/pluvio_meta.csv")

meta_auto.start_date = pd.to_datetime(meta_auto.start_date)
meta_auto.end_date = pd.to_datetime(meta_auto.end_date)

meta_pluvio.start_date = pd.to_datetime(meta_pluvio.start_date)
meta_pluvio.end_date = pd.to_datetime(meta_pluvio.end_date)

meta_auto["station_type"] = ["auto"]*len(meta_auto)
meta_pluvio["station_type"] = ["pluvio"]*len(meta_pluvio)

meta_auto["total_years"] = (meta_auto.end_date - meta_auto.start_date)/pd.Timedelta('365 days')
meta_pluvio["total_years"] = (meta_pluvio.end_date - meta_pluvio.start_date)/pd.Timedelta('365 days')





meta = pd.concat([meta_auto,meta_pluvio])
meta.station.iloc[572]= "Mendanha"
meta.station.iloc[574]= "Saude"
meta.station.iloc[577]= "riocentro"

files = glob.glob("D:/Brazil/data/*")
filenames = [0]*len(meta)
cleaned_years = [0]*len(meta)
start_time = [0]*len(meta)

for i in np.arange(0,len(meta)):
    start_time[i] = time.time()
    
    filenames[i] = f"D:/Brazil/data\\{meta.station.iloc[i]}.csv"
    G = pd.read_csv(filenames[i],names = ['prec_time',name_col])
    G = G.drop(0,axis=0)
    G.prec_time = pd.to_datetime(G.prec_time)
    G = G.set_index('prec_time')
    
    data_clean = S.remove_incomplete_years(G, name_col) #remove incomplete years (below tolerance)
    cleaned_years[i] = np.size(np.unique(data_clean.index.year))
    
    
    
    
    if i%25 == 0:    
        time_taken = (time.time()-start_time[i-9])/10
        time_left = (len(meta)-i)*time_taken/60
        print(f"{i}/{len(meta)}. Approx time left: {time_left:.0f} mins") #this is only correct after 50 loops
    else:
        pass

    
meta["cleaned_years"] = cleaned_years
meta = meta.drop('Unnamed: 0',axis=1)


newlist = []
for file in files:
    if file.lower() not in filenames_lower:
        newlist.append(file)
    else:
        pass
    
newnames = [name[15:-4] for name in newlist]


meta2 = pd.read_csv('D:/Brazil/data\\meta.csv')
meta2 = meta2.drop('Unnamed: 0',axis=1)

meta2.station = meta2.station.lower()
meta2.station.iloc[22] = "iraja"
meta2.station.iloc[6] = "mendanha"
meta2.station.iloc[15] = "grajau"
meta2.station.iloc[18] = "tanque"
meta2.station.iloc[19] = "barrinha"
meta2.station.iloc[20] = "recreio"

meta2.start_date = pd.to_datetime(meta2.start_date)
meta2.end_date = pd.to_datetime(meta2.end_date)

meta2_cut = meta2[meta2["station"].isin(newnames)]
meta2_cut = meta2_cut.drop(17,axis=0)


filenames2 = [0]*len(meta2_cut)
cleaned_years2 = [0]*len(meta2_cut)



for i in np.arange(0,len(meta2_cut)):
    start_time[i] = time.time()
    
    filenames2[i] = f"D:/Brazil/data\\{meta2_cut.station.iloc[i]}.csv"
    G = pd.read_csv(filenames2[i],names = ['prec_time',name_col])
    G = G.drop(0,axis=0)
    G.prec_time = pd.to_datetime(G.prec_time)
    G = G.set_index('prec_time')
    
    data_clean = S.remove_incomplete_years(G, name_col) #remove incomplete years (below tolerance)
    cleaned_years2[i] = np.size(np.unique(data_clean.index.year))


meta2_cut["total_years"] = (meta2_cut.end_date - meta2_cut.start_date)/pd.Timedelta('365 days')
meta2_cut["cleaned_years"] = cleaned_years2

metafull = pd.concat([meta,meta2_cut])
newlat = [0] * len(metafull)
newlon = [0] * len(metafull)


for i in np.arange(0,len(metafull)):
    if type(metafull.latitude.iloc[i]) == str:
        newlat[i] = float(metafull.latitude.iloc[i].replace(",", "."))
    else:
        newlat[i] = metafull.latitude.iloc[i]
        
    if type(metafull.longitude.iloc[i]) == str:
        newlon[i] = float(metafull.longitude.iloc[i].replace(",", "."))
    else:
        newlon[i] = metafull.longitude.iloc[i]


metafull.latitude = newlat
metafull.longitude = newlon

metafull.to_csv("D:/metadata/Brazil_fulldata.csv",index = False)

    
plt.hist(metafull.cleaned_years)
plt.title('Number of complete years '+country)
plt.show()

yrs_above_10 =  metafull.cleaned_years[metafull.cleaned_years>10]
yrs_above_20 =  metafull.cleaned_years[metafull.cleaned_years>20]

print('files longer than 20 years: '+str(np.size(yrs_above_20)))
print('files longer than 10 years: '+str(np.size(yrs_above_10)))
print('total files: '+str(np.size(metafull.cleaned_years)))



fig = plt.figure()

proj = ccrs.PlateCarree()
ax1 = fig.add_subplot(1, 1, 1, projection=proj)

# Add map features
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')

sc = plt.scatter(metafull.longitude,metafull.latitude,c = metafull.cleaned_years,cmap = "PuBuGn",s = 3)

plt.colorbar(sc)
plt.show()


