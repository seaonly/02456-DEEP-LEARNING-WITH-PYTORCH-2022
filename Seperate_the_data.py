# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:15:04 2022

@author: nafan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:27:55 2022

@author: nafan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:06:05 2022

@author: nafan
"""

import numpy as np 
import xarray as xr
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import SMOTE

tr = range(2007,2014)
vl = range(2014,2018)
te = range(2018,2021)
month =range(5,10)
# feature data

dir1 = 'N:/hpc/Data/'
dir2 = 'H:/DeepLear/Data/'
dir3 = 'H:/Explortory/Data/'

def predict1(dirk,data,var1,var2,generate_tensor=  False):
    file_path = dirk + data +'.nc'
    if generate_tensor == True:
        ds2 = xr.open_dataset(file_path ,drop_variables="geopotential", engine='netcdf4')
    else:
        ds2 = xr.open_dataset(file_path , engine='netcdf4')
    ds2  = ds2.sel(time=ds2.time.dt.month.isin([month]))   
    # get actual timesteps   
    actual_days = ds2.time.values 
    
    # get Month-Day of each timestep
    dates_grouped = pd.to_datetime(ds2.time.values).strftime('%m%d')      
    
    # 5-day smoothed climatology. Rolling can be applied directly because the daily data refer to consequtive days. If
    # days are not consecutive, firstly the xr.resample should be applied, so that missing days are generated with NaN
    Smoothed = ds2.rolling(time=5, center=True, min_periods=1).mean() # 5-day smoothing
    
    # change the time to Month-Day
    ds2 = ds2.assign_coords({'time': dates_grouped}) 
    
    # change the time to Month-Day
    Smoothed = Smoothed.assign_coords({'time': dates_grouped}) 
      
    # climatology of the smoothed data 
    Climatology = Smoothed.groupby('time').mean() 
    #If we do not want 5 day moving window
    #Climatology = Daily.groupby('time').mean() 
   
    #sutract the climatology   
    Anomalies = ds2.groupby('time') - Climatology
    
    # change back to the original timestep information
    Anomalies = Anomalies.assign_coords({'time': actual_days}) 
    Anomalies = Anomalies.rename({var1: var2})
    return Anomalies

z500 = predict1(dir3, "Z500", 'daily','Z500', generate_tensor = True)
z1000 = predict1(dir1, "era5_z1000_day_na_1979-2020", 'geopotential','Z1000')
z300 = predict1(dir1, "era5_z300_day_na_1979-2020", 'geopotential','Z300')
t500 = predict1(dir2, "era5_t500_day_na_1979-2020", 'ta','t500')
t850 = predict1(dir2, "era5_t850_day_na_1979-2020", 'ta','t850')

ds7 = xr.merge([z500,z1000,z300,t500,t850])
x_train  = ds7.sel(time=ds7.time.dt.year.isin([tr]))
x_val  = ds7.sel(time=ds7.time.dt.year.isin([vl]))
x_test  = ds7.sel(time=ds7.time.dt.year.isin([te]))
x_train.to_netcdf("N:/hpc/Data/x_train.nc") # save data used for plotting
x_val.to_netcdf("N:/hpc/Data/x_val.nc") # save data used for plotting
x_test.to_netcdf("N:/hpc/Data/x_test.nc") # save data used for plotting


df = pd.read_csv('H:\Explortory\Data\PCS.csv')
df1 = df.assign(year = df["date"].str[:4])
df2 = df1[["extreme","year"]]
df2.loc[df2['extreme']== "Yes", 'ex'] = 1 
df2.loc[df2['extreme']== "No", 'ex'] = 0
df2.shape
"""
df = pd.read_csv('N:\hpc\Data\prec_era5.csv')
df1 = df.assign(year = df["date"].str[:4])
df2 = df1[["P90","year"]]
df2.loc[df2['P90']== "Yes", 'ex'] = 1 
df2.loc[df2['P90']== "No", 'ex'] = 0
df2.shape
"""
years_to_train = ["2007","2008","2009","2010","2011","2012","2013"]
years_to_valid = ["2014","2015","2016","2017"]
years_to_test = ["2018","2019","2020"]


y_train = df2[df2.year.isin(years_to_train)]
y_val = df2[df2.year.isin(years_to_valid)]

y_test = df2[df2.year.isin(years_to_test)]

y_train.to_csv("N:/hpc/Data/y_train.csv") 
y_val.to_csv("N:/hpc/Data/y_val.csv") 
y_test.to_csv("N:/hpc/Data/y_test.csv") 
