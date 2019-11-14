"""
main.py

The high level script that runs the system optimization and reports results
This is a wrapper around the workflow of:
- Step 0: Defining technologies to evaluate and their constraints
- Step 1: Resource allocation: getting the optimal mix and size of technologies
- Step 2: Optimal design: using Python to optimize the hybrid system design, evaluating performance and financial metrics with SAM
- Step 3: Evaluation model: Performing uncertainty quantification
"""

import time
import numpy as np
import pandas as pd
import os.path
import glob
import scipy.stats
from os import listdir

from shapely.geometry import Point, Polygon
import geopandas as gpd
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata

import sys

sys.path.append('../')

import multiprocessing
#from hybrid.resource import Resource, SolarResource, WindResource

# directory to resource_files
main_dir = '/Users/abarker/Desktop/Hybrid Model/Code/v2/hybrid_analysis/hybrid_systems/'
#main_dir = '/Users/abarker/Desktop/HybridPowerPlants/code/'
solar_dir = main_dir + 'resource_files_all/solar/'
wind_dir = main_dir + 'resource_files_all/wind/'

N_lat = 16  # number of data points
N_lon = 30
lat = np.linspace(24.7433195, 49.3457868, N_lat)
lon = np.linspace(-124.7844079, -66.9513812, N_lon)
year = 2012

# get the wind data
HH = 80
wind_filenames = []
solar_filenames = []
count = 0

# commented out code for downloading from Wind toolkit and NSRDB - i have a broken api at the moment...
# for i in range(N_lat):
#     for j in range(N_lon):
#         t1 = time.time()
#         print('Latitude = ', lat[i], 'Longitude = ', lon[j])
#         wind_resource = WindResource(lat[i], lon[j], year, HH, download=True)
#         solar_resource = SolarResource(lat[i], lon[j], year, download=True)
#
#         t2 = time.time()
#         wind_filenames.append(wind_resource.filename)
#         solar_filenames.append(solar_resource.filename)
#         print(i,j,'Total time = ', t2-t1)
#         print('==========================================================')
#
#         x_lon[count] = lon[j]
#         y_lat[count] = lat[i]
#         count = count + 1
files_solar = []
files_wind = []
for file in os.listdir(solar_dir):
    if file.endswith(".csv"):
        files_solar.append(file)

for file in os.listdir(wind_dir):
    if file.endswith(".srw"):
        files_wind.append(file)


print('Getting solar data...')
x_lon_solar = np.zeros(len(files_solar) - 3)
y_lat_solar = np.zeros(len(files_solar) - 3)

# get size of the files
strFile = solar_dir + files_solar[3]
df = pd.read_csv(strFile, nrows=1)
df_solar_ghi = np.array(pd.read_csv(strFile, skiprows=2)['GHI'])
solar_ghi = np.zeros((len(files_solar) - 3, len(df_solar_ghi)))

df_solar_dhi = np.array(pd.read_csv(strFile, skiprows=2)['DHI'])
solar_dhi = np.zeros((len(files_solar) - 3, len(df_solar_dhi)))

df_solar_dni = np.array(pd.read_csv(strFile, skiprows=2)['DNI'])
solar_dni = np.zeros((len(files_solar) - 3, len(df_solar_dni)))

df_solar_ZA = np.array(pd.read_csv(strFile, skiprows=2)['Solar Zenith Angle'])
solar_ZA = np.zeros((len(files_solar) - 3, len(df_solar_ZA)))

# strFile = '../resource_files/wind/' + files_wind[3]
# df = pd.read_csv(strFile)
# print(df.columns[5:7])
# xxx

# solar
mean_solar = np.zeros(len(files_solar)-3)
for i in range(3, len(files_solar)):
    strFile = solar_dir + files_solar[i]
    df = pd.read_csv(strFile, nrows=1)
    x_lon_solar[i - 3] = df['Longitude']
    y_lat_solar[i - 3] = df['Latitude']
    solar_ghi[i - 3, :] = np.array(pd.read_csv(strFile, skiprows=2)['GHI'])
    solar_dhi[i - 3, :] = np.array(pd.read_csv(strFile, skiprows=2)['DHI'])
    solar_dni[i - 3, :] = np.array(pd.read_csv(strFile, skiprows=2)['DNI'])
    solar_ZA[i - 3, :] = np.array(pd.read_csv(strFile, skiprows=2)['Solar Zenith Angle'])
    mean_solar[i-3] = np.mean(solar_ghi[i-3,:])

# print('Getting wind data...')
x_lon_wind = np.zeros(len(files_wind)-3)
y_lat_wind = np.zeros(len(files_wind)-3)

print('Getting wind data...')
# get size of the files
strFile = wind_dir + files_wind[3]
df = pd.read_csv(strFile,nrows=1)
df_wind = np.array(pd.read_csv(strFile,skiprows=2)['Speed'])[2:]
wind = np.zeros((len(files_wind)-3,len(df_wind)))
mean_wind = np.zeros(len(files_wind)-3)

# wind
count = 0
for i in range(3,len(files_wind)):
    strFile = wind_dir + files_wind[i]
    df = pd.read_csv(strFile, nrows=1)
    if float(df.columns[6]) in x_lon_wind and float(df.columns[5]) in y_lat_wind:
        print('Duplicate coordinate: ', df.columns[5], df.columns[6])
    else:
        # print(count, 'out of ', len(files_wind)-3)
        x_lon_wind[count] = float(df.columns[6])
        y_lat_wind[count] = float(df.columns[5])
        wind[count,:] = np.array(pd.read_csv(strFile, skiprows=2)['Speed'])[2:]
        mean_wind[count] = np.mean(wind[count,:])
        count = count + 1

# print(np.array(x_lon_wind))
x_lon_wind = x_lon_wind[0:count-1]
y_lat_wind = y_lat_wind[0:count-1]
wind = wind[0:count-1,:]
mean_wind = mean_wind[0:count-1]
# xxx
# print(len(x_lon_wind),len(y_lat_wind),len(wind))
# # print(len(np.unique(x_lon_wind + y_lat_wind) ))
# xxx

# interpolate the values onto a new grid
from scipy import interpolate
print('Converting solar and wind data...')
solar_fine_ghi = np.zeros((N_lat*N_lon,8760))
solar_fine_dhi = np.zeros((N_lat*N_lon,8760))
solar_fine_dni = np.zeros((N_lat*N_lon,8760))
solar_fine_ZA = np.zeros((N_lat*N_lon,8760))
wind_fine = np.zeros((N_lat*N_lon,8760))
x_lon_fine = np.zeros(N_lat*N_lon)
y_lat_fine = np.zeros(N_lat*N_lon)

# t = np.linspace(1,8760,8760)
count = 0
for i in range(N_lon):
    for j in range(N_lat):
        x_lon_fine[count] = lon[i]
        y_lat_fine[count] = lat[j]
        count = count + 1
#
for k in range(8760):
    #print('Evaluating resource at ', k, 'out of ',8760)

    # solar interpolation
    #f = interpolate.Rbf(x_lon_solar, y_lat_solar, solar[:,k], kind='linear')
    #solar_fine[:,k] = f(x_lon_fine,y_lat_fine)
    solar_fine_ghi[:, k] = interpolate.griddata((x_lon_solar, y_lat_solar), solar_ghi[:, k], (x_lon_fine, y_lat_fine),
                                           method='nearest')
    solar_fine_dhi[:, k] = interpolate.griddata((x_lon_solar, y_lat_solar), solar_dhi[:, k], (x_lon_fine, y_lat_fine),
                                           method='nearest')
    solar_fine_dni[:, k] = interpolate.griddata((x_lon_solar, y_lat_solar), solar_dni[:, k], (x_lon_fine, y_lat_fine),
                                           method='nearest')
    solar_fine_ZA[:, k] = interpolate.griddata((x_lon_solar, y_lat_solar), solar_ZA[:, k], (x_lon_fine, y_lat_fine),
                                           method='nearest')


    # wind interpolation
    #f = interpolate.Rbf(x_lon_wind, y_lat_wind, wind[:, k], kind='linear')
    #wind_fine[:, k] = f(x_lon_fine, y_lat_fine)
    wind_fine[:,k] = interpolate.griddata((x_lon_wind,y_lat_wind),wind[:,k],(x_lon_fine,y_lat_fine),method='nearest')

# Aaron's code here
# solar_fine has the solar data for each location defined on lines 42-43 (lat/long)
# wind_fine has the wind data for each location
# basically all this mess was to get them on the same grid
# wind_fine[i,j] : i = the x_lon_fine, y_lat_fine point, and j = time index out of 8760 (so this is only hourly data)
# I think what you can do is loop through solar_fine and wind_fine and put this in your code where we put row[2] = 20.0, you can put wind_fine[i,:] instead
# I hadn't done this for solar yet, just wind in your version of the code... let me know if you want me to tackle that.


print('Saving solar data...')
np.save('fine_solar_ghi', solar_fine_ghi)
np.save('fine_solar_dhi', solar_fine_dhi)
np.save('fine_solar_dni', solar_fine_dni)
np.save('fine_solar_za', solar_fine_ZA)
np.save('fine_wind', wind_fine)

# solar_fine = np.load('fine_solar.npy')
# wind_fine = np.load('fine_wind.npy')


print('Get quantities on solar/wind...')
coeff = np.zeros(N_lat*N_lon)
mag_solar = np.zeros(N_lat*N_lon)
mag_wind = np.zeros(N_lat*N_lon)
for i in range(N_lat * N_lon):
    coeff[i] = scipy.stats.pearsonr(solar_fine_ghi[i,:], wind_fine[i,0:8760])[0]
    mag_wind[i] = np.mean(wind_fine[i,0:8760])
    mag_solar[i] = np.mean(solar_fine_ghi[i,:])



# #plot the points on the U.S. map in 2D
# print('Mapping data...')
#
# # define the Graph object
# usa = gpd.read_file('USMap/states_21basic/states.shp')
# # can plot
#
# # find the points inside a state
# levels = np.linspace(-0.5,0.5,20)
# fig, ax = plt.subplots(figsize=(10, 7))
# plt.tricontourf(x_lon_fine, y_lat_fine, coeff, cmap='gray',levels=levels)
# plt.colorbar()
# # plot the lower 48
# usa[1:50].plot(ax=ax, edgecolor='k', alpha=0.1)
# states = usa.STATE_NAME[1:50]
# plt.title('Correlation', fontsize=15)
# # plt.scatter(x_lon,y_lat,c=coeff)
# # plt.show()
#
# # find the points inside a state
# fig, ax = plt.subplots(figsize=(10, 7))
# plt.tricontourf(x_lon_fine, y_lat_fine, mag_wind, cmap='Greys')
# plt.colorbar()
# # plot the lower 48
# usa[1:50].plot(ax=ax, edgecolor='k', alpha=0.1)
# states = usa.STATE_NAME[1:50]
# plt.title('Wind - New Grid', fontsize=15)
# # plt.scatter(x_lon,y_lat,c=coeff)
# # plt.show()
#
# # find the points inside a state
# fig, ax = plt.subplots(figsize=(10, 7))
# plt.tricontourf(x_lon_wind, y_lat_wind, mean_wind, cmap='Greys')
# plt.colorbar()
# # plot the lower 48
# usa[1:50].plot(ax=ax, edgecolor='k', alpha=0.1)
# states = usa.STATE_NAME[1:50]
# plt.title('Wind - Old Grid', fontsize=15)
# # plt.scatter(x_lon,y_lat,c=coeff)
# # plt.show()
#
# # find the points inside a state
# fig, ax = plt.subplots(figsize=(10, 7))
# levels = np.linspace(0.4,1.0,20)
# tot_resource = mag_wind/np.max(mag_wind) + mag_solar/np.max(mag_solar)
# tot_resource = tot_resource/np.max(tot_resource)
# plt.tricontourf(x_lon_fine, y_lat_fine, tot_resource, cmap='Greys',levels=levels)
# plt.colorbar()
# # plot the lower 48
# usa[1:50].plot(ax=ax, edgecolor='k', alpha=0.1)
# states = usa.STATE_NAME[1:50]
# plt.title('Best Resource', fontsize=15)
# # plt.scatter(x_lon,y_lat,c=coeff)
# # plt.show()
#
# # find the points inside a state
# fig, ax = plt.subplots(figsize=(10, 7))
# plt.tricontourf(x_lon_fine, y_lat_fine, mag_solar, cmap='Greys')
# plt.colorbar()
# # plot the lower 48
# usa[1:50].plot(ax=ax, edgecolor='k', alpha=0.1)
# states = usa.STATE_NAME[1:50]
# plt.title('Solar - New Grid', fontsize=15)
# # plt.scatter(x_lon,y_lat,c=coeff)
#
# fig, ax = plt.subplots(figsize=(10, 7))
# plt.tricontourf(x_lon_solar, y_lat_solar, mean_solar, cmap='Greys')
# plt.colorbar()
# # plot the lower 48
# usa[1:50].plot(ax=ax, edgecolor='k', alpha=0.1)
# states = usa.STATE_NAME[1:50]
# plt.title('Solar - Old Grid', fontsize=15)
# # plt.scatter(x_lon,y_lat,c=coeff)
# plt.show()







