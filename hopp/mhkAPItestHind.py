import mhkit
import numpy as np
from datetime import datetime
import csv

# Need to configure NREL's highly scalable data service (HSDS) see: https://github.com/NREL/hsds-examples

lat_lon = [48.494, -124.728] # NDBC 46087

region = mhkit.wave.io.hindcast.hindcast.region_selection(lat_lon)
print(region)

data_type = '3-hour' # can be '1-hour' or '3-hour'
parameter = ['energy_period', 'significant_wave_height']
years = [2005] # 1979 to 2010 are available

waveData = mhkit.wave.io.hindcast.hindcast.request_wpto_point_data(data_type, parameter, lat_lon, years, tree=None, unscale=True, str_decode=True, hsds=True)
print(waveData)

TeAndHs = waveData[0]
Te = TeAndHs['energy_period_0']
print(Te[0])
Hs = TeAndHs['significant_wave_height_0']
print(Hs[0])

# Access time stamps
datesTS = Te.index.tolist()
datesSTRs = []
for i in range(len(datesTS)):
    datesSTRs.append(str(datetime.strptime(str(datesTS[i]), '%Y-%m-%d %H:%M:%S%z')))
# print(datesSTRs)

yr = np.empty(len(Te))
for i in range(len(yr)):
    yr[i] = float(datesSTRs[i][:4])
print(yr)

month = np.empty(len(Te))
for i in range(len(month)):
    month[i] = float(datesSTRs[i][5:7])
print(month)

day = np.empty(len(Te))
for i in range(len(day)):
    day[i] = float(datesSTRs[i][8:10])
print(day)

hour = np.empty(len(Te))
for i in range(len(hour)):
    hour[i] = float(datesSTRs[i][11:13])
print(hour)

minute = np.empty(len(Te))
for i in range(len(minute)):
    minute[i] = float(datesSTRs[i][14:16])
print(minute)

fields = ["Year", "Month", "Day", "Hour", "Minute", "Significant Wave Height","Energy Period"]

resultsMat = np.empty((len(Hs),7))
for i in range(len(Hs)):
    resultsMat[i,0] = yr[i]

for i in range(len(Hs)):
    resultsMat[i,1] = month[i]

for i in range(len(Hs)):
    resultsMat[i,2] = day[i]

for i in range(len(Hs)):
    resultsMat[i,3] = hour[i]

for i in range(len(Hs)):
    resultsMat[i,4] = minute[i]

for i in range(len(Hs)):
    resultsMat[i,5] = Hs[i]

for j in range(len(Hs)):
    resultsMat[j,6] = Te[j]
print(resultsMat[0,:])

# Access additional location details
locData = waveData[1]
print(locData)
depth = locData['water_depth'][0]
latit = locData['latitude'][0]
longit = locData['longitude'][0]
dist = locData['distance_to_shore'][0]
tz = locData['timezone'][0]
jur = locData['jurisdiction'][0]
print(locData['timezone'][0])
print(tz)

mainHeader = ["Source", "Station ID", "Jurisdiction", "Latitude", "Longitude", "Time Zone", "Local Time", "Distance to Shore", "Directionality Coefficient", "Energy Period", "Maximum Energy Direction", "Mean Absolute Period", "Mean Wave Direction", "Mean Zero-Crossing Period", "Omni-Directional Wave Power", "Peak Period", "Significant Wave Height", "Spectral Width", "Water Depth", "Version"]
subHeader = ["WPTO Hindcast Data", float(0), jur, latit, longit, float(0), tz, dist, "-", "s", "deg", "s", "deg", "s", "W/m", "s", "m", "-", depth, "v1.0.0"]
with open('APIhindCallTest.csv', 'w', newline='') as file: 
    writer = csv.writer(file)

    writer.writerow(mainHeader)
    writer.writerow(subHeader)
    writer.writerow(fields)
    writer.writerows(resultsMat) 