import mhkit
import numpy as np
import csv
from datetime import datetime
from statistics import mean

# Pulling timeseries data for significant wave height and energy period from various sources

# # WPTO Hindcast Test
# data_type = '1-hour'
# parameter = ['energy_period', 'significant_wave_height']
# lat_lon = ('33.22510', '−119.89175')
# years = [2004]
# wpto_hind_test = mhkit.wave.io.hindcast.hindcast.request_wpto_point_data(data_type, parameter, lat_lon, years, tree=None, unscale=True, str_decode=True, hsds=True)
# print(wpto_hind_test)

# CDiP test 
station = '067'
data_type = 'historic'
year = 2010
paras = ['waveHs','waveTp','waveFrequency','metaWaterDepth','metaStationLatitude','metaStationLongitude']
# paras = 'significant_wave_height'
start = '2010-01-01'
end = '2010-12-31'
# CDiP_test = mhkit.wave.io.cdip.request_netCDF(station_number, data_type)
CDiP_test2 = mhkit.wave.io.cdip.request_parse_workflow(nc=None, station_number=station, parameters=paras, years=year, start_date=start, end_date=end, data_type='historic', all_2D_variables=False)
print(CDiP_test2)
Tp = CDiP_test2['data']['wave']['waveTp'] # Peak period in seconds
# Tp = Tp_flt.tolist()
# print(Tp)

Hs = CDiP_test2['data']['wave']['waveHs'] # Significant wave height in meters
# Hs = Hs_flt.tolist()
# print(len(Hs)/2)
f = CDiP_test2['metadata']['wave']['waveFrequency'] 
# print(f)
latit = CDiP_test2['metadata']['meta']['metaStationLatitude']
# latit = mean(lats)
# print(mean(lats))
longit = CDiP_test2['metadata']['meta']['metaStationLongitude']
# longit = mean(longs)
# print(mean(longs))
depths = CDiP_test2['metadata']['meta']['metaWaterDepth']
depth = mean(depths)
# print(mean(depths))

# Access time stamps
datesTS = Tp.index.tolist()
# datesSTR = datetime.strptime(str(datesTS[0]), '%Y-%m-%d %H:%M:%S')
# print(str(datesSTR))
# datesSTRs = np.empty(len(datesTS))
datesSTRs = []
for i in range(len(datesTS)):
    datesSTRs.append(str(datetime.strptime(str(datesTS[i]), '%Y-%m-%d %H:%M:%S')))
# print(datesSTRs)
# year1 = int(datesSTRs[0][:4])
# print(year1)
year = np.empty(len(Tp))
for i in range(len(year)):
    year[i] = float(datesSTRs[i][:4])
print(year)

month = np.empty(len(Tp))
for i in range(len(month)):
    month[i] = float(datesSTRs[i][5:7])
print(month)

day = np.empty(len(Tp))
for i in range(len(day)):
    day[i] = float(datesSTRs[i][8:10])
print(day)

hour = np.empty(len(Tp))
for i in range(len(hour)):
    hour[i] = float(datesSTRs[i][11:13])
print(hour)

minute = np.empty(len(Tp))
for i in range(len(minute)):
    minute[i] = float(datesSTRs[i][14:16])
print(minute)

second = np.empty(len(Tp))
for i in range(len(second)):
    second[i] = float(datesSTRs[i][17:19])
print(second)

S_pm0 = mhkit.wave.resource.pierson_moskowitz_spectrum(f, Tp[0], Hs[0])
# print(S_pm)

Te0_raw = mhkit.wave.resource.energy_period(S_pm0, frequency_bins=None)
# print(Te0_raw)
Te0 = Te0_raw['Te'].values.tolist()[0]
# Te0 = Te0_raw['Te']
# print(Te0)

# For loop to get rest of the energy period values
Te = np.empty(len(Hs))
for i in range(len(Te)):
    S_pm_i = mhkit.wave.resource.pierson_moskowitz_spectrum(f, Tp[i], Hs[i])
    Tei_raw = mhkit.wave.resource.energy_period(S_pm_i, frequency_bins=None)
    Te[i] = Tei_raw['Te'].values.tolist()[0]

# print(Te)



resultsDict = {}
resultsDict["Hs"] = Hs
resultsDict["Te"] = Te
# print(resultsDict)

fields = ["Year", "Month", "Day", "Hour", "Minute", "Significant Head (m)","Energy Period (s)"]

resultsMat = np.empty((len(Hs),7))
for i in range(len(Hs)):
    resultsMat[i,0] = year[i]

for i in range(len(Hs)):
    resultsMat[i,1] = month[i]

for i in range(len(Hs)):
    resultsMat[i,2] = day[i]

for i in range(len(Hs)):
    resultsMat[i,3] = hour[i]

for i in range(len(Hs)):
    resultsMat[i,4] = minute[i]

# for i in range(len(Hs)):
#     resultsMat[i,5] = second[i]

for i in range(len(Hs)):
    resultsMat[i,5] = Hs[i]

for j in range(len(Hs)):
    resultsMat[j,6] = Te[j]
print(resultsMat[0,:])

# Switch from 30 min data to hourly
resultsMatHrly = np.empty((int(len(Hs)/2),7))
j = 0 # counter for new matrix
for i in range(len(Hs)):
    if i % 2 == 0:
        resultsMatHrly[j,:] = resultsMat[i,:]
        j = j+1

mainHeader = ["Source", "Station ID", "Jurisdiction", "Latitude", "Longitude", "Time Zone", "Local Time", "Distance to Shore", "Directionality Coefficient", "Energy Period", "Maximum Energy Direction", "Mean Absolute Period", "Mean Wave Direction", "Mean Zero-Crossing Period", "Omni-Directional Wave Power", "Peak Period", "Significant Wave Height", "Spectral Width", "Water Depth", "Version"]
subHeader = ["CDIP", str(station), "N/A", float(latit), float(longit), "N/A", "N/A", "N/A", "-", "s", "deg", "s", "deg", "s", "W/m", "s", "m", "-", depth, "N/A"]
with open('APIcallTest.csv', 'w', newline='') as file: 
    writer = csv.writer(file)

    writer.writerow(mainHeader)
    writer.writerow(subHeader)
    writer.writerow(fields)
    writer.writerows(resultsMatHrly) 