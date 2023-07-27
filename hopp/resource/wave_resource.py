import pandas as pd
import PySAM.WaveFileReader as wavefile

from hopp.log import hybrid_logger as logger
from hopp.resource.resource import *
import mhkit
import numpy as np
from datetime import datetime
import csv


class WaveResource(Resource):
    """
    Class to manage Wave Resource data
    """
    def __init__(self, lat, lon, year, path_resource="", filepath="", **kwargs):
        """
        
        :param lat: float
        :param lon: float
        :param year: int
        :param path_resource: directory where to save downloaded files
        :param filepath: file path of resource file to load
        :param kwargs:
        """
        super().__init__(lat, lon, year)

        if os.path.isdir(path_resource):
            self.path_resource = path_resource

        self.wave_attributes = ['energy_period', 'significant_wave_height']

        self.path_resource = os.path.join(self.path_resource, 'wave')

        # Force override any internal definitions if passed in
        self.__dict__.update(kwargs)

        # resource_files files
        if filepath == "":
            self.filename = "" # adapted from wind resource
        else:
            self.filename = filepath

        self.check_download_dir()

        if not os.path.isfile(self.filename):
            self.download_resource()
            # raise ValueError("Wave resource file must be loaded.") # Remove ValueError once resource can be downloaded.
            
        self.format_data()

        # logger.info("WaveResource: {}".format(self.filename))

    def download_resource(self):
        # NOTE THE USER MUST RUN hsconfigure AND FOLLOW INSTRUCTIONS IN https://github.com/NREL/hsds-examples 
        
        data_type = '3-hour' # can be '1-hour' or '3-hour'
        parameter = self.wave_attributes
        lat_lon = [self.latitude, self.longitude]
        years = self.year
        waveData = mhkit.wave.io.hindcast.hindcast.request_wpto_point_data(data_type, parameter, lat_lon, years, tree=None, unscale=True, str_decode=True, hsds=True)
        TeAndHs = waveData[0]
        Te = TeAndHs['energy_period_0']
        Hs = TeAndHs['significant_wave_height_0']
        
        # Access time stamps
        datesTS = Te.index.tolist()
        datesSTRs = []
        for i in range(len(datesTS)):
            datesSTRs.append(str(datetime.strptime(str(datesTS[i]), '%Y-%m-%d %H:%M:%S%z')))
        
        yr = np.empty(len(Te))
        for i in range(len(yr)):
            yr[i] = float(datesSTRs[i][:4])
        
        month = np.empty(len(Te))
        for i in range(len(month)):
            month[i] = float(datesSTRs[i][5:7])
        
        day = np.empty(len(Te))
        for i in range(len(day)):
            day[i] = float(datesSTRs[i][8:10])
        
        hour = np.empty(len(Te))
        for i in range(len(hour)):
            hour[i] = float(datesSTRs[i][11:13])

        minute = np.empty(len(Te))
        for i in range(len(minute)):
            minute[i] = float(datesSTRs[i][14:16])
        
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
        
        locData = waveData[1]
        depth = locData['water_depth'][0]
        latit = locData['latitude'][0]
        longit = locData['longitude'][0]
        dist = locData['distance_to_shore'][0]
        tz = locData['timezone'][0]
        jur = locData['jurisdiction'][0]

        mainHeader = ["Source", "Station ID", "Jurisdiction", "Latitude", "Longitude", "Time Zone", "Local Time", "Distance to Shore", "Directionality Coefficient", "Energy Period", "Maximum Energy Direction", "Mean Absolute Period", "Mean Wave Direction", "Mean Zero-Crossing Period", "Omni-Directional Wave Power", "Peak Period", "Significant Wave Height", "Spectral Width", "Water Depth", "Version"]
        subHeader = ["WPTO Hindcast Data", float(0), jur, latit, longit, float(0), tz, dist, "-", "s", "deg", "s", "deg", "s", "W/m", "s", "m", "-", depth, "v1.0.0"]
        with open(self.filename, 'w', newline='') as file: 
            writer = csv.writer(file)

            writer.writerow(mainHeader)
            writer.writerow(subHeader)
            writer.writerow(fields)
            writer.writerows(resultsMat) 

        return

    def format_data(self):
        """
        Format as 'wave_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename + " does not exist. Try `download_resource` first.")

        self.data = self.filename

    @Resource.data.setter
    def data(self, data_file):
        """
        Sets the wave resource data to a dictionary in the SAM Wave format

        :key significant_wave_height: sequence, wave height time series data [m]
        :key energy period: sequence, wave period time series data [s]
        :key year: sequence
        :key month: sequence
        :key day: sequence
        :key hour: sequence
        :key minute: sequence
        """
        wavefile_model = wavefile.new()
        #Load resource file
        wavefile_model.WeatherReader.wave_resource_filename_ts = str(self.filename)
        wavefile_model.WeatherReader.wave_resource_model_choice = 1 #Time-series=1 JPD=0

        #Read in resource file, output time series arrays to pass to wave performance module
        wavefile_model.execute() 
        hours = wavefile_model.Outputs.hour

        if len(hours) < 8760:
            # Set up dataframe for data manipulation
            df = pd.DataFrame()
            df['year'] = wavefile_model.Outputs.year 
            df['month'] = wavefile_model.Outputs.month
            df['day'] = wavefile_model.Outputs.day
            df['hour'] = wavefile_model.Outputs.hour
            df['minute'] = wavefile_model.Outputs.minute
            df['date_time'] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day, hour=df.hour, minute=df.minute))
            df = df.drop(['year','month','day','hour','minute'], axis=1)
            df = df.set_index(['date_time'])
            df['significant_wave_height'] = wavefile_model.Outputs.significant_wave_height
            df['energy_period'] = wavefile_model.Outputs.energy_period

            # Resample data and linearly interpolate to hourly data
            data_df = df.resample("H").mean()
            data_df = data_df.interpolate(method='linear')

            # If data cannot interpolate last hours
            if len(data_df['energy_period']) < 8760:
                last_hour = data_df.index.max()
                missing_hours = 8760 - len(data_df['energy_period'])

                missing_time = pd.date_range(last_hour + pd.Timedelta(hours=1),periods=missing_hours, freq='H')
                missing_rows = pd.DataFrame(index=missing_time, columns=df.columns)
                data_df = pd.concat([data_df, missing_rows]).sort_index()
                data_df = data_df.fillna(method='ffill') # forward fill


            data_df = data_df.reset_index()
            dic = dict()

            # Extract outputs
            print("data file",data_file)
            dic['significant_wave_height'] = data_df['significant_wave_height']
            dic['energy_period'] = data_df['energy_period']
            dic['year'] = data_df['index'].dt.year
            dic['month'] = data_df['index'].dt.month
            dic['day'] = data_df['index'].dt.day
            dic['hour'] = data_df['index'].dt.hour
            dic['minute'] = data_df['index'].dt.minute

        elif len(hours) == 8760:
            dic['significant_wave_height'] = wavefile_model.Outputs.significant_wave_height
            dic['energy_period'] = wavefile_model.Outputs.energy_period
            dic['year'] = wavefile_model.Outputs.year 
            dic['month'] = wavefile_model.Outputs.month
            dic['day'] = wavefile_model.Outputs.day
            dic['hour'] = wavefile_model.Outputs.hour
            dic['minute'] = wavefile_model.Outputs.minute
        else:
            raise ValueError("Resource time-series cannot be subhourly.")

        self._data = dic