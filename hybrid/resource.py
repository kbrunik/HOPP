import csv
import os
import requests
import time

from keys import developer_nrel_gov_key


class Resource:
    """
    Class to manage resource_files data
    """
    def __init__(self, lat, lon, year, **kwargs):
        """
        Parameters
        ---------
        lat: float
            The latitude
        lon: float
            The longitude
        year: int
            The year of resource_files data
        """

        self.latitude = lat
        self.longitude = lon
        self.year = year
        self.api_key = developer_nrel_gov_key

        # generic api settings
        self.interval = '60'
        self.leap_year = 'false'
        self.utc = 'false'
        self.name = 'hybrid-systems'
        self.affiliation = 'NREL'
        self.reason = 'hybrid-analysis'
        self.email = 'nicholas.diorio@nrel.gov'
        self.mailing_list = 'true'

        # paths
        self.path_current = os.path.dirname(os.path.abspath(__file__))
        self.path_resource = os.path.join(self.path_current, '..', 'resource_files')

        # update any passed in
        self.__dict__.update(kwargs)

    @staticmethod
    def get_data(url, filename):
        """
        Parameters
        ---------
        url: string
            The API endpoint to return data from
        filename: string
            The filename where data should be written
        """
        n_tries = 0
        success = False
        while n_tries < 5:

            try:
                r = requests.get(url)
                if r:
                    localfile = open(filename, mode='w+')
                    localfile.write(r.text)
                    localfile.close()
                    if os.path.isfile(filename):
                        success = True
                        break
                elif r.status_code == 403:
                    print(r.url)
                    print(r.text)
                    raise requests.exceptions.HTTPError
                elif r.status_code == 404:
                    raise requests.exceptions.HTTPError
            except requests.exceptions.Timeout:
                time.sleep(0.2)
                n_tries += 1

        return success


class SolarResource(Resource):
    """
        Class to manage Solar Resource data
        """

    def __init__(self, lat, lon, year, download=False, force_download=False, **kwargs):
        """
        Parameters
        ---------
        lat: float
            The latitude
        lon: float
            The longitude
        year: int
            The year of resource_files data
        download: bool
            Download immediately
        force_download:
            if download set to true, force download even if file exists
        """
        super().__init__(lat, lon, year)

        self.solar_attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'

        self.path_resource = os.path.join(self.path_resource, 'solar')

        # Force override any internal definitions if passed in
        self.__dict__.update(kwargs)

        # resource_files files
        self.filename = os.path.join( self.path_resource, str(lat) + "_" + str(lon) +
                                      "_psmv3_" + str(self.interval) + "_" + str(year) +".csv")

        if download:
            self.download_solar_resource(force_download)

    def download_solar_resource(self, force_download=False):
        """
        Parameters
        ---------
        force_download: bool
            Overwrite downloaded resource_files data if exists
        """
        success = os.path.isfile(self.filename)
        if not success or force_download:
            url = 'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}+{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(
                year=self.year, lat=self.latitude, lon=self.longitude, leap=self.leap_year, interval=self.interval,
                utc=self.utc, name=self.name, email=self.email,
                mailing_list=self.mailing_list, affiliation=self.affiliation, reason=self.reason, api=self.api_key,
                attr=self.solar_attributes)

            success = self.get_data(url, filename=self.filename)

        return success


class WindResource(Resource):
    """ Class to manage Wind Resource data

    Attributes:
        hub_height_meters - the system height
        file_resource_heights - dictionary of heights and filenames to download from Wind Toolkit
        filename - the combined resource filename
    """

    allowed_hub_height_meters = [10, 40, 60, 80, 100, 120, 140, 160, 200]

    def __init__(self, lat, lon, year, wind_turbine_hub_ht, download=False, force_download=False, **kwargs):
        """
        Parameters
        ---------
        lat: float
            The latitude
        lon: float
            The longitude
        year: int
            The year of resource_files data
        wind_turbine_hub_ht: int
            The height of turbines
        download: bool
            Download immediately
        force_download:
            if download set to true, force download even if file exists
        """
        super().__init__(lat, lon, year)

        self.path_resource = os.path.join(self.path_resource, 'wind')

        self.__dict__.update(kwargs)

        self.hub_height_meters = wind_turbine_hub_ht
        self.file_resource_heights = None
        self.filename = None

        self.calculate_heights_to_download()

        if download:
            self.download_wind_resource(force_download)

    def calculate_heights_to_download(self):
        """
        Given the system hub height, and the available hubheights from WindToolkit,
        determine which heights to download to bracket the hub height
        """
        hub_height_meters = self.hub_height_meters

        # evaluate hub height, determine what heights to download
        heights = [hub_height_meters]
        if hub_height_meters not in self.allowed_hub_height_meters:
            height_low = self.allowed_hub_height_meters[0]
            height_high = self.allowed_hub_height_meters[-1]
            for h in self.allowed_hub_height_meters:
                if h < hub_height_meters:
                    height_low = h
                elif h > hub_height_meters:
                    height_high = h
                    break
            heights[0] = height_low
            heights.append(height_high)

        file_resource_base = os.path.join(self.path_resource, str(self.latitude) + "_" + str(self.longitude) + "_windtoolkit_" + str(
            self.year) + "_" + str(self.interval) + "min")
        file_resource_full = file_resource_base
        file_resource_heights = dict()

        for h in heights:
            file_resource_heights[h] = file_resource_base + '_' + str(h) + 'm.srw'
            file_resource_full += "_" + str(h) + 'm'
        file_resource_full += ".srw"

        self.file_resource_heights = file_resource_heights
        self.filename = file_resource_full

    def update_height(self, hub_height_meters):
        self.hub_height_meters = hub_height_meters
        self.calculate_heights_to_download()

    def download_wind_resource(self, force_download=False):
        """
        Parameters
        ---------
        force_download: bool
            Overwrite downloaded resource_files data if exists
        """
        success = os.path.isfile(self.filename)
        if not success or force_download:

            for height, f in self.file_resource_heights.items():
                url = 'http://developer.nrel.gov/api/wind-toolkit/wind/wtk_srw_download?year={year}&lat={lat}&lon={lon}&hubheight={hubheight}&api_key={api_key}'.format(
                    year=self.year, lat=self.latitude, lon=self.longitude, hubheight=height, api_key=self.api_key)

                success = self.get_data(url, filename=f)

            if not success:
                raise ValueError('Unable to download wind data')

        # combine into one file to pass to SAM
        if len(list(self.file_resource_heights.keys())) > 1:
            success = self.combine_wind_files(self.file_resource_heights, self.filename)

            if not success:
                raise ValueError('Could not combine wind resource files successfully')

        return success

    def combine_wind_files(self):
        """
        Parameters
        ---------
        file_resource_heights: dict
            Keys are height in meters, values are corresponding files
            example {40: path_to_file, 60: path_to_file2}
        file_out: string
            File path to write combined srw file
        """
        data = [None] * 2
        for height, f in self.file_resource_heights.items():
            if os.path.isfile(f):
                with open(f) as file_in:
                    csv_reader = csv.reader(file_in, delimiter=',')
                    line = 0
                    for row in csv_reader:
                        if line < 2:
                            data[line] = row
                        else:
                            if line >= len(data):
                                data.append(row)
                            else:
                                data[line] += row
                        line += 1

        with open(self.filename, 'w') as fo:
            writer = csv.writer(fo)
            writer.writerows(data)

        return os.path.isfile(self.filename)
