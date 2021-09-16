from typing import Optional, Union, Sequence

import rapidjson                # NOTE: install 'python-rapidjson' NOT 'rapidjson'

import pandas as pd
import numpy as np
import datetime
import os

from hybrid.pySSC_daotk.ssc_wrap import ssc_wrap
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch
from hybrid.power_source import PowerSource
from hybrid.sites import SiteInfo


class CspPlant(PowerSource):
    _system_model: None
    _financial_model: Singleowner
    # _layout: TroughLayout
    _dispatch: CspDispatch

    def __init__(self,
                 name: str,
                 tech_name: str,
                 site: SiteInfo,
                 financial_model: Singleowner,
                 csp_config: dict):
        """

        :param trough_config: dict, with keys ('system_capacity_kw', 'solar_multiple', 'tes_hours')
        """
        required_keys = ['cycle_capacity_kw', 'solar_multiple', 'tes_hours']
        if any(key not in csp_config.keys() for key in required_keys):
            is_missing = [key not in csp_config.keys() for key in required_keys]
            missing_keys = [missed_key for (missed_key, missing) in zip(required_keys, is_missing) if missing]
            raise ValueError(type(self).__name__ + " requires the following keys: " + str(missing_keys))

        self.name = name
        self.site = site

        self._financial_model = financial_model
        self._layout = None
        self._dispatch = CspDispatch
        self.set_construction_financing_cost_per_kw(0)

        # TODO: Should 'SSC' object be a protected attr
        # Initialize ssc and get weather data
        self.ssc = ssc_wrap(
            wrapper='pyssc',  # ['pyssc' | 'pysam']
            tech_name=tech_name,  # ['tcsmolten_salt' | 'trough_physical]
            financial_name=None,
            defaults_name=None)  # ['MSPTSingleOwner' | 'PhysicalTroughSingleOwner']  NOTE: not used for pyssc
        self.initialize_params()

        self.year_weather_df = self.tmy3_to_df()  # read entire weather file

        self.cycle_capacity_kw: float = csp_config['cycle_capacity_kw']
        self.solar_multiple: float = csp_config['solar_multiple']
        self.tes_hours: float = csp_config['tes_hours']

        self.cycle_efficiency_tables = self.get_cycle_efficiency_tables()

    def param_file_paths(self, relative_path):
        cwd = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(cwd, relative_path)
        for key in self.param_files.keys():
            filename = self.param_files[key]
            self.param_files[key] = os.path.join(data_path, filename)

    def initialize_params(self):
        self.set_params_from_files()
        self.ssc.set({'time_steps_per_hour': 1})  # FIXME: defaults to 60
        n_steps_year = int(8760 * self.ssc.get('time_steps_per_hour'))
        self.ssc.set({'sf_adjust:hourly': n_steps_year * [0]})

    def tmy3_to_df(self):
        # NOTE: be careful of leading spaces in the column names, they are hard to catch and break the parser
        df = pd.read_csv(self.site.solar_resource.filename, sep=',', skiprows=2, header=0)
        date_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
        df.index = pd.to_datetime(df[date_cols])
        df.index.name = 'datetime'
        df.drop(date_cols, axis=1, inplace=True)

        df.index = df.index.map(lambda t: t.replace(year=df.index[0].year))  # normalize all years to that of 1/1
        df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]  # drop unnamed columns (which are empty)
        timestep = df.index[1] - df.index[0]
        if timestep == datetime.timedelta(hours=1) and df.index[0].minute == 30:
            df.index = df.index.map(
                lambda t: t.replace(minute=0))  # make minute convention 0 instead of 30 in hourly files

        def get_weatherfile_location(tmy3_path):
            df_meta = pd.read_csv(tmy3_path, sep=',', header=0, nrows=1)
            return {
                'latitude': float(df_meta['Latitude'][0]),
                'longitude': float(df_meta['Longitude'][0]),
                'timezone': int(df_meta['Time Zone'][0]),
                'elevation': float(df_meta['Elevation'][0])
            }

        location = get_weatherfile_location(self.site.solar_resource.filename)
        df.attrs.update(location)
        return df

    def set_params_from_files(self):
        # Loads default case
        with open(self.param_files['tech_model_params_path'], 'r') as f:
            ssc_params = rapidjson.load(f)
        self.ssc.set(ssc_params)

        # NOTE: Don't set if passing weather data in via solar_resource_data
        # ssc.set({'solar_resource_file': param_files['solar_resource_file_path']})

        dispatch_factors_ts = np.array(pd.read_csv(self.param_files['dispatch_factors_ts_path']))
        self.ssc.set({'dispatch_factors_ts': dispatch_factors_ts})
        # TODO: remove dispatch factor file and use site
        # self.ssc.set({'dispatch_factors_ts': self.site.elec_prices.data})  # returning a empty array...

        ud_ind_od = np.array(pd.read_csv(self.param_files['ud_ind_od_path']))
        self.ssc.set({'ud_ind_od': ud_ind_od})

        wlim_series = np.array(pd.read_csv(self.param_files['wlim_series_path']))
        self.ssc.set({'wlim_series': wlim_series})

    def set_weather(self, weather_df: pd.DataFrame,
                    start_datetime: datetime = None,
                    end_datetime: datetime = None):
        """
        Sets 'solar_resource_data' for pySSC simulation. If start and end (datetime) are not provided, full year is
        assumed.
        :param weather_df: weather information
        :param start_datetime: start of pySSC simulation (datetime)
        :param end_datetime: end of pySSC simulation (datetime)
        """
        weather_timedelta = weather_df.index[1] - weather_df.index[0]
        weather_time_steps_per_hour = int(1 / (weather_timedelta.total_seconds() / 3600))
        ssc_time_steps_per_hour = self.ssc.get('time_steps_per_hour')
        if weather_time_steps_per_hour != ssc_time_steps_per_hour:
            raise Exception('Configured time_steps_per_hour ({x}) is not that of weather file ({y})'.format(
                x=ssc_time_steps_per_hour, y=weather_time_steps_per_hour))

        if start_datetime is None and end_datetime is None:
            if len(weather_df) != ssc_time_steps_per_hour * 8760:
                raise Exception('Full year weather dataframe required if start and end datetime are not provided')
            weather_df_part = weather_df
        else:
            weather_year = weather_df.index[0].year
            if start_datetime.year != weather_year:
                print('Replacing start and end years ({x}) with weather file\'s ({y}).'.format(
                    x=start_datetime.year, y=weather_year))
                start_datetime = start_datetime.replace(year=weather_year)
                end_datetime = end_datetime.replace(year=weather_year)

            if end_datetime <= start_datetime:
                end_datetime = start_datetime + weather_timedelta
            weather_df_part = weather_df[start_datetime:(
                        end_datetime - weather_timedelta)]  # times in weather file are the start (or middle) of timestep

        def weather_df_to_ssc_table(weather_df):
            rename_from_to = {
                'Tdry': 'Temperature',
                'Tdew': 'Dew Point',
                'RH': 'Relative Humidity',
                'Pres': 'Pressure',
                'Wspd': 'Wind Speed',
                'Wdir': 'Wind Direction'
            }
            weather_df = weather_df.rename(columns=rename_from_to)

            solar_resource_data = {}
            solar_resource_data['tz'] = weather_df.attrs['timezone']
            solar_resource_data['elev'] = weather_df.attrs['elevation']
            solar_resource_data['lat'] = weather_df.attrs['latitude']
            solar_resource_data['lon'] = weather_df.attrs['longitude']
            solar_resource_data['year'] = list(weather_df.index.year)
            solar_resource_data['month'] = list(weather_df.index.month)
            solar_resource_data['day'] = list(weather_df.index.day)
            solar_resource_data['hour'] = list(weather_df.index.hour)
            solar_resource_data['minute'] = list(weather_df.index.minute)
            solar_resource_data['dn'] = list(weather_df['DNI'])
            solar_resource_data['df'] = list(weather_df['DHI'])
            solar_resource_data['gh'] = list(weather_df['GHI'])
            solar_resource_data['wspd'] = list(weather_df['Wind Speed'])
            solar_resource_data['tdry'] = list(weather_df['Temperature'])
            solar_resource_data['pres'] = list(weather_df['Pressure'])
            solar_resource_data['tdew'] = list(weather_df['Dew Point'])

            def pad_solar_resource_data(solar_resource_data):
                datetime_start = datetime.datetime(
                    year=solar_resource_data['year'][0],
                    month=solar_resource_data['month'][0],
                    day=solar_resource_data['day'][0],
                    hour=solar_resource_data['hour'][0],
                    minute=solar_resource_data['minute'][0])
                n = len(solar_resource_data['dn'])
                if n < 2:
                    timestep = datetime.timedelta(hours=1)  # assume 1 so minimum of 8760 results
                else:
                    datetime_second_time = datetime.datetime(
                        year=solar_resource_data['year'][1],
                        month=solar_resource_data['month'][1],
                        day=solar_resource_data['day'][1],
                        hour=solar_resource_data['hour'][1],
                        minute=solar_resource_data['minute'][1])
                    timestep = datetime_second_time - datetime_start
                steps_per_hour = int(3600 / timestep.seconds)
                # Substitute a non-leap year (2009) to keep multiple of 8760 assumption:
                i0 = int((datetime_start.replace(year=2009) - datetime.datetime(2009, 1, 1, 0, 0,
                                                                                0)).total_seconds() / timestep.seconds)
                diff = 8760 * steps_per_hour - n
                front_padding = [0] * i0
                back_padding = [0] * (diff - i0)

                if diff > 0:
                    for k in solar_resource_data:
                        if isinstance(solar_resource_data[k], list):
                            solar_resource_data[k] = front_padding + solar_resource_data[k] + back_padding
                    return solar_resource_data
                else:
                    return solar_resource_data

            solar_resource_data = pad_solar_resource_data(solar_resource_data)
            return solar_resource_data

        self.ssc.set({'solar_resource_data': weather_df_to_ssc_table(weather_df_part)})

    def get_cycle_efficiency_tables(self) -> dict:
        """
        Gets off-design cycle performance tables from pySSC.
        :return cycle_efficiency_tables: if tables exist, tables are return,
                                        else if user defined cycle, tables are calculated,
                                        else return emtpy dictionary with warning
        """
        start_datetime = datetime.datetime(self.year_weather_df.index[0].year, 1, 1, 0, 0, 0)  # start of first timestep
        self.set_weather(self.year_weather_df, start_datetime, start_datetime)  # only one weather timestep is needed
        self.ssc.set({'time_start': 0})
        self.ssc.set({'time_stop': 0})
        ssc_outputs = self.ssc.execute()

        required_tables = ['cycle_eff_load_table', 'cycle_eff_Tdb_table', 'cycle_wcond_Tdb_table']
        if all(table in ssc_outputs for table in required_tables):
            return {table: ssc_outputs[table] for table in required_tables}
        if ssc_outputs['pc_config'] == 1:
            # Tables not returned from ssc, but can be taken from user-defined cycle inputs
            return {'ud_ind_od': ssc_outputs['ud_ind_od']}
        else:
            print('WARNING: Cycle efficiency tables not found. Dispatch optimization will assume a constant cycle '
                  'efficiency and no ambient temperature dependence.')
            return {}

    def simulate_with_dispatch(self, n_periods: int, sim_start_time: int = None):
        """
        Step through dispatch solution and simulate trough system
        """
        # Set up start and end time of simulation
        start_datetime, end_datetime = CspDispatch.get_start_end_datetime(sim_start_time, n_periods)
        self.value('time_start', CspDispatch.seconds_since_newyear(start_datetime))
        self.value('time_stop', CspDispatch.seconds_since_newyear(end_datetime))

        self.set_dispatch_targets(n_periods)

        # Simulate
        results = self.ssc.execute()
        if not results["cmod_success"]:
            raise ValueError('PySSC simulation failed...')
        # Save plant state at end of simulation

        # Save simulation output TODO: save information in a structure



        pass

    def set_dispatch_targets(self, n_periods: int):
        """Set pySSC targets using dispatch model solution."""
        # Set targets
        dis = self.dispatch

        dispatch_targets = {'is_dispatch_targets': 1,
                            # Receiver on, startup, (or standby - NOT in dispatch currently)
                            'is_rec_su_allowed_in': [1 if (dis.is_field_generating[t] + dis.is_field_starting[t]) > 0.01
                                                     else 0 for t in range(n_periods)],
                            # Receiver standby - NOT in dispatch currently
                            'is_rec_sb_allowed_in': [0 for t in range(n_periods)],
                            # Cycle on or startup
                            'is_pc_su_allowed_in': [1 if (dis.is_cycle_generating[t] + dis.is_cycle_starting[t]) > 0.01
                                                    else 0 for t in range(n_periods)],
                            # Cycle standby - NOT in dispatch currently
                            'is_pc_sb_allowed_in': [0 for t in range(n_periods)],
                            # Cycle start up thermal power
                            'q_pc_target_su_in': [dis.allowable_cycle_startup_power if dis.is_cycle_starting[t] > 0.01
                                                  else 0.0 for t in range(n_periods)],
                            # Cycle thermal power
                            'q_pc_target_on_in': dis.cycle_thermal_power[0:n_periods],
                            # Cycle max thermal power allowed
                            'q_pc_max_in': [self.cycle_thermal_rating for t in range(n_periods)]}
        self.ssc.set(dispatch_targets)

    @property
    def _system_model(self):
        """Used for dispatch to mimic other dispatch class building in hybrid dispatch builder"""
        return self

    @property
    def system_capacity_kw(self) -> float:
        return self.cycle_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the power cycle capacity and updates the system model
        :param size_kw:
        :return:
        """
        self.cycle_capacity_kw = size_kw

    @property
    def cycle_capacity_kw(self) -> float:
        """ P_ref is in [MW] returning [kW] """
        return self.ssc.get('P_ref') * 1000.

    @cycle_capacity_kw.setter
    def cycle_capacity_kw(self, size_kw: float):
        """
        Sets the power cycle capacity and updates the system model TODO:, cost and financial model
        :param size_kw:
        :return:
        """
        self.ssc.set({'P_ref': size_kw / 1000.})

    @property
    def solar_multiple(self) -> float:
        raise NotImplementedError

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        raise NotImplementedError

    @property
    def tes_hours(self) -> float:
        return self.ssc.get('tshours')

    @tes_hours.setter
    def tes_hours(self, tes_hours: float):
        """
        Equivalent full-load thermal storage hours [hr]
        :param tes_hours:
        :return:
        """
        self.ssc.set({'tshours': tes_hours})

    @property
    def cycle_thermal_rating(self) -> float:
        raise NotImplementedError

    @property
    def field_thermal_rating(self) -> float:
        raise NotImplementedError

    @property
    def cycle_nominal_efficiency(self) -> float:
        raise NotImplementedError

    @property
    def number_of_reflector_units(self) -> float:
        raise NotImplementedError

    @property
    def minimum_receiver_power_fraction(self) -> float:
        raise NotImplementedError

    @property
    def field_tracking_power(self) -> float:
        raise NotImplementedError

    def value(self, var_name, var_value=None):
        attr_obj = None
        ssc_value = None
        if var_name in self.__dir__():
            attr_obj = self
        if not attr_obj:
            for a in self._financial_model.__dir__():
                group_obj = getattr(self._financial_model, a)
                try:
                    if var_name in group_obj.__dir__():
                        attr_obj = group_obj
                        break
                except:
                    pass
        if not attr_obj:
            try:
                ssc_value = self.ssc.get(var_name)
                attr_obj = self.ssc
            except:
                pass
        if not attr_obj:
            raise ValueError("Variable {} not found in technology or financial model {}".format(
                var_name, self.__class__.__name__))

        if var_value is None:
            if ssc_value is None:
                return getattr(attr_obj, var_name)
            else:
                return ssc_value
        else:
            try:
                if ssc_value is None:
                    setattr(attr_obj, var_name, var_value)
                else:
                    self.ssc.set({var_name: var_value})
            except Exception as e:
                raise IOError(f"{self.__class__}'s attribute {var_name} could not be set to {var_value}: {e}")

    def set_construction_financing_cost_per_kw(self, construction_financing_cost_per_kw):
        # TODO: CSP doesn't scale per kw -> need to update?
        self._construction_financing_cost_per_kw = construction_financing_cost_per_kw

    def get_construction_financing_cost(self) -> float:
        return self._construction_financing_cost_per_kw * self.system_capacity_kw

    def simulate(self, project_life: int = 25, skip_fin=False):
        """
        Run the system and financial model
        """
        raise NotImplementedError
        # if not self._system_model:
        #     return
        #
        # if self.system_capacity_kw <= 0:
        #     return
        #
        # if project_life > 1:
        #     self._financial_model.Lifetime.system_use_lifetime_output = 1
        # else:
        #     self._financial_model.Lifetime.system_use_lifetime_output = 0
        # self._financial_model.FinancialParameters.analysis_period = project_life
        #
        # self._system_model.execute(0)
        #
        # if skip_fin:
        #     return
        #
        # self._financial_model.SystemOutput.gen = self._system_model.value("gen")
        # self._financial_model.value("construction_financing_cost", self.get_construction_financing_cost())
        # self._financial_model.Revenue.ppa_soln_mode = 1
        # if len(self._financial_model.SystemOutput.gen) == self.site.n_timesteps:
        #     single_year_gen = self._financial_model.SystemOutput.gen
        #     self._financial_model.SystemOutput.gen = list(single_year_gen) * project_life
        #
        # if self.name != "Grid":
        #     self._financial_model.SystemOutput.system_pre_curtailment_kwac = self._system_model.value("gen") * project_life
        #     self._financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self._system_model.value("annual_energy")
        #
        # self._financial_model.execute(0)
        # logger.info(f"{self.name} simulation executed with AEP {self.annual_energy_kw}")

    #
    # Outputs
    #
    @property
    def dispatch(self):
        return self._dispatch

    # TODO: create a outputs struct that allows ssc to store results as we step through the year,
    #  then update below outputs calls

    @property
    def annual_energy_kw(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("annual_energy")
        else:
            return 0

    @property
    def generation_profile(self) -> list:
        if self.system_capacity_kw:
            return list(self._system_model.value("gen"))
        else:
            return [0] * self.site.n_timesteps

    @property
    def capacity_factor(self) -> float:
        if self.system_capacity_kw > 0:
            return self._system_model.value("capacity_factor")
        else:
            return 0
