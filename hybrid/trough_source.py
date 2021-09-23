from typing import Optional, Union, Sequence
import os

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.trough_dispatch import TroughDispatch

from hybrid.power_source import *
from hybrid.csp_source import CspPlant


class TroughPlant(CspPlant):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    # _layout: TroughLayout
    _dispatch: TroughDispatch

    def __init__(self,
                 site: SiteInfo,
                 trough_config: dict):
        """

        :param trough_config: dict, with keys ('system_capacity_kw', 'solar_multiple', 'tes_hours')
        """
        financial_model = Singleowner.default('PhysicalTroughSingleOwner')

        # set-up param file paths
        # TODO: Site should have dispatch factors consistent across all models
        self.param_files = {'tech_model_params_path': 'tech_model_defaults.json',
                            'dispatch_factors_ts_path': 'dispatch_factors_ts.csv',
                            'ud_ind_od_path': 'ud_ind_od.csv',
                            'wlim_series_path': 'wlim_series.csv'}
        rel_path_to_param_files = os.path.join('pySSC_daotk', 'trough_data')
        self.param_file_paths(rel_path_to_param_files)

        super().__init__("TroughPlant", 'trough_physical', site, financial_model, trough_config)
        # Set weather once
        self.set_weather(self.year_weather_df)

        self._dispatch: TroughDispatch = None

    def calculate_total_installed_cost(self) -> float:
        # TODO: Janna copy SSC calculations here
        return 0.0

    @staticmethod
    def estimate_receiver_pumping_parasitic():
        return 0.0125  # [MWe/MWt] Assuming because troughs pressure drop is difficult to estimate reasonably

    @staticmethod
    def get_plant_state_io_map() -> dict:
        io_map = {  # State:
                  # Number Inputs                         # Arrays Outputs
                  'defocus_initial':                      'defocus_final',
                  'rec_op_mode_initial':                  'rec_op_mode_final',
                  'T_in_loop_initial':                    'T_in_loop_final',
                  'T_out_loop_initial':                   'T_out_loop_final',
                  'T_out_scas_initial':                   'T_out_scas_last_final',        # array

                  'T_tank_cold_init':                     'T_tes_cold',
                  'T_tank_hot_init':                      'T_tes_hot',
                  'init_hot_htf_percent':                 'hot_tank_htf_percent_final',

                  'pc_op_mode_initial':                   'pc_op_mode_final',
                  'pc_startup_time_remain_init':          'pc_startup_time_remain_final',
                  'pc_startup_energy_remain_initial':     'pc_startup_energy_remain_final',
                  # For dispatch ramping penalty
                  'heat_into_cycle':                      'q_pb'
                  }
        return io_map

    def set_initial_plant_state(self) -> dict:
        plant_state = super().set_initial_plant_state()
        # Use initial storage charge state that came from tech_model_defaults.json file
        plant_state['init_hot_htf_percent'] = self.value('init_hot_htf_percent')
        plant_state.pop('T_out_scas_initial')   # initially not needed
        return plant_state

    @property
    def solar_multiple(self) -> float:
        return self.ssc.get('specified_solar_multiple')

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        """
        Set the solar multiple and updates the system model. Solar multiple is defined as the the ratio of receiver
        design thermal power over power cycle design thermal power.
        :param solar_multiple:
        :return:
        """
        self.ssc.set({'specified_solar_multiple': solar_multiple})

    @property
    def cycle_thermal_rating(self) -> float:
        return self.value('P_ref') / self.value('eta_ref')

    @property
    def field_thermal_rating(self) -> float:
        # TODO: This doesn't work with specified field area option
        return self.value('specified_solar_multiple') * self.cycle_thermal_rating

    @property
    def cycle_nominal_efficiency(self) -> float:
        return self.value('eta_ref')

    @property
    def number_of_reflector_units(self) -> float:
        """Returns number of solar collector assemblies within the field."""
        return self.value('nSCA') * self.value('nLoops') * self.value('FieldConfig')

    @property
    def minimum_receiver_power_fraction(self) -> float:
        """Returns minimum field mass flowrate fraction."""
        return self.value('m_dot_htfmin')/self.value('m_dot_htfmax')

    @property
    def field_tracking_power(self) -> float:
        """Returns power load for field to track sun position in MWe"""
        return self.value('SCA_drives_elec') * self.number_of_reflector_units / 1e6  # W to MW

    @property
    def htf_cold_design_temperature(self) -> float:
        """Returns cold design temperature for HTF [C]"""
        return self.value('T_loop_in_des')

    @property
    def htf_hot_design_temperature(self) -> float:
        """Returns hot design temperature for HTF [C]"""
        return self.value('T_loop_out')

    @property
    def initial_tes_hot_mass_fraction(self) -> float:
        """Returns initial thermal energy storage fraction of mass in hot tank [-]"""
        return self.plant_state['init_hot_htf_percent'] / 100.

