from typing import Optional, Union, Sequence
import os
import datetime

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.tower_dispatch import TowerDispatch
from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch


from hybrid.power_source import *
from hybrid.csp_source import CspPlant

class TowerPlant(CspPlant):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    # _layout: TowerLayout
    _dispatch: TowerDispatch

    def __init__(self,
                 site: SiteInfo,
                 tower_config: dict):
        """

        :param tower_config: dict, with keys ('cycle_capacity_kw', 'solar_multiple', 'tes_hours')
        """
        financial_model = Singleowner.default('MSPTSingleOwner')

        # set-up param file paths
        # TODO: Site should have dispatch factors consistent across all models
        # TODO: get
        self.param_files = {'tech_model_params_path': 'tech_model_defaults.json',
                            'dispatch_factors_ts_path': 'dispatch_factors_ts.csv',
                            'ud_ind_od_path': 'ud_ind_od.csv',
                            'wlim_series_path': 'wlim_series.csv',
                            'helio_positions_path': 'helio_positions.csv'}
        rel_path_to_param_files = os.path.join('pySSC_daotk', 'tower_data')
        self.param_file_paths(rel_path_to_param_files)

        super().__init__("TowerPlant", 'tcsmolten_salt', site, financial_model, tower_config)

        # TODO: This needs to update layout, tower, and receiver based on user inputs
        # Calculate flux and eta maps for all simulations
        self.set_field_layout_and_flux_eta_maps(self.create_field_layout_and_simulate_flux_eta_maps())

        # Set weather once -> required to be after set_flux_eta_maps call
        self.set_weather(self.year_weather_df)

        self.set_intial_plant_state()
        self.update_ssc_inputs_from_plant_state()

        self._dispatch: TowerDispatch = None

    def initialize_params(self, keep_eta_flux_maps=False):
        if keep_eta_flux_maps:
            flux_eta_maps = {k:self.ssc.get(k) for k in ['eta_map', 'flux_maps', 'A_sf_in', 'helio_positions', 'N_hel', 'D_rec', 'rec_height', 'h_tower', 'land_area_base']}

        super().initialize_params()

        if keep_eta_flux_maps:
            self.set_flux_eta_maps(flux_eta_maps)

    def set_params_from_files(self):
        super().set_params_from_files()

        # load heliostat field  # TODO: Can we get rid of this if always creating a new field layout?
        heliostat_layout = np.genfromtxt(self.param_files['helio_positions_path'], delimiter=',')
        N_hel = heliostat_layout.shape[0]
        helio_positions = [heliostat_layout[j, 0:2].tolist() for j in range(N_hel)]
        self.ssc.set({'helio_positions': helio_positions})

    def create_field_layout_and_simulate_flux_eta_maps(self):
        start_datetime = datetime.datetime(1900, 1, 1, 0, 0, 0)  # start of first timestep
        self.set_weather(self.year_weather_df, start_datetime, start_datetime)  # only one weather timestep is needed
        print('Creating field layout and simulating flux and eta maps ...')
        self.ssc.set({'time_start': 0})
        self.ssc.set({'time_stop': 0})

        # TODO: change to field_model_type = 0 to optimize receiver/tower sizing
        self.ssc.set({'field_model_type': 1})  # Create field layout and generate flux and eta maps, but don't optimize field or tower 

        # Check if specified receiver dimensions make sense relative to heliostat dimensions (if not optimizing receiver sizing)
        if self.ssc.get('field_model_type') > 0 and min(self.ssc.get('rec_height'), self.ssc.get('D_rec')) < max(self.ssc.get('helio_width'), self.ssc.get('helio_height')):
            print('Warning: Receiver height or diameter is smaller than the heliostat dimension')

        original_values = {k: self.ssc.get(k) for k in['is_dispatch_targets', 'rec_clearsky_model', 'time_steps_per_hour', 'sf_adjust:hourly']}
        # set so unneeded dispatch targets and clearsky DNI are not required
        # TODO: probably don't need hourly sf adjustment factors
        self.ssc.set({'is_dispatch_targets': False, 'rec_clearsky_model': 1, 'time_steps_per_hour': 1,
                      'sf_adjust:hourly': [0.0 for j in range(8760)]})
        tech_outputs = self.ssc.execute()
        print('Finished creating field layout and simulating flux and eta maps ...')
        self.ssc.set(original_values)
        eta_map = tech_outputs["eta_map_out"]
        flux_maps = [r[2:] for r in tech_outputs['flux_maps_for_import']]  # don't include first two columns
        A_sf_in = tech_outputs["A_sf"]
        field_and_flux_maps = {'eta_map': eta_map, 'flux_maps': flux_maps, 'A_sf_in': A_sf_in}
        for k in ['helio_positions', 'N_hel', 'D_rec', 'rec_height', 'h_tower', 'land_area_base']:
            field_and_flux_maps[k] = tech_outputs[k]
        return field_and_flux_maps

    def set_field_layout_and_flux_eta_maps(self, field_and_flux_maps):
        self.ssc.set(field_and_flux_maps)  # set flux maps etc. so they don't have to be recalculated
        self.ssc.set({'field_model_type': 3})  # use the provided flux and eta map inputs
        self.ssc.set({'eta_map_aod_format': False})  #


    def get_plant_state_io_map(self):
        io_map = { # State:
                   # Number Inputs                         # Arrays Outputs (end of timestep)
                   'is_field_tracking_init':               'is_field_tracking_final',
                   'rec_op_mode_initial':                  'rec_op_mode_final',
                   'rec_startup_time_remain_init':         'rec_startup_time_remain_final',
                   'rec_startup_energy_remain_init':       'rec_startup_energy_remain_final',

                   'T_tank_cold_init':                     'T_tes_cold',
                   'T_tank_hot_init':                      'T_tes_hot',
                   'csp.pt.tes.init_hot_htf_percent':      'hot_tank_htf_percent_final',

                   'pc_op_mode_initial':                   'pc_op_mode_final',
                   'pc_startup_time_remain_init':          'pc_startup_time_remain_final',
                   'pc_startup_energy_remain_initial':     'pc_startup_energy_remain_final'
                   }
        return io_map        

    # TODO: Better place to handle initial state?  Inputs are optional (with default values in ssc) and many are not included in tech_model_defaults.json. Don't "need" to initialize for ssc, but might be handy to pass to dispatch?
    def set_intial_plant_state(self):  
        io_map = self.get_plant_state_io_map()
        self.plant_state = {k:0 for k in io_map.keys()}   # Note values for inital startup time/energy requiements will be set by ssc internally if cycle or receiver is initially off
        self.plant_state['rec_op_mode_initial'] = 0  # Receiver initially off
        self.plant_state['pc_op_mode_initial'] = 3  # Cycle initially off
        self.plant_state['csp.pt.tes.init_hot_htf_percent'] = self.ssc.get('csp.pt.tes.init_hot_htf_percent')  # Use initial storage charge state that came from tech_model_defaults.json file
        self.plant_state['T_tank_cold_init'] = self.ssc.get('T_htf_cold_des')
        self.plant_state['T_tank_hot_init'] = self.ssc.get('T_htf_hot_des')
        self.plant_state['sim_time_at_last_update'] = 0.0
        return

    # TODO: Put this into the parent class, but note trough input 'T_out_scas_last_final' is not a time series and needs to be handled differently
    def set_plant_state_from_ssc_outputs(self, ssc_outputs, seconds_relative_to_start):  
        time_steps_per_hour = self.ssc.get('time_steps_per_hour')
        time_start = self.ssc.get('time_start')
        idx = round(seconds_relative_to_start/3600) * int(time_steps_per_hour) - 1  # Note: values returned in ssc_outputs are at the front of the output arrays
        io_map = self.get_plant_state_io_map()
        for input, output in io_map.items():
            self.plant_state[input] = ssc_outputs[output][idx]
        self.plant_state['sim_time_at_last_update'] = time_start + seconds_relative_to_start  # Track time at which plant state was last updated
        return

    #TODO: Put this into the parent class
    def update_ssc_inputs_from_plant_state(self):
        state = self.plant_state.copy()
        state.pop('sim_time_at_last_update') 
        self.ssc.set(state)
        return


    @property
    def solar_multiple(self) -> float:
        return self.ssc.get('solarm')

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        """
        Set the solar multiple and updates the system model. Solar multiple is defined as the the ratio of receiver
        design thermal power over power cycle design thermal power.
        :param solar_multiple:
        :return:
        """
        self.ssc.set({'solarm': solar_multiple})

    @property
    def cycle_thermal_rating(self) -> float:
        return self.value('P_ref') / self.value('design_eff')

    @property
    def field_thermal_rating(self) -> float:
        return self.value('solarm') * self.cycle_thermal_rating

    @property
    def cycle_nominal_efficiency(self) -> float:
        return self.value('design_eff')

    @property
    def number_of_reflector_units(self) -> float:
        """Returns number of heliostats within the field"""
        return self.value('N_hel')

    @property
    def minimum_receiver_power_fraction(self) -> float:
        """Returns minimum receiver mass flow rate turn down fraction."""
        return self.value('f_rec_min')

    @property
    def field_tracking_power(self) -> float:
        """Returns power load for field to track sun position in MWe"""
        return self.value('p_track') * self.number_of_reflector_units / 1e3
