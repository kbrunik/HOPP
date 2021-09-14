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
        start_datetime = datetime.datetime(1900, 1, 1, 0, 0, 0)  # start of first timestep
        self.set_weather(self.year_weather_df, start_datetime, start_datetime)  # only one weather timestep is needed
        self.set_field_layout_and_flux_eta_maps(self.create_field_layout_and_simulate_flux_eta_maps())

        # Set weather once -> required to be after set_flux_eta_maps call
        self.set_weather(self.year_weather_df)

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
        print('Creating field layout and simulating flux and eta maps ...')
        self.ssc.set({'time_start': 0})
        self.ssc.set({'time_stop': 0})

        # TODO: change to field_model_type = 0 to optimize receiver/tower sizing
        self.ssc.set({'field_model_type': 1})  # Create field layout and generate flux and eta maps, but don't optimize field or tower 

        # Check if specified receiver dimensions make sense relative to heliostat dimensions (if not optimizing receiver sizing)
        if self.ssc.get('field_model_type') > 0 and min(self.ssc.get('rec_height'), self.ssc.get('D_rec')) < max(self.ssc.get('helio_width'), self.ssc.get('helio_height')):
            print('Warning: Receiver height or diameter is smaller than the heliostat dimension')

        original_values = {k: self.ssc.get(k) for k in['is_dispatch_targets', 'rec_clearsky_model', 'time_steps_per_hour', 'sf_adjust:hourly']}
        self.ssc.set({'is_dispatch_targets': False, 'rec_clearsky_model': 1, 'time_steps_per_hour': 1,
                      'sf_adjust:hourly': [0.0 for j in range(8760)]})  # set so unneeded dispatch targets and clearsky DNI are not required  # TODO: probably don't need hourly sf adjustment factors

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

