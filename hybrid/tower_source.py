from typing import Optional, Union, Sequence
import os

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.tower_dispatch import TowerDispatch


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
        self.param_files = {'tech_model_params_path': 'tech_model_defaults.json',
                            'dispatch_factors_ts_path': 'dispatch_factors_ts.csv',
                            'ud_ind_od_path': 'ud_ind_od.csv',
                            'wlim_series_path': 'wlim_series.csv',
                            'helio_positions_path': 'helio_positions.csv'}
        rel_path_to_param_files = os.path.join('pySSC_daotk', 'tower_data')
        self.param_file_paths(rel_path_to_param_files)

        super().__init__("TowerPlant", 'tcsmolten_salt', site, financial_model, tower_config)

        self._dispatch: TowerDispatch = None

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
