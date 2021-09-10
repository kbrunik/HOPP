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
        self.param_files = {'tech_model_params_path': 'tech_model_defaults.json',
                            'dispatch_factors_ts_path': 'dispatch_factors_ts.csv',
                            'ud_ind_od_path': 'ud_ind_od.csv',
                            'wlim_series_path': 'wlim_series.csv',
                            'helio_positions_path': ''}
        rel_path_to_param_files = os.path.join('pySSC_daotk', 'trough_data')
        self.param_file_paths(rel_path_to_param_files)

        super().__init__("TroughPlant", 'trough_physical', site, financial_model, trough_config)

        self._dispatch: TroughDispatch = None

        # TODO: updated these
        # self.cycle_capacity_kw: float = trough_config['cycle_capacity_kw']
        # self.solar_multiple: float = trough_config['solar_multiple']
        # self.tes_hours: float = trough_config['tes_hours']


    def simulate_with_dispatch(self, n_periods: int, sim_start_time: int = None):
        """
        Step through dispatch solution and simulate trough system
        """
        pass


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
        return self._system_model.Powerblock.P_ref * 1000.

    @cycle_capacity_kw.setter
    def cycle_capacity_kw(self, size_kw: float):
        """
        Sets the power cycle capacity and updates the system model TODO:, cost and financial model
        :param size_kw:
        :return:
        """
        self._system_model.Powerblock.P_ref = size_kw / 1000.

    @property
    def solar_multiple(self) -> float:
        return self._system_model.Controller.specified_solar_multiple

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        """
        Set the solar multiple and updates the system model. Solar multiple is defined as the the ratio of receiver
        design thermal power over power cycle design thermal power.
        :param solar_multiple:
        :return:
        """
        self._system_model.Controller.specified_solar_multiple = solar_multiple

    @property
    def tes_hours(self) -> float:
        return self._system_model.TES.tshours

    @tes_hours.setter
    def tes_hours(self, tes_hours: float):
        """
        Equivalent full-load thermal storage hours [hr]
        :param tes_hours:
        :return:
        """
        self._system_model.TES.tshours = tes_hours