from typing import Optional, Union, Sequence
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.tower_dispatch import TowerDispatch

from hybrid.power_source import *


class TowerPlant(PowerSource):
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
        required_keys = ['cycle_capacity_kw', 'solar_multiple', 'tes_hours']
        if all(key not in tower_config.keys() for key in required_keys):
            raise ValueError

        # system_model = Tower.default('MSPTSingleOwner')
        # financial_model = Singleowner.from_existing(system_model, 'MSPTSingleOwner')
        #
        # super().__init__("TowerPlant", site, system_model, financial_model)
        #
        # self._system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        self._dispatch: TowerDispatch = None

        self.cycle_capacity_kw: float = tower_config['cycle_capacity_kw']
        self.solar_multiple: float = tower_config['solar_multiple']
        self.tes_hours: float = tower_config['tes_hours']

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
        return self._system_model.SystemDesign.P_ref * 1000.

    @cycle_capacity_kw.setter
    def cycle_capacity_kw(self, size_kw: float):
        """
        Sets the power cycle capacity and updates the system model TODO:, cost and financial model
        :param size_kw:
        :return:
        """
        self._system_model.SystemDesign.P_ref = size_kw / 1000.

    @property
    def solar_multiple(self) -> float:
        return self._system_model.SystemDesign.solarm

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        """
        Set the solar multiple and updates the system model. Solar multiple is defined as the the ratio of receiver
        design thermal power over power cycle design thermal power.
        :param solar_multiple:
        :return:
        """
        self._system_model.SystemDesign.solarm = solar_multiple

    @property
    def tes_hours(self) -> float:
        return self._system_model.SystemDesign.tshours

    @tes_hours.setter
    def tes_hours(self, tes_hours: float):
        """
        Equivalent full-load thermal storage hours [hr]
        :param tes_hours:
        :return:
        """
        self._system_model.SystemDesign.tshours = tes_hours
