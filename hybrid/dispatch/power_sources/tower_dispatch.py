import datetime
from pyomo.environ import ConcreteModel, Set

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch


class TowerDispatch(CspDispatch):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: None,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'tower'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        """
        This is where we need to simulate the future and capture performance for dispatch parameters
        : param start_time: hour of the year starting dispatch horizon
        """
        n_horizon = len(self.blocks.index_set())
        super().update_time_series_dispatch_model_parameters(start_time)
        tech_outputs = self._system_model.ssc.execute()

        self.available_thermal_generation = tech_outputs['Q_thermal'][0:n_horizon]
        # TODO: could update ssc to calculate 'disp_pceff_expected' and output condenser load estimates...
        #  Both estimates are driven by Tdry
        # In lore this handle by set_off_design_cycle_inputs for the tower...
        # we would need to output these tables to get this for trough
        self.cycle_ambient_efficiency_correction = [1.0]*n_horizon
        self.condenser_losses = [0.0]*n_horizon
