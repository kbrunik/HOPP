from pyomo.environ import ConcreteModel, Set
import datetime

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch


class TroughDispatch(CspDispatch):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    """

    """
    def __init__(self,
                 pyomo_model: ConcreteModel,
                 indexed_set: Set,
                 system_model: None,
                 financial_model: Singleowner.Singleowner,
                 block_set_name: str = 'trough'):
        super().__init__(pyomo_model, indexed_set, system_model, financial_model, block_set_name=block_set_name)

    def update_time_series_dispatch_model_parameters(self, start_time: int):
        """
        This is where we need to simulate the future and capture performance for dispatch parameters
        : param start_time: hour of the year starting dispatch horizon
        """
        n_horizon = len(self.blocks.index_set())
        super().update_time_series_dispatch_model_parameters(start_time)
        tech_outputs = self._system_model.ssc.execute()

        # TODO: set up simulation for the next n_horizon...
        #  - Does storage need to be increased to ensure no curtailment?

        self.available_thermal_generation = tech_outputs['q_inc_sf_tot'][0:n_horizon]
        # TODO: could update ssc to calculate 'disp_pceff_expected' and output condenser load estimates...
        #  Both estimates are driven by Tdry
        self.cycle_ambient_efficiency_correction = [1.0]*n_horizon
        self.condenser_losses = [0.0]*n_horizon

    def update_initial_conditions(self):
        # FIXME: There is a bit of work to do here
        # TODO: set these values here
        self.initial_thermal_energy_storage = 0.0  # Might need to calculate this

        # TODO: This appears to be coming from AMPL data files... This will take getters to be set up in pySAM...
        self.initial_receiver_startup_inventory = (self.receiver_required_startup_energy
                                                   - self._system_model.value('rec_startup_energy_remain_final') )
        self.is_field_generating_initial = self._system_model.value('is_field_tracking_final')
        self.is_field_starting_initial = self._system_model.value('rec_op_mode_final')  # TODO: this is not right

        self.initial_cycle_startup_inventory = (self.cycle_required_startup_energy
                                                - self._system_model.value('pc_startup_energy_remain_final') )
        self.initial_cycle_thermal_power = self._system_model.value('q_pb')
        self.is_cycle_generating_initial = self._system_model.value('pc_op_mode_final')  # TODO: figure out what this is...
        self.is_cycle_starting_initial = False
