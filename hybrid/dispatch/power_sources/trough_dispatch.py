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
