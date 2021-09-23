import pytest
import pandas as pd
import datetime


from hybrid.sites import SiteInfo, flatirons_site
from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch
from hybrid.tower_source import TowerPlant
from hybrid.trough_source import TroughPlant
from hybrid.hybrid_simulation import HybridSimulation

@pytest.fixture
def site():
    return SiteInfo(flatirons_site)


def test_pySSC_tower_model(site):
    """Testing pySSC tower model using heuristic dispatch method"""
    tower_config = {'cycle_capacity_kw': 100 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}

    expected_energy = 4029953.45

    csp = TowerPlant(site, tower_config)

    start_datetime = datetime.datetime(2012, 10, 21, 0, 0, 0)  # start of first timestep
    end_datetime = datetime.datetime(2012, 10, 24, 0, 0, 0)  # end of last timestep

    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})

    # csp.ssc.create_lk_inputs_file("test.lk", csp.site.solar_resource.filename)  # Energy output: 4029953.44
    tech_outputs = csp.ssc.execute()
    annual_energy = tech_outputs['annual_energy']

    print('Three days all at once starting 10/21, annual energy = {e:.0f} MWhe'.format(e=annual_energy * 1.e-3))

    # Testing if configuration was not overwritten
    assert csp.cycle_capacity_kw == tower_config['cycle_capacity_kw']
    assert csp.solar_multiple == tower_config['solar_multiple']
    assert csp.tes_hours == tower_config['tes_hours']

    assert annual_energy == pytest.approx(expected_energy, 1e-5)


def test_pySSC_tower_increment_simulation(site):
    """Testing pySSC tower model using heuristic dispatch method and incrementing simulation"""
    tower_config = {'cycle_capacity_kw': 100 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}

    csp = TowerPlant(site, tower_config)

    start_datetime = datetime.datetime(2012, 10, 21, 0, 0, 0)  # start of first timestep
    end_datetime = datetime.datetime(2012, 10, 24, 0, 0, 0)  # end of last timestep

    increment_duration = datetime.timedelta(hours=24)  # Time duration of each simulated horizon

    # Without Increments
    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})
    tech_outputs = csp.ssc.execute()
    wo_increments_annual_energy = tech_outputs['annual_energy']

    # With increments
    n = int((end_datetime - start_datetime).total_seconds() / increment_duration.total_seconds())
    for j in range(n):
        start_datetime_new = start_datetime + j * increment_duration
        end_datetime_new = start_datetime_new + increment_duration
        csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime_new)})
        csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime_new)})
        csp.update_ssc_inputs_from_plant_state()
        tech_outputs = csp.ssc.execute()
        csp.outputs.update_from_ssc_output(tech_outputs)
        csp.set_plant_state_from_ssc_outputs(tech_outputs, increment_duration.total_seconds())

    increments_annual_energy = csp.outputs.ssc_annual['annual_energy']

    assert increments_annual_energy == pytest.approx(wo_increments_annual_energy, 1e-5)


def test_pySSC_trough_model(site):
    """Testing pySSC trough model using heuristic dispatch method"""
    trough_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}   # Different than json

    expected_energy = 2100428.199

    csp = TroughPlant(site, trough_config)

    start_datetime = datetime.datetime(2012, 10, 21, 0, 0, 0)  # start of first timestep
    end_datetime = datetime.datetime(2012, 10, 24, 1, 0, 0)  # end of last timestep
    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})

    # csp.ssc.create_lk_inputs_file("trough_test.lk", csp.site.solar_resource.filename)  # Energy output: 2100428.248543
    tech_outputs = csp.ssc.execute()
    print('Three days all at once starting 10/21, annual energy = {e:.0f} MWhe'.format(e=tech_outputs['annual_energy'] * 1.e-3))

    assert csp.cycle_capacity_kw == trough_config['cycle_capacity_kw']
    assert csp.solar_multiple == trough_config['solar_multiple']
    assert csp.tes_hours == trough_config['tes_hours']

    assert tech_outputs['annual_energy'] == pytest.approx(expected_energy, 1e-5)


def test_pySSC_trough_increment_simulation(site):
    """Testing pySSC trough model using heuristic dispatch method and incrementing simulation"""
    trough_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}

    csp = TroughPlant(site, trough_config)

    start_datetime = datetime.datetime(2012, 10, 21, 0, 0, 0)  # start of first timestep
    end_datetime = datetime.datetime(2012, 10, 24, 0, 0, 0)  # end of last timestep

    increment_duration = datetime.timedelta(hours=24)  # Time duration of each simulated horizon

    # Without Increments
    csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime)})
    tech_outputs = csp.ssc.execute()
    wo_increments_annual_energy = tech_outputs['annual_energy']

    # With increments
    n = int((end_datetime - start_datetime).total_seconds() / increment_duration.total_seconds())
    for j in range(n):
        start_datetime_new = start_datetime + j * increment_duration
        end_datetime_new = start_datetime_new + increment_duration
        csp.ssc.set({'time_start': CspDispatch.seconds_since_newyear(start_datetime_new)})
        csp.ssc.set({'time_stop': CspDispatch.seconds_since_newyear(end_datetime_new)})
        csp.update_ssc_inputs_from_plant_state()
        tech_outputs = csp.ssc.execute()
        csp.outputs.update_from_ssc_output(tech_outputs)
        csp.set_plant_state_from_ssc_outputs(tech_outputs, increment_duration.total_seconds())

    increments_annual_energy = csp.outputs.ssc_annual['annual_energy']

    assert increments_annual_energy == pytest.approx(wo_increments_annual_energy, 1e-5)


def test_value_csp_call(site):
    """Testing csp override of PowerSource value()"""
    trough_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}

    csp = TroughPlant(site, trough_config)

    # Testing value call get and set - system model
    assert csp.value('startup_time') == csp.ssc.get('startup_time')
    csp.value('startup_time', 0.25)
    assert csp.value('startup_time') == 0.25
    # financial model
    assert csp.value('inflation_rate') == csp._financial_model.FinancialParameters.inflation_rate
    csp.value('inflation_rate', 3.0)
    assert csp._financial_model.FinancialParameters.inflation_rate == 3.0
    # class setter and getter
    assert csp.value('tes_hours') == trough_config['tes_hours']
    csp.value('tes_hours', 6.0)
    assert csp.tes_hours == 6.0


def test_tower_with_dispatch_model(site):
    """Testing pySSC tower model using HOPP built-in dispatch model"""
    expected_energy = 3684265.8779387316

    interconnection_size_kw = 50000
    technologies = {'tower': {'cycle_capacity_kw': 50 * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 6.0},
                    'grid': 50000}

    system = {key: technologies[key] for key in ('tower', 'grid')}
    system = HybridSimulation(system, site,
                              interconnect_kw=interconnection_size_kw,
                              dispatch_options={'is_test_start_year': True,
                                                'is_test_end_year': True})
    system.ppa_price = (0.12, )
    system.simulate()

    assert system.tower.annual_energy_kw == pytest.approx(expected_energy, 1e-5)

    # Check dispatch targets
    disp_outputs = system.tower.outputs.dispatch
    ssc_outputs = system.tower.outputs.ssc_time_series
    for i in range(len(ssc_outputs['gen'])):
        # cycle start-up allowed
        target = 1 if (disp_outputs['is_cycle_generating'][i] + disp_outputs['is_cycle_starting'][i]) > 0.01 else 0
        assert target == pytest.approx(ssc_outputs['is_pc_su_allowed'][i], 1e-5)
        # receiver start-up allowed
        target = 1 if (disp_outputs['is_field_generating'][i] + disp_outputs['is_field_starting'][i]) > 0.01 else 0
        assert target == pytest.approx(ssc_outputs['is_rec_su_allowed'][i], 1e-5)
        # cycle thermal power
        start_power = system.tower.dispatch.allowable_cycle_startup_power if disp_outputs['is_cycle_starting'][i] else 0
        target = disp_outputs['cycle_thermal_power'][i] + start_power
        assert target == pytest.approx(ssc_outputs['q_dot_pc_target_on'][i], 1e-3)
        # thermal energy storage state-of-charge
        if i % system.dispatch_builder.options.n_roll_periods == 0:
            tes_estimate = disp_outputs['thermal_energy_storage'][i]
            tes_actual = ssc_outputs['e_ch_tes'][i]
            assert tes_estimate == pytest.approx(tes_actual, 0.01)
        # else:
        #     assert tes_estimate == pytest.approx(tes_actual, 0.15)


def test_trough_with_dispatch_model(site):
    """Testing pySSC tower model using HOPP built-in dispatch model"""
    expected_energy = 1825848.670976261

    # TODO: These don't add up...
    # 1825848.670976261 (both)
    # 1428416.7440278511 (start)
    # 435687.8311307585 (end)

    interconnection_size_kw = 50000
    technologies = {'trough': {'cycle_capacity_kw': 50 * 1000,
                              'solar_multiple': 2.0,
                              'tes_hours': 6.0},
                    'grid': 50000}

    system = {key: technologies[key] for key in ('trough', 'grid')}
    system = HybridSimulation(system, site,
                              interconnect_kw=interconnection_size_kw,
                              dispatch_options={'is_test_start_year': True,
                                                'is_test_end_year': True})
    system.ppa_price = (0.12,)
    system.simulate()

    assert system.trough.annual_energy_kw == pytest.approx(expected_energy, 1e-5)

    # TODO: This fails most like due to poor estimates of trough thermal power input
    # Check dispatch targets
    disp_outputs = system.trough.outputs.dispatch
    ssc_outputs = system.trough.outputs.ssc_time_series
    for i in range(len(ssc_outputs['gen'])):
        # cycle start-up allowed
        target = 1 if (disp_outputs['is_cycle_generating'][i] + disp_outputs['is_cycle_starting'][i]) > 0.01 else 0
        assert target == pytest.approx(ssc_outputs['is_pc_su_allowed'][i], 1e-5)
        # receiver start-up allowed
        target = 1 if (disp_outputs['is_field_generating'][i] + disp_outputs['is_field_starting'][i]) > 0.01 else 0
        assert target == pytest.approx(ssc_outputs['is_rec_su_allowed'][i], 1e-5)
        # cycle thermal power
        start_power = system.trough.dispatch.allowable_cycle_startup_power if disp_outputs['is_cycle_starting'][i] else 0
        target = disp_outputs['cycle_thermal_power'][i] + start_power
        assert target == pytest.approx(ssc_outputs['q_dot_pc_target'][i], 1e-3)
        # thermal energy storage state-of-charge
        # if i % system.dispatch_builder.options.n_roll_periods == 0:
        #     tes_estimate = disp_outputs['thermal_energy_storage'][i]
        #     tes_actual = ssc_outputs['e_ch_tes'][i]
        #     assert tes_estimate == pytest.approx(tes_actual, 0.01)
        # else:
        #     assert tes_estimate == pytest.approx(tes_actual, 0.15)


def test_tower_year_end_simulation(site):
    interconnection_size_kw = 50000
    technologies = {'tower': {'cycle_capacity_kw': 50 * 1000,
                                   'solar_multiple': 2.0,
                                   'tes_hours': 12.0},
                         'pv': {'system_capacity_kw': 50 * 1000},
                         'grid': 50000}

    solar_hybrid = {key: technologies[key] for key in ('tower', 'pv', 'grid')}
    hybrid_plant = HybridSimulation(solar_hybrid, site,
                                    interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'is_test_end_year': True})
    hybrid_plant.ppa_price = (0.12, )  # $/kWh
    hybrid_plant.pv.dc_degradation = [0] * 25

    hybrid_plant.simulate()
    # Simulate PV system:
    #hybrid_plant.pv.simulate(25)

    # Simulate End of Year CSP:



    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == pytest.approx(87692005.68, 1e-3)
    assert aeps.tower == pytest.approx(3514289.31, 1e-3)
    assert aeps.hybrid == pytest.approx(90775519.95, 1e-3)

    # TODO: check npv for csp would require a full simulation
    assert npvs.pv == pytest.approx(45233832.23, 1e3)
    #assert npvs.tower == pytest.approx(-13909363, 1e3)
    #assert npvs.hybrid == pytest.approx(-19216589, 1e3)