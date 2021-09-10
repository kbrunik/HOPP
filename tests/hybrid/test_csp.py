import pytest
import pandas as pd
import datetime


from hybrid.sites import SiteInfo, flatirons_site
from hybrid.csp_source import CspPlant
from hybrid.tower_source import TowerPlant
from hybrid.trough_source import TroughPlant

@pytest.fixture
def site():
    return SiteInfo(flatirons_site)


def test_pySSC_tower_model(site):
    """Testing pySSC tower model using heuristic dispatch method"""
    tower_config = {'cycle_capacity_kw': 100 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}

    expected_energy = 4265347.56120

    csp = TowerPlant(site, tower_config)

    start_datetime = datetime.datetime(2018, 10, 21, 0, 0, 0)               # start of first timestep
    end_datetime = datetime.datetime(2018, 10, 24, 0, 0, 0)                 # end of last timestep
    # csp.initialize_params(keep_eta_flux_maps=True) # Will result in design variables to be overwritten
    csp.ssc.set({'time_start': csp.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': csp.seconds_since_newyear(end_datetime)})
    csp.set_weather(csp.year_weather_df, start_datetime, end_datetime)
    tech_outputs = csp.ssc.execute()
    print('Three days all at once starting 10/21/2018, annual energy = {e:.0f} MWhe'.format(e=tech_outputs['annual_energy'] * 1.e-3))

    assert csp.cycle_capacity_kw == tower_config['cycle_capacity_kw']
    assert csp.solar_multiple == tower_config['solar_multiple']
    assert csp.tes_hours == tower_config['tes_hours']

    assert tech_outputs['annual_energy'] == pytest.approx(expected_energy, 1e-5)

def test_pySSC_trough_model(site):
    """Testing pySSC trough model using heuristic dispatch method"""
    trough_config = {'cycle_capacity_kw': 100 * 1000,
                     'solar_multiple': 1.5,
                     'tes_hours': 5.0}   # Different than json

    expected_energy = 2100886.0210265624

    csp = TroughPlant(site, trough_config)

    start_datetime = datetime.datetime(2018, 10, 21, 0, 0, 0)               # start of first timestep
    end_datetime = datetime.datetime(2018, 10, 24, 0, 0, 0)                 # end of last timestep

    #csp.initialize_params(keep_eta_flux_maps=True)
    # TODO: when does this need to be called? Does this have to be called before each simulation?
    #  If so, we need to store the variables within the class
    csp.ssc.set({'time_start': csp.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': csp.seconds_since_newyear(end_datetime)})
    csp.set_weather(csp.year_weather_df, start_datetime, end_datetime)
    tech_outputs = csp.ssc.execute()
    print('Three days all at once starting 10/21/2018, annual energy = {e:.0f} MWhe'.format(e=tech_outputs['annual_energy'] * 1.e-3))

    assert csp.cycle_capacity_kw == trough_config['cycle_capacity_kw']
    assert csp.solar_multiple == trough_config['solar_multiple']
    assert csp.tes_hours == trough_config['tes_hours']

    assert tech_outputs['annual_energy'] == pytest.approx(expected_energy, 1e-5)


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
