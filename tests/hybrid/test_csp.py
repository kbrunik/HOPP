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
    tower_config = {'cycle_capacity_kw': 110 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}   # NOTE: not being used yet

    expected_energy = 5512681.74

    csp = TowerPlant(site, tower_config)

    start_datetime = datetime.datetime(2018, 10, 21, 0, 0, 0)               # start of first timestep
    end_datetime = datetime.datetime(2018, 10, 24, 0, 0, 0)                 # end of last timestep
    csp.initialize_params(keep_eta_flux_maps=True)
    csp.ssc.set({'time_start': csp.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': csp.seconds_since_newyear(end_datetime)})
    csp.set_weather(csp.year_weather_df, start_datetime, end_datetime)
    tech_outputs = csp.ssc.execute()
    print('Three days all at once starting 10/21/2018, annual energy = {e:.0f} MWhe'.format(e=tech_outputs['annual_energy'] * 1.e-3))

    assert tech_outputs['annual_energy'] == pytest.approx(expected_energy, 1e-5)


def test_pySSC_trough_model(site):
    """Testing pySSC trough model using heuristic dispatch method"""
    trough_config = {'cycle_capacity_kw': 110 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}   # NOTE: not being used yet

    expected_energy = 3170500.729

    csp = TroughPlant(site, trough_config)

    start_datetime = datetime.datetime(2018, 10, 21, 0, 0, 0)               # start of first timestep
    end_datetime = datetime.datetime(2018, 10, 24, 0, 0, 0)                 # end of last timestep
    csp.initialize_params(keep_eta_flux_maps=True)
    csp.ssc.set({'time_start': csp.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': csp.seconds_since_newyear(end_datetime)})
    csp.set_weather(csp.year_weather_df, start_datetime, end_datetime)
    tech_outputs = csp.ssc.execute()
    print('Three days all at once starting 10/21/2018, annual energy = {e:.0f} MWhe'.format(e=tech_outputs['annual_energy'] * 1.e-3))

    assert tech_outputs['annual_energy'] == pytest.approx(expected_energy, 1e-5)


