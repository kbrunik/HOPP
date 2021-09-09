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
    """Testing pySSC tower model"""
    csp_config = {'cycle_capacity_kw': 110 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}   # NOTE: not being used yet

    csp = CspPlant(site, csp_config)

    start_datetime = datetime.datetime(2018, 10, 21, 0, 0, 0)               # start of first timestep
    end_datetime = datetime.datetime(2018, 10, 24, 0, 0, 0)                 # end of last timestep
    csp.initialize_params(keep_eta_flux_maps=True)
    csp.ssc.set({'time_start': csp.seconds_since_newyear(start_datetime)})
    csp.ssc.set({'time_stop': csp.seconds_since_newyear(end_datetime)})
    csp.set_weather(csp.year_weather_df, start_datetime, end_datetime)
    tech_outputs = csp.ssc.execute()
    print('Three days all at once starting 10/21/2018, annual energy = {e:.0f} MWhe'.format(e=tech_outputs['annual_energy'] * 1.e-3))
    pass


def test_default_trough_model(site):
    """Testing PySAM trough model using heuristic dispatch method """
    trough_config = {'cycle_capacity_kw': 110 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}

    # Getting default values
    trough_model = Trough.default('PhysicalTroughSingleOwner')
    trough_config = {'cycle_capacity_kw': trough_model.value('P_ref') * 1000.,
                    'solar_multiple': trough_model.value('specified_solar_multiple'),
                    'tes_hours': trough_model.value('tshours')}

    model = TroughPlant(site, trough_config)
    sr_data = model._system_model.value('solar_resource_data')
    #filename = model.value('file_name')        # This doesn't exist in trough
    model.simulate(1)
    print("CSP Trough annual energy (TroughPlant): " + str(model.value('annual_energy')))  # 333950296.71266896


    trough_model = Trough.default('PhysicalTroughSingleOwner')
    trough_model.value('solar_resource_data', sr_data)
    #trough_model.value('file_name', sr_data)
    trough_model.execute()
    annual_energy = trough_model.value('annual_energy')
    print("CSP Trough annual energy (direct): " + str(annual_energy))   # 333950296.71266896

    assert annual_energy > 0.0
    assert model.value('annual_energy') == pytest.approx(annual_energy, 1e-5)

