from hybrid.solar.layout import calculate_solar_extent
from tests.data.defaults_data import defaults

def test_pv_extent():
    pv = defaults['Solar']['Pvsamv1']
    solar_extent = calculate_solar_extent(pv)
    assert(solar_extent == (48, 7307, 2))


