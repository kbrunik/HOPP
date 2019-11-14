import pytest
import os

from hybrid.resource import SolarResource, WindResource

year = 2012
lat = 39.7555
lon = -105.2211
hubheight = 80


@pytest.fixture
def solar_resource():
    return SolarResource(lat=lat, lon=lon, year=year)


@pytest.fixture
def wind_resource():
    return WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=hubheight)


def test_nsrdb(solar_resource):
    assert(solar_resource.download_solar_resource(force_download=True))
    assert(solar_resource.download_solar_resource(force_download=False))


def test_wind_toolkit(wind_resource):
    assert(wind_resource.download_wind_resource(force_download=True))
    assert(wind_resource.download_wind_resource(force_download=False))


def test_wind_combine():
    path_file = os.path.dirname(os.path.abspath(__file__))

    kwargs = {'path_resource': os.path.join(path_file, 'data')}

    wind_resource = WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=70, **kwargs)

    if os.path.isfile(wind_resource.filename):
        os.remove(wind_resource.filename)

    assert(wind_resource.combine_wind_files())
