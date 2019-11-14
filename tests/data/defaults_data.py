from defaults.pv_singleowner import PV_pvsingleowner, Singleowner_pvsingleowner
from defaults.wind_singleowner import wind_windsingleowner, singleowner_windsingleowner
from defaults.genericsystem_singleowner import genericsystem_genericsystemsingleowner, battery_genericsystemsingleowner, \
    singleowner_genericsystemsingleowner

Site = {
    'site': {
        "lat": 39.7555,
        "lon": -105.2211,
        "elev": 1879,
        "year": 2012,
        "tz": -7
    }
}

defaults = {
    'Solar': {
        'Pvsamv1': PV_pvsingleowner,
        'Singleowner': Singleowner_pvsingleowner
    },
    'Wind': {
        'Windpower': wind_windsingleowner,
        'Singleowner': singleowner_windsingleowner
    },
    'Generic': {
        'GenericSystem': genericsystem_genericsystemsingleowner,
        'StandAloneBattery': battery_genericsystemsingleowner,
        'Singleowner': singleowner_genericsystemsingleowner
    }
}