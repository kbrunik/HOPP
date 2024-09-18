import math
import numpy as np
from attrs import asdict
from pytest import approx, fixture, raises
from greenheart.simulation.technologies.mcdr import echem_mcc
from greenheart.simulation.technologies.mcdr.echem_mcc import Pump


@fixture
def power_profile():
    # EXAMPLE: Sin function for power input
    days = 365
    exTime = np.zeros(24*days) # Example time in hours
    for i in range(len(exTime)):
        exTime[i] = i+1
    maxPwr = 500 * 10**6 # W
    Amp = maxPwr/2
    periodT = 24 
    movUp = Amp
    movSide = -1*math.pi/2
    exPwr = np.zeros(len(exTime))
    for i in range(len(exTime)):
        exPwr[i] = Amp*math.sin(2*math.pi/periodT*exTime[i] + movSide) + movUp
        if int(exTime[i]/24) % 5 == 0:
            exPwr[i] = exPwr[i] * 0.25
    return exPwr


def assert_dict_equal(actual, expected, subtests):
    for key in expected:
        with subtests.test(key=key):
            assert key in actual, f"Key '{key}' not found in actual data"

            expected_value = expected[key]
            actual_value = actual[key]

            if isinstance(expected_value, dict):
                # If the value is a dictionary, recurse with subtests
                assert_dict_equal(actual_value, expected_value, subtests)
            elif isinstance(expected_value, np.ndarray):
                # If the value is a numpy array, compare arrays
                np.testing.assert_allclose(actual_value, expected_value, rtol=1e-3)
            else:
                # For other types, use simple equality
                assert (
                    actual_value == approx(expected_value,rel=1e-3)
                ), f"Mismatch found in key '{key}': {actual_value} != {expected_value}"


def test_power_chemical_ranges(subtests):
    # Run the function to get actual outputs
    co2_outputs = echem_mcc.co2_purification(
        ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3)
    )
    actual_data: echem_mcc.ElectrodialysisRangeOutputs = (
        echem_mcc.initialize_power_chemical_ranges(
            ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3),
            pump_config=echem_mcc.PumpInputs(),
            seawater_config=echem_mcc.SeaWaterInputs(),
            co2_config=co2_outputs,
        )
    )

    actual_data = asdict(actual_data)

    # Define expected outputs
    expected_data = {
        "S1": {
            "volAcid": np.array([0.0, 0.0, 0.0]),
            "volBase": np.array([0.0, 0.0, 0.0]),
            "mCC": np.array([55.39335485, 110.7867097, 166.18006455]),
            "pH_f": np.array([10.42, 10.42, 10.42]),
            "dic_f": np.array([0.00044309, 0.00044309, 0.00044309]),
            "c_a": np.array([0.44444443, 0.44444443, 0.44444443]),
            "c_b": np.array([0.44442918, 0.44442918, 0.44442918]),
            "Qin": np.array([200.0, 400.0, 600.0]),
            "Qout": np.array([200.0, 400.0, 600.0]),
            "pwrRanges": np.array([9.87450524e07, 2.07127144e08, 3.25075006e08]),
        },
        "S2": {
            "volAcid": np.array([0.0, 3600.0, 7200.0, 0.0, 3600.0, 0.0]),
            "volBase": np.array([0.0, 3600.0, 7200.0, 0.0, 3600.0, 0.0]),
            "mCC": np.array(
                [
                    55.39335485,
                    55.39335485,
                    55.39335485,
                    110.7867097,
                    110.7867097,
                    166.18006455,
                ]
            ),
            "pH_f": np.array([10.42, 10.42, 10.42, 10.42, 10.42, 10.42]),
            "dic_f": np.array(
                [0.00044309, 0.00044309, 0.00044309, 0.00044309, 0.00044309, 0.00044309]
            ),
            "c_a": np.array(
                [0.44444443, 0.44444443, 0.44444443, 0.44444443, 0.44444443, 0.44444443]
            ),
            "c_b": np.array(
                [0.44442918, 0.44442918, 0.44442918, 0.44442918, 0.44442918, 0.44442918]
            ),
            "Qin": np.array([200.0, 202.0, 204.0, 400.0, 402.0, 600.0]),
            "Qout": np.array([200.0, 200.0, 200.0, 400.0, 400.0, 600.0]),
            "pwrRanges": np.array(
                [
                    9.87450524e07,
                    1.79034386e08,
                    2.59502386e08,
                    2.07127144e08,
                    2.87683144e08,
                    3.25075006e08,
                ]
            ),
        },
        "S3": {
            "volAcid": np.array([-3600.0, -7200.0, -10800.0]),
            "volBase": np.array([-3600.0, -7200.0, -10800.0]),
            "mCC": np.array([55.34835687, 110.69671373, 166.0450706]),
            "pH_f": np.array([10.41, 10.41, 10.41]),
            "dic_f": np.array([0.00046198, 0.00046198, 0.00046198]),
            "c_a": np.array([0.44444443, 0.44444443, 0.44444443]),
            "c_b": np.array([0.44442918, 0.44442918, 0.44442918]),
            "Qin": np.array([200.0, 400.0, 600.0]),
            "Qout": np.array([202.0, 404.0, 606.0]),
            "pwrRanges": np.array(
                [18709202.21935714, 46965689.2323066, 84698352.1523112]
            ),
        },
        "S4": {
            "volAcid": np.array([3600.0, 7200.0, 10800.0]),
            "volBase": np.array([3600.0, 7200.0, 10800.0]),
            "mCC": np.array([0.0, 0.0, 0.0]),
            "pH_f": np.array([8.1, 8.1, 8.1]),
            "dic_f": np.array([0.0022, 0.0022, 0.0022]),
            "c_a": np.array([0.44444443, 0.44444443, 0.44444443]),
            "c_b": np.array([0.44442918, 0.44442918, 0.44442918]),
            "Qin": np.array([2.0, 4.0, 6.0]),
            "Qout": np.array([0.0, 0.0, 0.0]),
            "pwrRanges": np.array([8.00444444e07, 1.60266667e08, 2.40666667e08]),
        },
        "S5": {
            "volAcid": 0,
            "volBase": 0,
            "mCC": 0,
            "pH_f": 8.1,
            "dic_f": 0.0022,
            "c_a": 7.943282347242822e-09,
            "c_b": 7.6339520880939e-06,
            "Qin": 0,
            "Qout": 0,
            "pwrRanges": np.array([0.0]),
        },
        "P_minS1_tot": 98745052.42242806,
        "P_minS2_tot": 98745052.42242806,
        "P_minS3_tot": 18709202.21935714,
        "P_minS4_tot": 80044444.44444443,
        "V_aT_max": 43200.0,
        "V_bT_max": 43200.0,
        "V_a3_min": 3600.0,
        "V_b3_min": 3600.0,
        "N_range": 3,
        "S2_tot_range": 6,
        "S2_ranges": np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
            ]
        ),
        "pump_power_min": 2.266666666666666,
        "pump_power_max": 33.999999999999995,
        "vacuum_power_min": 0.655697029462398,
        "vacuum_power_max": 3.609265596301979,
    }

    assert_dict_equal(actual_data, expected_data, subtests)


def test_simulate_electrodialysis(power_profile, subtests):
    co2_outputs = echem_mcc.co2_purification(
        ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3)
    )
    ranges: echem_mcc.ElectrodialysisRangeOutputs = (
        echem_mcc.initialize_power_chemical_ranges(
            ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3),
            pump_config=echem_mcc.PumpInputs(),
            seawater_config=echem_mcc.SeaWaterInputs(),
            co2_config=co2_outputs,
        )
    )
    actual_data: echem_mcc.ElectrodialysisOutputs = echem_mcc.simulate_electrodialysis(
        ranges=ranges,
        ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3),
        power_profile=power_profile,
        initial_tank_volume_m3=0,
    )

    with subtests.test("ED_outputs: N_ed"):
        assert sum(actual_data.ED_outputs['N_ed']) == approx(14810.0)

    with subtests.test("ED_outputs: P_xs"):
        assert sum(actual_data.ED_outputs["P_xs"]) == approx(546133724365.8523)

    with subtests.test("ED_outputs: volAcid"):
        assert sum(actual_data.ED_outputs["volAcid"]) == approx(32400.0)

    with subtests.test("ED_outputs: volBase"):
        assert sum(actual_data.ED_outputs["volBase"]) == approx(32400.0)

    with subtests.test("ED_outputs: mCC"):
        assert sum(actual_data.ED_outputs["mCC"]) == approx(694918.997854312)

    with subtests.test("ED_outputs: pH_f"):
        assert sum(actual_data.ED_outputs["pH_f"]) == approx(87019.04999999255)

    with subtests.test("ED_outputs: dic_f"):
        assert sum(actual_data.ED_outputs["dic_f"]) == approx(7.124071682366895)

    with subtests.test("ED_outputs: c_a"):
        assert sum(actual_data.ED_outputs["c_a"]) == approx(3112.4443471469253)

    with subtests.test("ED_outputs: c_b"):
        assert sum(actual_data.ED_outputs["c_b"]) == approx(3112.350936165564)

    with subtests.test("ED_outputs: Qin"):
        assert sum(actual_data.ED_outputs["Qin"]) == approx(2513926.0)

    with subtests.test("ED_outputs: Qout"):
        assert sum(actual_data.ED_outputs["Qout"]) == approx(2513908.0)

    with subtests.test("capacity_factor"):
        assert actual_data.capacity_factor == approx(0.791095890410959)

    with subtests.test("mCC_yr"):
        assert actual_data.mCC_yr == approx(694918.997854312)

    with subtests.test("mCC_yr_MaxPwr"):
        assert actual_data.mCC_yr_MaxPwr == approx(1455737.3654722548)


def test_electrodialysis_costs(power_profile, subtests):
    co2_outputs = echem_mcc.co2_purification(
        ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3)
    )

    ranges: echem_mcc.ElectrodialysisRangeOutputs = (
        echem_mcc.initialize_power_chemical_ranges(
            ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3),
            pump_config=echem_mcc.PumpInputs(),
            seawater_config=echem_mcc.SeaWaterInputs(),
            co2_config=co2_outputs,
        )
    )
    simulation: echem_mcc.ElectrodialysisOutputs = echem_mcc.simulate_electrodialysis(
        ranges=ranges,
        ed_config=echem_mcc.ElectrodialysisInputs(N_edMax=3),
        power_profile=power_profile,
        initial_tank_volume_m3=0,
    )

    costs: echem_mcc.ElectrodialysisCostOutputs = echem_mcc.electrodialysis_cost_model(
        echem_mcc.ElectrodialysisCostInputs(
            electrodialysis_inputs=echem_mcc.ElectrodialysisInputs(),
            mCC_yr=simulation.mCC_yr,
            max_theoretical_mCC=max(ranges.S1["mCC"]),
            total_tank_volume=ranges.V_aT_max+ranges.V_bT_max,
            infrastructure_type="swCool",
        )
    )

    costs = asdict(costs)

    expected_data = {
        "initial_capital_cost": 1643937996.5385807,
        "yearly_capital_cost": 129506915.7774168,
        "yearly_operational_cost": 182763696.43568406,
        "initial_bop_capital_cost": 1497792942.1620674,
        "yearly_bop_capital_cost": 117933003.69014753,
        "yearly_bop_operational_cost": 158441531.51078314,
        "initial_ed_capital_cost": 146145054.3765131,
        "yearly_ed_capital_cost": 11573912.087269258,
        "yearly_ed_operational_cost": 24322164.92490092,
        "initial_tank_capital_cost": 8640000.0,
        "yearly_tank_cost": 684242.1104198317
    }

    assert_dict_equal(costs, expected_data, subtests)
