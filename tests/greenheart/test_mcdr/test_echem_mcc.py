import math
import numpy as np
from attrs import asdict
from pytest import approx, fixture, raises
from greenheart.simulation.technologies.mcdr import echem_mcc
from greenheart.simulation.technologies.mcdr.echem_mcc import Pump


@fixture
def power_profile():
    exTime = 8760  # Number of time steps (typically hours in a year)
    maxPwr = 500 * 10**6  # Maximum power in watts
    Amp = maxPwr / 2  # Amplitude
    periodT = 24  # Period of the sine wave (e.g., 24 hours)
    movUp = Amp  # Vertical shift
    movSide = -1 * math.pi / 2  # Phase shift
    exPwr = np.zeros(exTime)  # Initialize the power array

    # Corrected loop: Use range and sine wave computation
    for i in range(exTime):
        exPwr[i] = Amp * math.sin(2 * math.pi / periodT * i + movSide) + movUp
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
                    actual_value == expected_value
                ), f"Mismatch found in key '{key}': {actual_value} != {expected_value}"


def test_power_chemical_ranges(subtests):
    # Run the function to get actual outputs
    actual_data: echem_mcc.ElectrodialysisRangeOutputs = (
        echem_mcc.initialize_power_chemical_ranges(
            ed_config=echem_mcc.ElectrodialysisInputs(),
            pump_config=echem_mcc.PumpInputs(),
            seawater_config=echem_mcc.SeaWaterInputs(),
        )
    )

    actual_data = asdict(actual_data)

    # Define expected outputs
    expected_data = {
        "S1": {
            "volAcid": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "volBase": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "mCC": np.array(
                [31.09742798, 62.19485597, 93.29228395, 124.38971194, 155.48713992]
            ),
            "pH_f": np.array(
                [10.58191318, 10.58191318, 10.58191318, 10.58191318, 10.58191318]
            ),
            "dic_f": np.array(
                [0.00022736, 0.00022736, 0.00022736, 0.00022736, 0.00022736]
            ),
            "c_a": np.array(
                [0.55555554, 0.55555554, 0.55555554, 0.55555554, 0.55555554]
            ),
            "c_b": np.array(
                [0.55554038, 0.55554038, 0.55554038, 0.55554038, 0.55554038]
            ),
            "Qin": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            "Qout": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            "pwrRanges": np.array(
                [
                    6.75585262e07,
                    1.43457761e08,
                    2.27697704e08,
                    3.20278356e08,
                    4.21197289e08,
                ]
            ),
        },
        "S2": {
            "volAcid": np.array(
                [
                    0.0,
                    1800.0,
                    3600.0,
                    5400.0,
                    7200.0,
                    0.0,
                    1800.0,
                    3600.0,
                    5400.0,
                    0.0,
                    1800.0,
                    3600.0,
                    0.0,
                    1800.0,
                    0.0,
                ]
            ),
            "volBase": np.array(
                [
                    0.0,
                    1800.0,
                    3600.0,
                    5400.0,
                    7200.0,
                    0.0,
                    1800.0,
                    3600.0,
                    5400.0,
                    0.0,
                    1800.0,
                    3600.0,
                    0.0,
                    1800.0,
                    0.0,
                ]
            ),
            "mCC": np.array(
                [
                    31.09742798,
                    31.09742798,
                    31.09742798,
                    31.09742798,
                    31.09742798,
                    62.19485597,
                    62.19485597,
                    62.19485597,
                    62.19485597,
                    93.29228395,
                    93.29228395,
                    93.29228395,
                    124.38971194,
                    124.38971194,
                    155.48713992,
                ]
            ),
            "pH_f": np.array(
                [
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                    10.58191318,
                ]
            ),
            "dic_f": np.array(
                [
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                    0.00022736,
                ]
            ),
            "c_a": np.array(
                [
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                    0.55555554,
                ]
            ),
            "c_b": np.array(
                [
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                    0.55554038,
                ]
            ),
            "Qin": np.array(
                [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    200.0,
                    201.0,
                    202.0,
                    203.0,
                    300.0,
                    301.0,
                    302.0,
                    400.0,
                    401.0,
                    500.0,
                ]
            ),
            "Qout": np.array(
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    200.0,
                    200.0,
                    200.0,
                    200.0,
                    300.0,
                    300.0,
                    300.0,
                    400.0,
                    400.0,
                    500.0,
                ]
            ),
            "pwrRanges": np.array(
                [
                    6.72903241e07,
                    1.17630828e08,
                    1.68027512e08,
                    2.18480376e08,
                    2.68989420e08,
                    1.43085762e08,
                    1.93551094e08,
                    2.44072605e08,
                    2.94650297e08,
                    2.27351448e08,
                    2.77941607e08,
                    3.28587946e08,
                    3.20061724e08,
                    3.70776711e08,
                    4.21197289e08,
                ]
            ),
        },
        "S3": {
            "volAcid": np.array([-1800.0, -3600.0, -5400.0, -7200.0, -9000.0]),
            "volBase": np.array([-1800.0, -3600.0, -5400.0, -7200.0, -9000.0]),
            "mCC": np.array(
                [31.4004411, 62.8008822, 94.20132329, 125.60176439, 157.00220549]
            ),
            "pH_f": np.array(
                [10.57597143, 10.57597143, 10.57597143, 10.57597143, 10.57597143]
            ),
            "dic_f": np.array(
                [0.00022796, 0.00022796, 0.00022796, 0.00022796, 0.00022796]
            ),
            "c_a": np.array(
                [0.55555554, 0.55555554, 0.55555554, 0.55555554, 0.55555554]
            ),
            "c_b": np.array(
                [0.55554038, 0.55554038, 0.55554038, 0.55554038, 0.55554038]
            ),
            "Qin": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            "Qout": np.array([101.0, 202.0, 303.0, 404.0, 505.0]),
            "pwrRanges": np.array(
                [
                    1.75822380e07,
                    4.35325488e07,
                    7.78509324e07,
                    1.20537389e08,
                    1.71591918e08,
                ]
            ),
        },
        "S4": {
            "volAcid": np.array([1800.0, 3600.0, 5400.0, 7200.0, 9000.0]),
            "volBase": np.array([1800.0, 3600.0, 5400.0, 7200.0, 9000.0]),
            "mCC": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "pH_f": np.array([8.1, 8.1, 8.1, 8.1, 8.1]),
            "dic_f": np.array([0.0022, 0.0022, 0.0022, 0.0022, 0.0022]),
            "c_a": np.array(
                [0.55555554, 0.55555554, 0.55555554, 0.55555554, 0.55555554]
            ),
            "c_b": np.array(
                [0.55554038, 0.55554038, 0.55554038, 0.55554038, 0.55554038]
            ),
            "Qin": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "Qout": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "pwrRanges": np.array(
                [
                    5.01666667e07,
                    1.00416667e08,
                    1.50750000e08,
                    2.01166667e08,
                    2.51666667e08,
                ]
            ),
        },
        "S5": {
            "volAcid": 0,
            "volBase": 0,
            "mCC": 0,
            "pH_f": 8.1,
            "dic_f": 0.0022,
            "c_a": 7.943282347242822e-09,
            "c_b": 7.58577575029182e-06,
            "Qin": 0,
            "Qout": 0,
            "pwrRanges": np.array([0.0]),
        },
        "P_minS1_tot": 66749999.99999999,
        "P_minS2_tot": 67290324.14662872,
        "P_minS3_tot": 17582238.016953908,
        "P_minS4_tot": 50166666.666666664,
        "V_aT_max": 10800.0,
        "V_bT_max": 10800.0,
        "V_a3_min": 1800.0,
        "V_b3_min": 1800.0,
        "N_range": 5,
        "S2_tot_range": 15,
        "S2_ranges": np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [2.0, 3.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [4.0, 0.0],
                [4.0, 1.0],
                [5.0, 0.0],
            ]
        ),
        "pump_power_min": 16.75,
        "pump_power_max": 168.05555555555551,
        "vacuum_power_min": 0.05403241466287295,
        "vacuum_power_max": 0.40919179590458093,
    }

    assert_dict_equal(actual_data, expected_data, subtests)


def test_simulate_electrodialysis(power_profile, subtests):
    ranges: echem_mcc.ElectrodialysisRangeOutputs = (
        echem_mcc.initialize_power_chemical_ranges(
            ed_config=echem_mcc.ElectrodialysisInputs(),
            pump_config=echem_mcc.PumpInputs(),
            seawater_config=echem_mcc.SeaWaterInputs(),
        )
    )

    actual_data: echem_mcc.ElectrodialysisOutputs = echem_mcc.simulate_electrodialysis(
        ranges=ranges,
        ed_config=echem_mcc.ElectrodialysisInputs(),
        power_profile=power_profile,
        initial_tank_volume_m3=0,
    )

    with subtests.test("ED_outputs: N_ed"):
        sum(actual_data.ED_outputs["N_ed"]) == approx(24091.0, 1e-3)

    with subtests.test("ED_outputs: P_xs"):
        sum(actual_data.ED_outputs["P_xs"]) == approx(322629167959.93, 1e-3)

    with subtests.test("ED_outputs: volAcid"):
        sum(actual_data.ED_outputs["volAcid"]) == approx(9000.0, 1e-3)

    with subtests.test("ED_outputs: volBase"):
        sum(actual_data.ED_outputs["volBase"]) == approx(9000.0, 1e-3)

    with subtests.test("ED_outputs: mCC"):
        sum(actual_data.ED_outputs["mCC"]) == approx(726563.522, 1e-3)

    with subtests.test("ED_outputs: pH_f"):
        sum(actual_data.ED_outputs["pH_f"]) == approx(89973.0511, 1e-3)

    with subtests.test("ED_outputs: dic_f"):
        sum(actual_data.ED_outputs["dic_f"]) == approx(4.154, 1e-3)

    with subtests.test("ED_outputs: c_a"):
        sum(actual_data.ED_outputs["c_a"]) == approx(4257.77, 1e-3)

    with subtests.test("ED_outputs: c_b"):
        sum(actual_data.ED_outputs["c_b"]) == approx(4257.67, 1e-3)

    with subtests.test("ED_outputs: Qout"):
        sum(actual_data.ED_outputs["Qout"]) == approx(2336429.0, 1e-3)

    with subtests.test("capacity_factor"):
        actual_data.capacity_factor == approx(0.8748, 1e-3)

    with subtests.test("mCC_yr"):
        actual_data.mCC_yr == approx(726563.522, 1e-3)

    with subtests.test("mCC_yr_MaxPwr"):
        actual_data.mCC_yr_MaxPwr == approx(1362067.34, 1e-3)


def test_electrodialysis_costs(power_profile, subtests):
    ranges: echem_mcc.ElectrodialysisRangeOutputs = (
        echem_mcc.initialize_power_chemical_ranges(
            ed_config=echem_mcc.ElectrodialysisInputs(),
            pump_config=echem_mcc.PumpInputs(),
            seawater_config=echem_mcc.SeaWaterInputs(),
        )
    )

    simulation: echem_mcc.ElectrodialysisOutputs = echem_mcc.simulate_electrodialysis(
        ranges=ranges,
        ed_config=echem_mcc.ElectrodialysisInputs(),
        power_profile=power_profile,
        initial_tank_volume_m3=0,
    )

    costs: echem_mcc.ElectrodialysisCostOutputs = echem_mcc.electrodialysis_cost_model(
        echem_mcc.ElectrodialysisCostInputs(
            electrodialysis_inputs=echem_mcc.ElectrodialysisInputs(),
            mCC_yr=simulation.mCC_yr,
            max_theoretical_mCC=max(ranges.S1["mCC"]),
        )
    )

    costs = asdict(costs)

    expected_data = {
        "initial_capital_cost": 7086199204.557611,
        "yearly_capital_cost": 561189340.1136382,
        "yearly_operational_cost": 385078666.65331024,
        "initial_bop_capital_cost": 6847445862.085994,
        "yearly_bop_capital_cost": 542281343.4790816,
        "yearly_bop_operational_cost": 369820832.6915753,
        "initial_ed_capital_cost": 233978275.62218532,
        "yearly_ed_capital_cost": 18529836.701865412,
        "yearly_ed_operational_cost": 15257833.961734934,
    }

    assert_dict_equal(costs, expected_data, subtests)
