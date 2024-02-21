import os
import pathlib
from hopp.simulation.hopp_interface import HoppInterface
from hopp.simulation.hybrid_simulation import HybridSimulation
import json
from hopp.tools.analysis import create_cost_calculator
import pandas as pd
import numpy as np

def hopp_for_h2(
    site,
    hopp_config, 
    scenario, 
    load, 
    custom_powercurve, 
    # these will be moved to cost_info
    wind_om_cost_kw,
    solar_om_cost_kw
):
    """
    Runs HOPP for H2 analysis purposes
    :param scenario: ``dict``,
        Dictionary of scenario options, includes location, year, technology pricing
    :param load: ``list``,
        (8760) hourly load profile of electrolyzer in kW. Default is continuous load at kw_continuous rating
    :param custom_powercurve: ``bool``,
        Flag to determine if custom wind turbine powercurve file is loaded

    :returns:

    :param hybrid_plant: :class: `hybrid.hybrid_simulation.HybridSimulation`,
        Base class for simulation a Hybrid Plant
    :param combined_pv_wind_power_production_hopp: ``list``,
        (8760x1) hourly sequence of combined pv and wind power in kW
    :param combined_pv_wind_curtailment_hopp: ``list``,
        (8760x1) hourly sequence of combined pv and wind curtailment/spilled energy in kW
    :param energy_shortfall_hopp: ``list``,
        (8760x1) hourly sequence of energy shortfall vs. load in kW
    :param annual_energies: ``dict``,
        Dictionary of AEP for each technology
    :param wind_plus_solar_npv: ``float``,
        Combined Net present value of wind + solar technologies
    :param npvs: ``dict``,
        Dictionary of net present values of technologies
    :param lcoe: ``float``
        Levelized cost of electricity for hybrid plant
    """

    hopp_config["site"] = site
    hi = HoppInterface(hopp_config)
    hybrid_plant = hi.system
    technologies = hopp_config["technologies"]

    hybrid_plant.set_om_costs_per_kw(pv_om_per_kw=solar_om_cost_kw, wind_om_per_kw=wind_om_cost_kw, hybrid_om_per_kw=None)
    if "pv" in technologies:
        solar_size_mw = technologies["pv"]["system_capacity_kw"] / 1000

        if solar_size_mw > 0:
            hybrid_plant.pv._financial_model.FinancialParameters.analysis_period = scenario["Useful Life"]
            hybrid_plant.pv._financial_model.FinancialParameters.debt_percent = scenario["Debt Equity"]
            hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
            # if scenario["ITC Available"]:
            #     hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 26
            # else:
            #     hybrid_plant.pv._financial_model.TaxCreditIncentives.itc_fed_percent = 0

    if "wind" in technologies:
        turbine_rating_kw = technologies["wind"]["turbine_rating_kw"]
        num_turbines = technologies["wind"]["num_turbines"]
        wind_size_mw = turbine_rating_kw * num_turbines / 1000

        hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.33
        hybrid_plant.wind.wake_model = 3
        hybrid_plant.wind.value("wake_int_loss", 3)
        hybrid_plant.wind._financial_model.FinancialParameters.analysis_period = scenario["Useful Life"]
        hybrid_plant.wind._financial_model.FinancialParameters.system_capacity = wind_size_mw * 1000
        # hybrid_plant.wind.om_capacity =
        hybrid_plant.wind._financial_model.FinancialParameters.debt_percent = scenario["Debt Equity"]
        hybrid_plant.wind._financial_model.value("debt_option", 0)
        hybrid_plant.wind._financial_model.FinancialParameters.debt_percent = scenario["Debt Equity"]
        hybrid_plant.wind._financial_model.value("debt_option", 0)
        print(scenario.keys())
        ptc_val = scenario["Wind PTC"]# hybrid_plant.wind.om_capacity =
        hybrid_plant.wind._financial_model.FinancialParameters.debt_percent = scenario["Debt Equity"]
        hybrid_plant.wind._financial_model.value("debt_option", 0)
        ptc_val = scenario["Wind PTC"]

        interim_list = list(
            hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount)
        interim_list[0] = ptc_val
        hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = tuple(interim_list)
        hybrid_plant.wind._system_model.Turbine.wind_turbine_hub_ht = scenario["Tower Height"]

        # import ipdb; ipdb.set_trace()
        hybrid_plant.wind._financial_model.TaxCreditIncentives.itc_fed_percent = [scenario["Wind ITC"]]
        hybrid_plant.wind._financial_model.FinancialParameters.real_discount_rate = 7
        if custom_powercurve:
            parent_path = os.path.abspath(os.path.dirname(__file__))
            powercurve_file = open(os.path.join(parent_path, scenario["Powercurve File"]))
            powercurve_file_extension = pathlib.Path(os.path.join(parent_path, scenario["Powercurve File"])).suffix
            if powercurve_file_extension == ".csv":
                curve_data = pd.read_csv(os.path.join(parent_path, scenario["Powercurve File"]))
                wind_speed = curve_data["Wind Speed [m/s]"].values.tolist()
                curve_power = curve_data["Power [kW]"]
                hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
                hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power

            else:
                powercurve_data = json.load(powercurve_file)
                powercurve_file.close()
                hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = \
                    powercurve_data["turbine_powercurve_specification"]["wind_speed_ms"]
                hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = \
                    powercurve_data["turbine_powercurve_specification"]["turbine_power_output"]

            hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)

    hybrid_plant.simulate(scenario["Useful Life"])

    # HOPP Specific Energy Metrics
    combined_pv_wind_power_production_hopp = hybrid_plant.grid._system_model.Outputs.system_pre_interconnect_kwac[0:8760]
    energy_shortfall_hopp = [x - y for x, y in
                             zip(load,combined_pv_wind_power_production_hopp)]
    energy_shortfall_hopp = [x if x > 0 else 0 for x in energy_shortfall_hopp]
    combined_pv_wind_curtailment_hopp = [x - y for x, y in
                             zip(combined_pv_wind_power_production_hopp,load)]
    combined_pv_wind_curtailment_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_hopp]

    # super simple dispatch battery model with no forecasting TODO: add forecasting
    # print("Length of "energy_shortfall_hopp is {}".format(len(energy_shortfall_hopp)))
    # print("Length of "combined_pv_wind_curtailment_hopp is {}".format(len(combined_pv_wind_curtailment_hopp)))
    # TODO: Fix bug in dispatch model that errors when first curtailment >0
    combined_pv_wind_curtailment_hopp[0] = 0
    #wind_plant_size_check = hybrid_plant.wind.system_capacity_kw
    # Save the outputs
    annual_energies = hybrid_plant.annual_energies
    if "wind" in technologies and "solar" in technologies:
        wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.pv
    elif "wind" not in technologies and "solar" in technologies:
        wind_plus_solar_npv = hybrid_plant.net_present_values.pv
    elif "solar" not in technologies and "wind" in technologies:
        wind_plus_solar_npv = hybrid_plant.net_present_values.wind
    else:
        wind_plus_solar_npv = 0
    npvs = hybrid_plant.net_present_values
    lcoe = hybrid_plant.lcoe_real.hybrid
    lcoe_nom = hybrid_plant.lcoe_nom.hybrid
    # print("discount rate", hybrid_plant.wind._financial_model.FinancialParameters.real_discount_rate)

    return hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp, \
           energy_shortfall_hopp,\
           annual_energies, wind_plus_solar_npv, npvs, lcoe, lcoe_nom