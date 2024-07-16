import os
import pandas as pd
import numpy_financial as npf
import numpy as np
import numpy as np
import openmdao.api as om
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pytest import approx
import copy
import os

import datetime as dt
from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml
from greenheart.tools.eco.utilities import ceildiv
from greenheart.simulation.technologies.hydrogen.electrolysis.run_h2_PEM import (
    run_h2_PEM,
)
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import (
    PEM_H2_Clusters as PEMClusters,
)
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_costs_Singlitico_model import (
    PEMCostsSingliticoModel,
)

from greenheart.simulation.technologies.hydrogen.h2_storage.storage_sizing import (
    hydrogen_storage_capacity,
)

from greenheart.simulation.technologies.hydrogen.h2_storage.salt_cavern.salt_cavern import (
    SaltCavernStorage,
)

import ProFAST

# load hopp config
hopp_config = load_yaml("./input/plant/texas-hopp-config-6MW.yaml")

# load greenheart config
greenheart_config = load_yaml("./input/plant/greenheart_config_onshore_tx.yaml")

# Override OM costs
hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][0] = (
    hopp_config["config"]["cost_info"]["wind_om_per_kw"]
)
# hopp_config['technologies']['wind']['fin_model']['system_costs']['om_variable'] = 0.015 #$/kWh


hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][0] = (
    hopp_config["config"]["cost_info"]["pv_om_per_kw"]
)

hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
    "om_batt_fixed_cost"
] = hopp_config["config"]["cost_info"]["battery_om_per_kw"]

hopp_config["site"]["desired_schedule"] = [
    362.560118387 # 1/3 the 
] * 8760  # reduced demand size by 3 [1087.68035516] * 8760 #rating of electrolyzer

# load hybrid system
hi = HoppInterface(hopp_config)
hi.system.wind.om_variable = 0.15


# update power curve to Vestas V90 2MW (reference SAM 2023.12.17)
# curve_data = pd.read_csv("./input/turbines/vestas-2mw.csv")
curve_data = pd.read_csv("./input/turbines/2023NREL_Bespoke_6MW_170.csv")
wind_speed = curve_data["Wind Speed [m/s]"].values.tolist()
curve_power = curve_data["Power [kW]"]
hi.system.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
hi.system.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power

# update solar tilt
# hi.system.pv._system_model.SystemDesign.tilt = 23.1 #based on latitude
hi.system.pv._system_model.SystemDesign.array_type = 1  # fixed axis

hi.simulate(project_life=20)

# store results for later use
hopp_results = {
    "hopp_interface": hi,
    "hybrid_plant": hi.system,
    "combined_hybrid_power_production_hopp": [value * 3 for value in hi.system.grid._system_model.Outputs.system_pre_interconnect_kwac[0:8760]],
    "combined_hybrid_curtailment_hopp": [value *3 for value in hi.system.grid.generation_curtailed],
    "energy_shortfall_hopp": [value *3 for value in hi.system.grid.missed_load],
    "annual_energies": hi.system.annual_energies,
}
print(len(hopp_results["energy_shortfall_hopp"]))
print("\nHOPP Results")
print("Hybrid Annual Energy: ", hopp_results["annual_energies"])
print("Capacity factors: ", hi.system.capacity_factors)
print("Real LCOE from HOPP: ", hi.system.lcoe_real)
print("Missed Load", sum(hopp_results["energy_shortfall_hopp"]))


electrolyzer_size_mw = greenheart_config["electrolyzer"]["rating"]
electrolyzer_capex_kw = greenheart_config["electrolyzer"]["electrolyzer_capex"]

elec_power = []
curtailed = []

for x in hopp_results['combined_hybrid_power_production_hopp'][0:8760]:
    if x < greenheart_config['electrolyzer']['cluster_rating_MW'] * 0.1 * 1E3:
        elec_power.append(0)
        curtailed.append(x)
    elif x <= greenheart_config['electrolyzer']['rating'] * 1E3:
        elec_power.append(x)
        curtailed.append(0)
    else:
        elec_power.append(greenheart_config['electrolyzer']['rating'] * 1E3)
        curtailed.append(x - greenheart_config['electrolyzer']['rating'] * 1E3)

sum_power = sum(elec_power)
basic_h2_conversion = sum_power * 55.49
print("Basic H2 conversion", basic_h2_conversion)
print("Curtailment, ", sum(curtailed))




# IF GRID CONNECTED
if greenheart_config["project_parameters"]["grid_connection"]:
    # NOTE: if grid-connected, it assumes that hydrogen demand is input and there is not
    # multi-cluster control strategies. This capability exists at the cluster level, not at the
    # system level.
    if greenheart_config["electrolyzer"]["sizing"]["hydrogen_dmd"] is not None:
        grid_connection_scenario = "grid-only"
        hydrogen_production_capacity_required_kgphr = greenheart_config["electrolyzer"][
            "sizing"
        ]["hydrogen_dmd"]
        energy_to_electrolyzer_kw = []
    else:
        grid_connection_scenario = "off-grid"
        hydrogen_production_capacity_required_kgphr = []
        energy_to_electrolyzer_kw = np.ones(8760) * electrolyzer_size_mw * 1e3
# IF NOT GRID CONNECTED
else:
    hydrogen_production_capacity_required_kgphr = []
    grid_connection_scenario = "off-grid"
    energy_to_electrolyzer_kw = np.asarray(
        hopp_results["combined_hybrid_power_production_hopp"]
    )

n_pem_clusters = int(
    ceildiv(
        electrolyzer_size_mw, greenheart_config["electrolyzer"]["cluster_rating_MW"]
    )
)

## run using greensteel model
pem_param_dict = {
    "eol_eff_percent_loss": greenheart_config["electrolyzer"]["eol_eff_percent_loss"],
    "uptime_hours_until_eol": greenheart_config["electrolyzer"][
        "uptime_hours_until_eol"
    ],
    "include_degradation_penalty": greenheart_config["electrolyzer"][
        "include_degradation_penalty"
    ],
    "turndown_ratio": greenheart_config["electrolyzer"]["turndown_ratio"],
}

H2_Results, h2_ts, h2_tot, power_to_electrolyzer_kw = run_h2_PEM(
    electrical_generation_timeseries=energy_to_electrolyzer_kw,
    electrolyzer_size=electrolyzer_size_mw,
    useful_life=greenheart_config["project_parameters"][
        "project_lifetime"
    ],  # EG: should be in years for full plant life - only used in financial model
    n_pem_clusters=n_pem_clusters,
    pem_control_type=greenheart_config["electrolyzer"]["pem_control_type"],
    electrolyzer_direct_cost_kw=electrolyzer_capex_kw,
    user_defined_pem_param_dictionary=pem_param_dict,
    grid_connection_scenario=grid_connection_scenario,  # if not offgrid, assumes steady h2 demand in kgphr for full year
    hydrogen_production_capacity_required_kgphr=hydrogen_production_capacity_required_kgphr,
    debug_mode=False,
    verbose=True,
)

electrolyzer_physics_results = {
    "H2_Results": H2_Results,
    "capacity_factor": H2_Results["Life: Capacity Factor"],
    "power_to_electrolyzer_kw": power_to_electrolyzer_kw,
}

print("Electrolyzer Capacity Factor", electrolyzer_physics_results['capacity_factor'])
print("Annual H2 production (kg)", H2_Results['Life: Annual H2 production [kg/year]'])

min_usable_power_kW = greenheart_config["electrolyzer"]["turndown_ratio"]*greenheart_config["electrolyzer"]["cluster_rating_MW"]*1e3
max_usable_power_kW = electrolyzer_size_mw*1e3

usable_power_kW = np.where(energy_to_electrolyzer_kw>max_usable_power_kW,max_usable_power_kW,energy_to_electrolyzer_kw)
usable_power_kW = np.where(usable_power_kW<min_usable_power_kW,0,usable_power_kW)

print(len(hopp_results["combined_hybrid_power_production_hopp"]))

a_lte = 0.01802 #kg/kWh
h2_production_kg_linear = usable_power_kW/a_lte
h2_production_kg_linear_tot = np.sum(h2_production_kg_linear)
print("Linear total: ", h2_production_kg_linear_tot)


# unpack inputs
H2_Results = electrolyzer_physics_results["H2_Results"]
electrolyzer_size_mw = greenheart_config["electrolyzer"]["rating"]
useful_life = greenheart_config["project_parameters"]["project_lifetime"]
atb_year = greenheart_config["project_parameters"]["atb_year"]
electrical_generation_timeseries = electrolyzer_physics_results[
    "power_to_electrolyzer_kw"
]
nturbines = hopp_config["technologies"]["wind"]["num_turbines"]

electrolyzer_cost_model = greenheart_config["electrolyzer"][
    "cost_model"
]  # can be "basic" or "singlitico2021"


offshore = 0

P_elec = electrolyzer_size_mw * 1e-3  # [GW]
RC_elec = greenheart_config["electrolyzer"]["electrolyzer_capex"]  # [USD/kW]

pem_offshore = PEMCostsSingliticoModel(elec_location=offshore)

(
    electrolyzer_capital_cost_musd,
    electrolyzer_om_cost_musd,
) = pem_offshore.run(P_elec, RC_elec)

electrolyzer_total_capital_cost = (
    electrolyzer_capital_cost_musd * 1e6
)  # convert from M USD to USD
electrolyzer_OM_cost = electrolyzer_om_cost_musd * 1e6  # convert from M USD to USD

# package outputs for return
electrolyzer_cost_results = {
    "electrolyzer_total_capital_cost": electrolyzer_total_capital_cost,
    "electrolyzer_OM_cost_annual": electrolyzer_OM_cost,
}

hydrogen_storage_demand = np.mean(
    electrolyzer_physics_results["H2_Results"]["Hydrogen Hourly Production [kg/hr]"]
)  # TODO: update demand based on end-use needs

h2_storage_results = dict()

storage_max_fill_rate = np.max(
    electrolyzer_physics_results["H2_Results"]["Hydrogen Hourly Production [kg/hr]"]
)
(
    hydrogen_storage_capacity_kg,
    hydrogen_storage_duration_hr,
    hydrogen_storage_soc,
) = hydrogen_storage_capacity(
    electrolyzer_physics_results["H2_Results"],
    greenheart_config["electrolyzer"]["rating"],
    hydrogen_storage_demand,
)
h2_storage_capacity_kg = hydrogen_storage_capacity_kg
h2_storage_results["hydrogen_storage_duration_hr"] = hydrogen_storage_duration_hr
h2_storage_results["hydrogen_storage_soc"] = hydrogen_storage_soc

print("H2 storage capacity",h2_storage_capacity_kg)



df_data = pd.DataFrame.from_dict(h2_storage_results)
df_data['h2 production hourly [kg]'] = H2_Results[
                        "Hydrogen Hourly Production [kg/hr]"
                    ]

# set start and end dates
dt_start = dt.datetime(2024, 1, 1, 0)
dt_end = dt.datetime(2024, 1, 9, 0)
def get_hour_from_datetime(dt_start: dt.datetime, dt_end: dt.datetime):

    dt_beginning_of_year = dt.datetime(dt_start.year, 1, 1, tzinfo=dt_start.tzinfo)

    hour_start = int((dt_start - dt_beginning_of_year).total_seconds() // 3600)
    hour_end = int((dt_end - dt_beginning_of_year).total_seconds() // 3600)

    return hour_start, hour_end

hour_start, hour_end = get_hour_from_datetime(dt_start, dt_end)
print("HR STRT", hour_start, "HR END",hour_end)

# get hydrogen demand
df_h_out_demand = df_data[["h2 production hourly [kg]"]]*1E-3
h2_demand = df_h_out_demand.mean().values[0]

df_data = df_data.iloc[hour_start:hour_end]

# set up plots
fig, ax = plt.subplots(3,1, sharex=True, figsize=(10,6))

# plot hydrogen production
df_h_out = df_data[["h2 production hourly [kg]"]]*1E-3

ax[0].plot(df_h_out)
ax[0].set(ylabel="Hydrogen (t)", xlabel="Hour", title="Hydrogen Production")

# plot storage SOC
df_h_soc = np.array(pd.DataFrame(df_data[["hydrogen_storage_soc"]])*1E-3)
df_h_soc_change = np.array([(df_h_soc[i] - df_h_soc[i-1]) for i in np.arange(1, len(df_h_soc))]).flatten()
initial_value = (df_h_out.iloc[0] - h2_demand) #df_h_soc_change[0]/2 #h2_demand - df_h_out.iloc[0]
# import pdb; pdb.set_trace()
df_h_soc_change = np.insert(df_h_soc_change, 0, initial_value, axis=0)
df_hour = np.arange(0, len(df_h_soc))
df_h_charge_val = df_h_soc_change[df_h_soc_change > 0]
df_h_charge_hour = df_hour[df_h_soc_change > 0]
df_h_discharge_val = df_h_soc_change[df_h_soc_change < 0]
df_h_discharge_hour= df_hour[df_h_soc_change < 0]

ax[1].plot(df_h_soc, label="state of charge")
ax[1].scatter(df_h_charge_hour, df_h_charge_val, label="Charging", s=1)
ax[1].scatter(df_h_discharge_hour, df_h_discharge_val, label="Dis-charging", s=1)
ax[1].set(ylabel="Hydrogen (t)", xlabel="Hour", title="Hydrogen Storage Flows")
ax[1].legend(frameon=False)

# plot net h2 available
net_flow = np.array(df_h_out).flatten() - np.array(df_h_soc_change)
print("Net Flow",net_flow[0])
ax[2].plot(net_flow, label="Output with storage")
ax[2].plot(df_h_out, "--", label="Original output")
# import pdb; pdb.set_trace()
ax[2].axhline(h2_demand, linestyle=":", label="Demand", color="k")
ax[2].set(ylabel="Hydrogen (t)", xlabel="Hour", title="Net Hydrogen Dispatch")
ax[2].legend(frameon=False)

# fig.add_axes((0, 0, 1, 0.5))
plt.tight_layout()

# plt.savefig(data_path+"h2_dispatch.pdf", transparent=True)
plt.show()

# plt.savefig(data_path+"/figures/production/hydrogen_flows.png")

# # initialize dictionary for salt cavern storage parameters
# storage_input = dict()

# # pull parameters from plant_config file
# storage_input["h2_storage_kg"] = h2_storage_capacity_kg
# storage_input["system_flow_rate"] = storage_max_fill_rate
# storage_input["model"] = "papadias"

# # run salt cavern storage model
# h2_storage = SaltCavernStorage(storage_input)

# h2_storage.salt_cavern_capex()
# h2_storage.salt_cavern_opex()

# h2_storage_results["storage_capex"] = h2_storage.output_dict[
#     "salt_cavern_storage_capex"
# ]
# h2_storage_results["storage_opex"] = h2_storage.output_dict[
#     "salt_cavern_storage_opex"
# ]
# h2_storage_results["storage_energy"] = 0.0

from greenheart.simulation.technologies.hydrogen.h2_storage.lined_rock_cavern.lined_rock_cavern import (
    LinedRockCavernStorage,
)

# initialize dictionary for salt cavern storage parameters
storage_input = dict()

# pull parameters from plat_config file
storage_input["h2_storage_kg"] = h2_storage_capacity_kg
storage_input["system_flow_rate"] = storage_max_fill_rate
storage_input["model"] = "papadias"

# run salt cavern storage model
h2_storage = LinedRockCavernStorage(storage_input)

h2_storage.lined_rock_cavern_capex()
h2_storage.lined_rock_cavern_opex()

h2_storage_results["storage_capex"] = h2_storage.output_dict[
    "lined_rock_cavern_storage_capex"
]
h2_storage_results["storage_opex"] = h2_storage.output_dict[
    "lined_rock_cavern_storage_opex"
]
h2_storage_results["storage_energy"] = 0.0


def run_capex(
    hopp_results,
    storage_input,
    electrolyzer_cost_results,
    h2_storage_results,
    hopp_config,
    greenheart_config,
    verbose=False,
):

    # total_wind_cost_no_export, total_used_export_system_costs = breakout_export_costs_from_orbit_results(orbit_project, greenheart_config, design_scenario)

    # if orbit_hybrid_electrical_export_project is not None:
    #     _, total_used_export_system_costs = breakout_export_costs_from_orbit_results(orbit_hybrid_electrical_export_project, greenheart_config, design_scenario)

    # wave capex
    if hopp_config["site"]["wave"]:
        cost_dict = hopp_results["hybrid_plant"].wave.mhk_costs.cost_outputs

        wcapex = (
            cost_dict["structural_assembly_cost_modeled"]
            + cost_dict["power_takeoff_system_cost_modeled"]
            + cost_dict["mooring_found_substruc_cost_modeled"]
        )
        wbos = (
            cost_dict["development_cost_modeled"]
            + cost_dict["eng_and_mgmt_cost_modeled"]
            + cost_dict["plant_commissioning_cost_modeled"]
            + cost_dict["site_access_port_staging_cost_modeled"]
            + cost_dict["assembly_and_install_cost_modeled"]
            + cost_dict["other_infrastructure_cost_modeled"]
        )
        welec_infrastruc_costs = (
            cost_dict["array_cable_system_cost_modeled"]
            + cost_dict["export_cable_system_cost_modeled"]
            + cost_dict["other_elec_infra_cost_modeled"]
        )  # +\
        # cost_dict['onshore_substation_cost_modeled']+\
        # cost_dict['offshore_substation_cost_modeled']
        # financial = cost_dict['project_contingency']+\
        # cost_dict['insurance_during_construction']+\
        # cost_dict['reserve_accounts']
        wave_capex = wcapex + wbos + welec_infrastruc_costs
    else:
        wave_capex = 0.0

    # wind capex
    if "wind" in hopp_config["technologies"].keys():
        wind_capex = hopp_results["hybrid_plant"].wind.total_installed_cost * 3
        print("Wind capex", wind_capex)
    else:
        wind_capex = 0.0

    # solar capex
    if "pv" in hopp_config["technologies"].keys():
        solar_capex = hopp_results["hybrid_plant"].pv.total_installed_cost * 3
    else:
        solar_capex = 0.0

    # battery capex
    if "battery" in hopp_config["technologies"].keys():
        battery_capex = hopp_results["hybrid_plant"].battery.total_installed_cost * 3
    else:
        battery_capex = 0.0

    ## electrolyzer capex
    # electrolyzer_total_capital_cost = electrolyzer_cost_results[
    #     "electrolyzer_total_capital_cost"
    # ]
    electrolyzer_total_capital_cost = (greenheart_config['electrolyzer']
                                       ['electrolyzer_capex']
                                       * greenheart_config['electrolyzer']['rating']*1000
                                       )
    

    ## h2 storage
    if greenheart_config["h2_storage"]["type"] == "none":
        h2_storage_capex = 0.0
    elif (
        greenheart_config["h2_storage"]["type"] == "pipe"
    ):  # ug pipe storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        greenheart_config["h2_storage"]["type"] == "turbine"
    ):  # ug pipe storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        greenheart_config["h2_storage"]["type"] == "pressure_vessel"
    ):  # pressure vessel storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    elif (
        greenheart_config["h2_storage"]["type"] == "salt_cavern"
    ):  # salt cavern storage model includes compression
        # h2_storage_capex = h2_storage_results["storage_capex"]
        h2_storage_capex = 160 * storage_input["h2_storage_kg"]
    elif (
        greenheart_config["h2_storage"]["type"] == "lined_rock_cavern"
    ):  # lined rock cavern storage model includes compression
        h2_storage_capex = h2_storage_results["storage_capex"]
    else:
        raise NotImplementedError(
            "the storage type you have indicated (%s) has not been implemented."
            % greenheart_config["h2_storage"]["type"]
        )

    # store capex component breakdown
    capex_breakdown = {
        "wind": wind_capex,
        "wave": wave_capex,
        "solar": solar_capex,
        "battery": battery_capex,
        # "platform": platform_costs,
        # "electrical_export_system": wind_cost_results.total_used_export_system_costs,
        # "desal": desal_capex,
        "electrolyzer": electrolyzer_total_capital_cost,
        # "h2_pipe_array": h2_pipe_array_results["capex"],
        # "h2_transport_compressor": h2_transport_compressor_capex,
        # "h2_transport_pipeline": h2_transport_pipe_capex,
        "h2_storage": h2_storage_capex,
    }

    for key in capex_breakdown.keys():
        if key == "h2_storage":
            # if design_scenario["h2_storage_location"] == "turbine" and greenheart_config["h2_storage"]["type"] == "turbine":
            #     cost_year = greenheart_config["finance_parameters"]["discount_years"][key][
            #         design_scenario["h2_storage_location"]
            #     ]
            # else:
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key][
                greenheart_config["h2_storage"]["type"]
            ]
        else:
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key]

        periods = greenheart_config["project_parameters"]["cost_year"] - cost_year

        capex_breakdown[key] = -npf.fv(
            greenheart_config["finance_parameters"]["costing_general_inflation"],
            periods,
            0.0,
            capex_breakdown[key],
        )

    total_system_installed_cost = sum(
        capex_breakdown[key] for key in capex_breakdown.keys()
    )
    if verbose:
        print("\nCAPEX Breakdown")
        for key in capex_breakdown.keys():
            print(key, "%.2f" % (capex_breakdown[key] * 1e-6), " M")

        print(
            "\nTotal system CAPEX: ",
            "$%.2f" % (total_system_installed_cost * 1e-9),
            " B",
        )

    return total_system_installed_cost, capex_breakdown


total_system_installed_cost, capex_breakdown = run_capex(
    hopp_results,
    storage_input,
    electrolyzer_cost_results,
    h2_storage_results,
    hopp_config,
    greenheart_config,
    verbose=True,
)


def run_opex(
    hopp_results,
    electrolyzer_physics_results,
    storage_input,
    # wind_cost_results,
    electrolyzer_cost_results,
    # h2_pipe_array_results,
    # h2_transport_compressor_results,
    # h2_transport_pipe_results,
    h2_storage_results,
    hopp_config,
    greenheart_config,
    # desal_results,
    # platform_results,
    verbose=False,
    total_export_system_cost=0,
):
    # WIND ONLY Total O&M expenses including fixed, variable, and capacity-based, $/year
    # use values from hybrid substation if a hybrid plant
    # if orbit_hybrid_electrical_export_project is None:

    # wave opex
    if hopp_config["site"]["wave"]:
        cost_dict = hopp_results["hybrid_plant"].wave.mhk_costs.cost_outputs
        wave_opex = cost_dict["maintenance_cost"] + cost_dict["operations_cost"]
    else:
        wave_opex = 0.0
    print("annual energies", hopp_results["annual_energies"])
    # wind opex
    if "wind" in hopp_config["technologies"].keys():
        wind_opex = hi.system.wind.om_total_expense[0]*3
        print(wind_opex)
        if wind_opex < 0.1:
            raise (RuntimeWarning(f"Wind OPEX returned as {wind_opex}"))
    else:
        wind_opex = 0.0

    # solar opex
    if "pv" in hopp_config["technologies"].keys():
        solar_opex = hi.system.pv.om_total_expense[0]*3
        if solar_opex < 0.1:
            raise (RuntimeWarning(f"Solar OPEX returned as {solar_opex}"))
    else:
        solar_opex = 0.0

    # battery opex
    if "battery" in hopp_config["technologies"].keys():
        battery_opex = (hopp_results["hybrid_plant"].battery.om_fixed
                        * hopp_config['technologies']['battery']['system_capacity_kw']*3
        ) 
        # if battery_opex < 0.1:
        #     raise (RuntimeWarning(f"Battery OPEX returned as {battery_opex}"))
    else:
        battery_opex = 0.0

    # H2 OPEX
    # platform_operating_costs = platform_results["opex"]  # TODO update this

    # annual_operating_cost_h2 = electrolyzer_cost_results["electrolyzer_OM_cost_annual"]
    annual_operating_cost_h2 = (31.03 * greenheart_config['electrolyzer']['rating'] #Fixed OM
                                + 1.68 * (electrolyzer_physics_results['H2_Results']['Life: Annual H2 production [kg/year]']*55.49/1E3)
                                ) 

    # h2_transport_compressor_opex = h2_transport_compressor_results[
    # "compressor_opex"
    # ]  # annual

    # h2_transport_pipeline_opex = h2_transport_pipe_results["annual operating cost [$]"][
    #     0
    # ]  # annual

    # storage_opex = h2_storage_results["storage_opex"]
    storage_opex = (10/12) * storage_input["h2_storage_kg"] #$10/kg-month
    # # desal OPEX
    # if desal_results != None:
    #     desal_opex = desal_results["desal_opex_usd_per_year"]
    # else:
    #     desal_opex = 0.0
    # annual_operating_cost_desal = desal_opex

    # store opex component breakdown
    opex_breakdown_annual = {
        "wind_and_electrical": wind_opex,
        # "platform": platform_operating_costs,
        #   "electrical_export_system": total_export_om_cost,
        "wave": wave_opex,
        "solar": solar_opex,
        "battery": battery_opex,
        # "desal": annual_operating_cost_desal,
        "electrolyzer": annual_operating_cost_h2,
        # "h2_pipe_array": h2_pipe_array_results["opex"],
        # "h2_transport_compressor": h2_transport_compressor_opex,
        # "h2_transport_pipeline": h2_transport_pipeline_opex,
        "h2_storage": storage_opex,
    }

    # discount opex to appropriate year for unified costing
    for key in opex_breakdown_annual.keys():
        if key == "h2_storage":
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key][
                greenheart_config["h2_storage"]["type"]
            ]
        else:
            cost_year = greenheart_config["finance_parameters"]["discount_years"][key]

        periods = greenheart_config["project_parameters"]["cost_year"] - cost_year
        opex_breakdown_annual[key] = -npf.fv(
            greenheart_config["finance_parameters"]["costing_general_inflation"],
            periods,
            0.0,
            opex_breakdown_annual[key],
        )

    # Calculate the total annual OPEX of the installed system
    total_annual_operating_costs = sum(opex_breakdown_annual.values())

    if verbose:
        print("\nAnnual OPEX Breakdown")
        for key in opex_breakdown_annual.keys():
            print(key, "%.2f" % (opex_breakdown_annual[key] * 1e-6), " M")

        print(
            "\nTotal Annual OPEX: ",
            "$%.2f" % (total_annual_operating_costs * 1e-6),
            " M",
        )
        print(opex_breakdown_annual)
    return total_annual_operating_costs, opex_breakdown_annual


total_annual_operating_costs, opex_breakdown_annual = run_opex(
    hopp_results,
    electrolyzer_physics_results,
    storage_input,
    electrolyzer_cost_results,
    h2_storage_results,
    hopp_config,
    greenheart_config,
    verbose=True,
)

design_scenario = {
    "h2_storage_location": "onshore",
    "electrolyzer_location": "onshore",
    "transportation": "colocated",  # can be one of ["hvdc", "pipeline", "none", hvdc+pipeline, "colocated"]
    "wind_location": "onshore",  # can be one of ["onshore", "offshore"]
    "pv_location": "onshore",  # can be one of ["none", "onshore", "platform"]
    "battery_location": "onshore",  # can be one of ["none", "onshore", "platform"]'
}

incentive_option = "1"


def run_profast_lcoe(
    greenheart_config,
    # wind_cost_results,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    incentive_option,
    design_scenario,
    verbose=False,
    show_plots=False,
    save_plots=False,
    output_dir="./output/",
):
    gen_inflation = greenheart_config["finance_parameters"]["profast_general_inflation"]

    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        if "land_cost" in greenheart_config["finance_parameters"]:
            land_cost = greenheart_config["finance_parameters"]["land_cost"]
        else:
            land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST()
    pf.set_params(
        "commodity",
        {
            "name": "electricity",
            "unit": "kWh",
            "initial price": 100,
            "escalation": gen_inflation,
        },
    )
    pf.set_params(
        "capacity",
        np.sum(hopp_results["combined_hybrid_power_production_hopp"]) / 365.0,
    )  # kWh/day
    pf.set_params("maintenance", {"value": 0, "escalation": gen_inflation})
    pf.set_params(
        "analysis start year", greenheart_config["project_parameters"]["atb_year"] + 1
    )
    pf.set_params(
        "operating life", greenheart_config["project_parameters"]["project_lifetime"]
    )
    pf.set_params("installation months", 36)
    pf.set_params(
        "installation cost",
        {
            "value": 0,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False,
        },
    )
    if land_cost > 0:
        pf.set_params("non depr assets", land_cost)
        pf.set_params(
            "end of proj sale non depr assets",
            land_cost
            * (1 + gen_inflation)
            ** greenheart_config["project_parameters"]["project_lifetime"],
        )
    pf.set_params("demand rampup", 0)
    pf.set_params("long term utilization", 1)
    pf.set_params("credit card fees", 0)
    pf.set_params(
        "sales tax", greenheart_config["finance_parameters"]["sales_tax_rate"]
    )
    pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
    pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
    pf.set_params(
        "property tax and insurance",
        greenheart_config["finance_parameters"]["property_tax"]
        + greenheart_config["finance_parameters"]["property_insurance"],
    )
    pf.set_params(
        "admin expense",
        greenheart_config["finance_parameters"][
            "administrative_expense_percent_of_sales"
        ],
    )
    pf.set_params(
        "total income tax rate",
        greenheart_config["finance_parameters"]["total_income_tax_rate"],
    )
    pf.set_params(
        "capital gains tax rate",
        greenheart_config["finance_parameters"]["capital_gains_tax_rate"],
    )
    pf.set_params("sell undepreciated cap", True)
    pf.set_params("tax losses monetized", True)
    pf.set_params("general inflation rate", gen_inflation)
    pf.set_params(
        "leverage after tax nominal discount rate",
        greenheart_config["finance_parameters"]["discount_rate"],
    )
    pf.set_params(
        "debt equity ratio of initial financing",
        (
            greenheart_config["finance_parameters"]["debt_equity_split"]
            / (100 - greenheart_config["finance_parameters"]["debt_equity_split"])
        ),
    )
    pf.set_params("debt type", greenheart_config["finance_parameters"]["debt_type"])
    pf.set_params(
        "loan period if used", greenheart_config["finance_parameters"]["loan_period"]
    )
    pf.set_params(
        "debt interest rate",
        greenheart_config["finance_parameters"]["debt_interest_rate"],
    )
    pf.set_params(
        "cash onhand", greenheart_config["finance_parameters"]["cash_onhand_months"]
    )

    # ----------------------------------- Add capital items to ProFAST ----------------
    if "wind" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wind System",
            cost=capex_breakdown["wind"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
    if "wave" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wave System",
            cost=capex_breakdown["wave"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )

    if "solar" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Solar System",
            cost=capex_breakdown["solar"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )

    if "battery" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Battery System",
            cost=capex_breakdown["battery"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0,0,0,0,0,0,0,0,0,0.85,0,0,0,0,0,0,0,0,0],
        )

    # if design_scenario["transportation"] == "hvdc+pipeline" or not (
    #     design_scenario["electrolyzer_location"] == "turbine"
    #     and design_scenario["h2_storage_location"] == "turbine"
    # ):
    #     pf.add_capital_item(
    #         name="Electrical Export system",
    #         cost=capex_breakdown["electrical_export_system"],
    #         depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
    #         depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
    #         refurb=[0],
    #     )

    # -------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(
        name="Wind and Electrical Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["wind_and_electrical"],
        escalation=gen_inflation,
    )
    print("opex breakdown", opex_breakdown["wind_and_electrical"])

    if "wave" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Wave O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["wave"],
            escalation=gen_inflation,
        )

    if "solar" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Solar O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["solar"],
            escalation=gen_inflation,
        )

    if "battery" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Battery O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["battery"],
            escalation=gen_inflation,
        )

    # ------------------------------------- add incentives -----------------------------------
    """ Note: ptc units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced
        Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""

    # catch incentive option and add relevant incentives
    incentive_dict = greenheart_config["policy_parameters"][
        "option%s" % (incentive_option)
    ]
    # add electricity_ptc ($/kW)
    # adjust from 1992 dollars to start year
    wind_ptc_in_dollars_per_kw = -npf.fv(
        greenheart_config["finance_parameters"]["costing_general_inflation"],
        greenheart_config["project_parameters"]["atb_year"] + round((36 / 12)) - 1992,
        0,
        incentive_dict["electricity_ptc"],
    )  # given in 1992 dollars but adjust for inflation

    pf.add_incentive(
        name="Electricity PTC",
        value=wind_ptc_in_dollars_per_kw,
        decay=-gen_inflation,
        sunset_years=10,
        tax_credit=True,
    )  # TODO check decay

        # add wind_itc (% of wind capex)
    electricity_itc_value_percent_wind_capex = incentive_dict["electricity_itc"]
    electricity_itc_value_dollars = electricity_itc_value_percent_wind_capex * (
        capex_breakdown["wind"] +capex_breakdown['solar'] ) #+ capex_breakdown["electrical_export_system"]
    # )
    pf.set_params(
        "one time cap inct",
        {
            "value": electricity_itc_value_dollars,
            "depr type": greenheart_config["finance_parameters"]["depreciation_method"],
            "depr period": greenheart_config["finance_parameters"][
                "depreciation_period"
            ],
            "depreciable": True,
        },
    )
    sol = pf.solve_price()

    lcoe = sol["price"]

    if verbose:
        print("\nProFAST LCOE: ", "%.2f" % (lcoe * 1e3), "$/MWh")

    if show_plots or save_plots:
        savepath = output_dir + "figures/wind_only/"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        pf.plot_costs_yearly(
            per_kg=False,
            scale="M",
            remove_zeros=True,
            remove_depreciation=False,
            fileout=savepath
            + "annual_cash_flow_wind_only_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_costs_yearly2(
            per_kg=False,
            scale="M",
            remove_zeros=True,
            remove_depreciation=False,
            fileout=savepath
            + "annual_cash_flow_wind_only_%i.html" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_capital_expenses(
            fileout=savepath + "capital_expense_only_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout=savepath + "cash_flow_wind_only_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )
        pf.plot_costs(
            fileout=savepath + "cost_breakdown_%i.png" % (design_scenario["id"]),
            show_plot=show_plots,
        )

    return lcoe, pf


lcoe, pf = run_profast_lcoe(
    greenheart_config,
    capex_breakdown,
    opex_breakdown_annual,
    hopp_results,
    incentive_option,
    design_scenario,
)

print("LCOE [$/MWh]", lcoe*1E3)


def run_profast_full_plant_model(
    greenheart_config,
    # wind_cost_results,
    electrolyzer_physics_results,
    capex_breakdown,
    opex_breakdown,
    hopp_results,
    incentive_option,
    design_scenario,
    # total_accessory_power_renewable_kw,
    total_accessory_power_grid_kw,
    verbose=False,
    show_plots=False,
    save_plots=False,
    output_dir="./output/",
):
    gen_inflation = greenheart_config["finance_parameters"]["profast_general_inflation"]

    if (
        design_scenario["h2_storage_location"] == "onshore"
        or design_scenario["electrolyzer_location"] == "onshore"
    ):
        if 'land_cost' in greenheart_config['finance_parameters']:
            land_cost = greenheart_config['finance_parameters']['land_cost']
        else:
            land_cost = 1e6  # TODO should model this
    else:
        land_cost = 0.0

    pf = ProFAST.ProFAST()
    pf.set_params(
        "commodity",
        {
            "name": "Hydrogen",
            "unit": "kg",
            "initial price": 100,
            "escalation": gen_inflation,
        },
    )
    pf.set_params(
        "capacity",
        electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ]
        / 365.0,
    )  # kg/day
    pf.set_params("maintenance", {"value": 0, "escalation": gen_inflation})
    pf.set_params(
        "analysis start year",
        greenheart_config["project_parameters"]["atb_year"]
        + 2,  # Add financial analysis start year
    )
    pf.set_params(
        "operating life", greenheart_config["project_parameters"]["project_lifetime"]
    )
    pf.set_params(
        "installation months",
        36,  # Add installation time to yaml default=0
    )
    pf.set_params(
        "installation cost",
        {
            "value": 0,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False,
        },
    )
    if land_cost > 0:
        pf.set_params("non depr assets", land_cost)
        pf.set_params(
            "end of proj sale non depr assets",
            land_cost
            * (1 + gen_inflation)
            ** greenheart_config["project_parameters"]["project_lifetime"],
        )
    pf.set_params("demand rampup", 0)
    pf.set_params("long term utilization", 1)  # TODO should use utilization
    pf.set_params("credit card fees", 0)
    pf.set_params(
        "sales tax", greenheart_config["finance_parameters"]["sales_tax_rate"]
    )
    pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
    pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
    # TODO how to handle property tax and insurance for fully offshore?
    pf.set_params(
        "property tax and insurance",
        greenheart_config["finance_parameters"]["property_tax"]
        + greenheart_config["finance_parameters"]["property_insurance"],
    )
    pf.set_params(
        "admin expense",
        greenheart_config["finance_parameters"][
            "administrative_expense_percent_of_sales"
        ],
    )
    pf.set_params(
        "total income tax rate",
        greenheart_config["finance_parameters"]["total_income_tax_rate"],
    )
    pf.set_params(
        "capital gains tax rate",
        greenheart_config["finance_parameters"]["capital_gains_tax_rate"],
    )
    pf.set_params("sell undepreciated cap", True)
    pf.set_params("tax losses monetized", True)
    pf.set_params("general inflation rate", gen_inflation)
    pf.set_params(
        "leverage after tax nominal discount rate",
        greenheart_config["finance_parameters"]["discount_rate"],
    )
    if greenheart_config["finance_parameters"]["debt_equity_split"]:
        pf.set_params(
            "debt equity ratio of initial financing",
            (
                greenheart_config["finance_parameters"]["debt_equity_split"]
                / (100 - greenheart_config["finance_parameters"]["debt_equity_split"])
            ),
        )  # TODO this may not be put in right
    elif greenheart_config["finance_parameters"]["debt_equity_ratio"]:
        pf.set_params(
            "debt equity ratio of initial financing",
            (greenheart_config["finance_parameters"]["debt_equity_ratio"]),
        )  # TODO this may not be put in right
    pf.set_params("debt type", greenheart_config["finance_parameters"]["debt_type"])
    pf.set_params(
        "loan period if used", greenheart_config["finance_parameters"]["loan_period"]
    )
    pf.set_params(
        "debt interest rate",
        greenheart_config["finance_parameters"]["debt_interest_rate"],
    )
    pf.set_params(
        "cash onhand", greenheart_config["finance_parameters"]["cash_onhand_months"]
    )

    # ----------------------------------- Add capital and fixed items to ProFAST ----------------
    if "wind" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wind System",
            cost=capex_breakdown["wind"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
    if "wave" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Wave System",
            cost=capex_breakdown["wave"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
    if "solar" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Solar System",
            cost=capex_breakdown["solar"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )

    if "battery" in capex_breakdown.keys():
        pf.add_capital_item(
            name="Battery System",
            cost=capex_breakdown["battery"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0,0,0,0,0,0,0,0,0,0.85,0,0,0,0,0,0,0,0,0],
        )

    if "platform" in capex_breakdown.keys() and capex_breakdown["platform"] > 0:
        pf.add_capital_item(
            name="Equipment Platform",
            cost=capex_breakdown["platform"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="Equipment Platform O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["platform"],
            escalation=gen_inflation,
        )

    pf.add_fixed_cost(
        name="Wind and Electrical Export Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["wind_and_electrical"],
        escalation=gen_inflation,
    )
    if "wave" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Wave O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["wave"],
            escalation=gen_inflation,
        )

    if "solar" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Solar O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["solar"],
            escalation=gen_inflation,
        )

    if "battery" in opex_breakdown.keys():
        pf.add_fixed_cost(
            name="Battery O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["battery"],
            escalation=gen_inflation,
        )

    # if design_scenario["transportation"] == "hvdc+pipeline" or not (
    #     design_scenario["electrolyzer_location"] == "turbine"
    #     and design_scenario["h2_storage_location"] == "turbine"
    # ):
    #     pf.add_capital_item(
    #         name="Electrical Export system",
    #         cost=capex_breakdown["electrical_export_system"],
    #         depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
    #         depr_period=greenheart_config["finance_parameters"]["depreciation_period"],
    #         refurb=[0],
    #     )
    #     # TODO assess if this makes sense (electrical export O&M included in wind O&M)

    electrolyzer_refurbishment_schedule = np.zeros(
        greenheart_config["project_parameters"]["project_lifetime"]
    )
    # refurb_period = round(
    #     greenheart_config["electrolyzer"]["time_between_replacement"] / (24 * 365)
    # )
    refurb_period = round(electrolyzer_physics_results['H2_Results']['Time Until Replacement [hrs]']/ (24 * 365)
    )
    electrolyzer_refurbishment_schedule[
        refurb_period : greenheart_config["project_parameters"][
            "project_lifetime"
        ] : refurb_period
    ] = greenheart_config["electrolyzer"]["replacement_cost_percent"]

    print("Electrolyzer refurb", electrolyzer_refurbishment_schedule)

    # pf.add_capital_item(
    #     name="Electrolysis System",
    #     cost=capex_breakdown["electrolyzer"],
    #     depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
    #     depr_period=greenheart_config["finance_parameters"][
    #         "depreciation_period_electrolyzer"
    #     ],
    #     refurb=list(electrolyzer_refurbishment_schedule),
    # )
    pf.add_capital_item(
        name="Electrolysis System",
        cost=capex_breakdown["electrolyzer"],
        depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
        depr_period=greenheart_config["finance_parameters"][
            "depreciation_period_electrolyzer"
        ],
        refurb=[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    )
    pf.add_fixed_cost(
        name="Electrolysis System Fixed O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=opex_breakdown["electrolyzer"],
        escalation=gen_inflation,
    )

    if design_scenario["electrolyzer_location"] == "turbine":
        pf.add_capital_item(
            name="H2 Pipe Array System",
            cost=capex_breakdown["h2_pipe_array"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="H2 Pipe Array Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_pipe_array"],
            escalation=gen_inflation,
        )

    if (
        (
            design_scenario["h2_storage_location"] == "onshore"
            and design_scenario["electrolyzer_location"] != "onshore"
        )
        or (
            design_scenario["h2_storage_location"] != "onshore"
            and design_scenario["electrolyzer_location"] == "onshore"
        )
        or (design_scenario["transportation"] == "hvdc+pipeline")
    ):
        pf.add_capital_item(
            name="H2 Transport Compressor System",
            cost=capex_breakdown["h2_transport_compressor"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_capital_item(
            name="H2 Transport Pipeline System",
            cost=capex_breakdown["h2_transport_pipeline"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )

        pf.add_fixed_cost(
            name="H2 Transport Compression Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_transport_compressor"],
            escalation=gen_inflation,
        )
        pf.add_fixed_cost(
            name="H2 Transport Pipeline Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_transport_pipeline"],
            escalation=gen_inflation,
        )

    if greenheart_config["h2_storage"]["type"] != "none":
        pf.add_capital_item(
            name="Hydrogen Storage System",
            cost=capex_breakdown["h2_storage"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="Hydrogen Storage Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["h2_storage"],
            escalation=gen_inflation,
        )

    # ---------------------- Add feedstocks, note the various cost options-------------------
    if design_scenario["electrolyzer_location"] == "onshore":
        galperkg = 3.785411784
        pf.add_feedstock(
            name="Water",
            usage=sum(
                electrolyzer_physics_results["H2_Results"][
                    "Water Hourly Consumption [kg/hr]"
                ]
            )
            * galperkg
            / electrolyzer_physics_results["H2_Results"][
                "Life: Annual H2 production [kg/year]"
            ],
            unit="gal",
            cost="US Average",
            escalation=gen_inflation,
        )
    else:
        pf.add_capital_item(
            name="Desal System",
            cost=capex_breakdown["desal"],
            depr_type=greenheart_config["finance_parameters"]["depreciation_method"],
            depr_period=greenheart_config["finance_parameters"][
                "depreciation_period_electrolyzer"
            ],
            refurb=[0],
        )
        pf.add_fixed_cost(
            name="Desal Fixed O&M Cost",
            usage=1.0,
            unit="$/year",
            cost=opex_breakdown["desal"],
            escalation=gen_inflation,
        )

    if (
        greenheart_config["project_parameters"]["grid_connection"]
        or total_accessory_power_grid_kw > 0
    ):

        energy_purchase = total_accessory_power_grid_kw * 365 * 24

        if greenheart_config["project_parameters"]["grid_connection"]:
            annual_energy_shortfall = np.sum(hopp_results["energy_shortfall_hopp"])
            energy_purchase += annual_energy_shortfall

        pf.add_fixed_cost(
            name="Electricity from grid",
            usage=1.0,
            unit="$/year",
            cost=energy_purchase * greenheart_config["project_parameters"]["ppa_price"],
            escalation=gen_inflation,
        )

    # ------------------------------------- add incentives -----------------------------------
    """ Note: units must be given to ProFAST in terms of dollars per unit of the primary commodity being produced
        Note: full tech-nutral (wind) tax credits are no longer available if constructions starts after Jan. 1 2034 (Jan 1. 2033 for h2 ptc)"""

    # catch incentive option and add relevant incentives
    incentive_dict = greenheart_config["policy_parameters"][
        "option%s" % (incentive_option)
    ]

    # add wind_itc (% of wind capex)
    electricity_itc_value_percent_wind_capex = incentive_dict["electricity_itc"]
    electricity_itc_value_dollars = electricity_itc_value_percent_wind_capex * (
        capex_breakdown["wind"] +capex_breakdown['solar'] ) #+ capex_breakdown["electrical_export_system"]
    # )
    pf.set_params(
        "one time cap inct",
        {
            "value": electricity_itc_value_dollars,
            "depr type": greenheart_config["finance_parameters"]["depreciation_method"],
            "depr period": greenheart_config["finance_parameters"][
                "depreciation_period"
            ],
            "depreciable": True,
        },
    )

    # add h2_storage_itc (% of h2 storage capex)
    itc_value_percent_h2_store_capex = incentive_dict["h2_storage_itc"]
    electricity_itc_value_dollars_h2_store = itc_value_percent_h2_store_capex * (
        capex_breakdown["h2_storage"]
    )
    pf.set_params(
        "one time cap inct",
        {
            "value": electricity_itc_value_dollars_h2_store,
            "depr type": greenheart_config["finance_parameters"]["depreciation_method"],
            "depr period": greenheart_config["finance_parameters"][
                "depreciation_period"
            ],
            "depreciable": True,
        },
    )

    # add electricity_ptc ($/kW)
    # adjust from 1992 dollars to start year
    electricity_ptc_in_dollars_per_kw = -npf.fv(
        greenheart_config['finance_parameters']['costing_general_inflation'],
        greenheart_config["project_parameters"]["atb_year"]
        + round((36 / 12))
        - 1992,
        0,
        incentive_dict["electricity_ptc"],
    )  # given in 1992 dollars but adjust for inflation
    kw_per_kg_h2 = (
        sum(hopp_results["combined_hybrid_power_production_hopp"])
        / electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ]
    )
    electricity_ptc_in_dollars_per_kg_h2 = (
        electricity_ptc_in_dollars_per_kw * kw_per_kg_h2
    )
    pf.add_incentive(
        name="Electricity PTC",
        value=electricity_ptc_in_dollars_per_kg_h2,
        decay=-gen_inflation,
        sunset_years=10,
        tax_credit=True,
    )  # TODO check decay

    # add h2_ptc ($/kg)
    h2_ptc_inflation_adjusted = -npf.fv(
        greenheart_config['finance_parameters']['costing_general_inflation'], # use ATB year (cost inflation 2.5%) costing_general_inflation
        greenheart_config["project_parameters"]["atb_year"]
        + round((36 / 12))
        - 2022,
        0,
        incentive_dict["h2_ptc"],
    )
    pf.add_incentive(
        name="H2 PTC",
        value=h2_ptc_inflation_adjusted,
        decay=-gen_inflation, #correct inflation
        sunset_years=10,
        tax_credit=True,
    )  # TODO check decay

    # ------------------------------------ solve and post-process -----------------------------

    sol = pf.solve_price()

    df = pf.cash_flow_out

    lcoh = sol["price"]

    if verbose:
        print("\nProFAST LCOH: ", "%.2f" % (lcoh), "$/kg")
        print("ProFAST NPV: ", "%.2f" % (sol["NPV"]))
        print("ProFAST IRR: ", "%.5f" % (max(sol["irr"])))
        print("ProFAST LCO: ", "%.2f" % (sol["lco"]), "$/kg")
        print("ProFAST Profit Index: ", "%.2f" % (sol["profit index"]))
        print("ProFAST payback period: ", sol["investor payback period"])

        MIRR = npf.mirr(
            df["Investor cash flow"],
            greenheart_config["finance_parameters"]["debt_interest_rate"],
            greenheart_config["finance_parameters"]["discount_rate"],
        )  # TODO probably ignore MIRR
        NPV = npf.npv(
            greenheart_config["finance_parameters"]["profast_general_inflation"],
            df["Investor cash flow"],
        )
        ROI = np.sum(df["Investor cash flow"]) / abs(
            np.sum(df["Investor cash flow"][df["Investor cash flow"] < 0])
        )  # ROI is not a good way of thinking about the value of the project

        # TODO project level IRR - capex and operating cash flow

        # note: hurdle rate typically 20% IRR before investing in it due to typically optimistic assumptions
        # note: negative retained earnings (keeping debt, paying down equity) - to get around it, do another line for retained earnings and watch dividends paid by the rpoject (net income/equity should stay positive this way)

        print("Investor NPV: ", np.round(NPV * 1e-6, 2), "M USD")
        print("Investor MIRR: ", np.round(MIRR, 5), "")
        print("Investor ROI: ", np.round(ROI, 5), "")

    if save_plots or show_plots:
        savepaths = [
            output_dir + "figures/capex/",
            output_dir + "figures/annual_cash_flow/",
            output_dir + "figures/lcoh_breakdown/",
            output_dir + "data/",
        ]
        for savepath in savepaths:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        pf.plot_capital_expenses(
            fileout=savepaths[0] + "capital_expense_%i.pdf" % (3),
            show_plot=show_plots,
        )
        pf.plot_cashflow(
            fileout=savepaths[1] + "cash_flow_%i.png" % (3),
            show_plot=show_plots,
        )

        pd.DataFrame.from_dict(data=pf.cash_flow_out).to_csv(
            savepaths[3] + "cash_flow_%i.csv" % (3)
        )

        pf.plot_costs(
            savepaths[2] + "lcoh_%i" % (3), show_plot=show_plots,
        )

    return lcoh, pf

total_accessory_power_grid_kw = 0

lcoh, pf = run_profast_full_plant_model(
    greenheart_config,
    electrolyzer_physics_results,
    capex_breakdown,
    opex_breakdown_annual,
    hopp_results,
    incentive_option,
    design_scenario,
    total_accessory_power_grid_kw,
    verbose=True,
    show_plots=True,
    save_plots=False,
    output_dir="./output/",
)

# print("LCOH [$/kg] ", lcoh)