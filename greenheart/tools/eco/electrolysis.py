import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os

# import hopp.tools.hopp_tools as hopp_tools

from greenheart.simulation.technologies.hydrogen.desal.desal_model_eco import (
    RO_desal_eco as RO_desal,
)
from greenheart.simulation.technologies.hydrogen.electrolysis.pem_mass_and_footprint import (
    mass as run_electrolyzer_mass,
)
from greenheart.simulation.technologies.hydrogen.electrolysis.pem_mass_and_footprint import (
    footprint as run_electrolyzer_footprint,
)
from greenheart.simulation.technologies.hydrogen.electrolysis.H2_cost_model import (
    basic_H2_cost_model,
)
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_costs_Singlitico_model import (
    PEMCostsSingliticoModel,
)

# from hopp.simulation.technologies.hydrogen.electrolysis.run_h2_PEM_eco import run_h2_PEM
from greenheart.simulation.technologies.hydrogen.electrolysis.run_h2_PEM import (
    run_h2_PEM,
)
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import (
    PEM_H2_Clusters as PEMClusters,
)

# from electrolyzer import run_electrolyzer


def run_electrolyzer_physics(
    hopp_results,
    greenheart_config,
    wind_resource,
    design_scenario,
    show_plots=False,
    save_plots=False,
    output_dir="./output/",
    verbose=False,
):

    electrolyzer_size_mw = greenheart_config["electrolyzer"]["rating"]
    electrolyzer_capex_kw = greenheart_config["electrolyzer"]["electrolyzer_capex"]
    
    # IF GRID CONNECTED
    if greenheart_config["project_parameters"]["grid_connection"]:
        #NOTE: if grid-connected, it assumes that hydrogen demand is input and there is not
        # multi-cluster control strategies. This capability exists at the cluster level, not at the
        # system level.
        if greenheart_config["electrolyzer"]["sizing"]["hydrogen_dmd"] is not None:
            grid_connection_scenario='grid-only'
            hydrogen_production_capacity_required_kgphr=greenheart_config["electrolyzer"]["sizing"]["hydrogen_dmd"]
            energy_to_electrolyzer_kw = []
        else:
            grid_connection_scenario='off-grid'
            hydrogen_production_capacity_required_kgphr = []
            energy_to_electrolyzer_kw = np.ones(8760)*electrolyzer_size_mw*1e3
    # IF NOT GRID CONNECTED
    else:
        hydrogen_production_capacity_required_kgphr = []
        grid_connection_scenario = 'off-grid'
        energy_to_electrolyzer_kw = np.asarray(hopp_results[
            "combined_hybrid_power_production_hopp"
        ])
    n_pem_clusters = electrolyzer_size_mw//greenheart_config["electrolyzer"]["cluster_rating_MW"]
        
    ## run using greensteel model
    pem_param_dict = {
        "eol_eff_percent_loss":greenheart_config["electrolyzer"]["eol_eff_percent_loss"],
        "uptime_hours_until_eol": greenheart_config["electrolyzer"]["uptime_hours_until_eol"],
        "include_degradation_penalty":greenheart_config["electrolyzer"]["include_degradation_penalty"],
        "turndown_ratio":greenheart_config["electrolyzer"]["turndown_ratio"],
    }
    
    H2_Results, h2_ts, h2_tot, power_to_electrolyzer_kw = run_h2_PEM(electrical_generation_timeseries=energy_to_electrolyzer_kw, 
               electrolyzer_size=electrolyzer_size_mw,
               useful_life=greenheart_config["project_parameters"]["project_lifetime"], # EG: should be in years for full plant life - only used in financial model
               n_pem_clusters=n_pem_clusters,  
               pem_control_type=greenheart_config["electrolyzer"]["pem_control_type"],
               electrolyzer_direct_cost_kw=electrolyzer_capex_kw, 
               user_defined_pem_param_dictionary=pem_param_dict,
               grid_connection_scenario=grid_connection_scenario, # if not offgrid, assumes steady h2 demand in kgphr for full year
               hydrogen_production_capacity_required_kgphr=hydrogen_production_capacity_required_kgphr,
               debug_mode = False,
               verbose=verbose
               )

    # calculate mass and foorprint of system
    mass_kg = run_electrolyzer_mass(electrolyzer_size_mw)
    footprint_m2 = run_electrolyzer_footprint(electrolyzer_size_mw)

    # store results for return
    electrolyzer_physics_results = {
        "H2_Results": H2_Results,
        "capacity_factor": H2_Results["Life: Capacity Factor"],
        "equipment_mass_kg": mass_kg,
        "equipment_footprint_m2": footprint_m2,
        "power_to_electrolyzer_kw": power_to_electrolyzer_kw,
    }

    if verbose:
        print("\nElectrolyzer Physics:")  # 61837444.34555772 145297297.29729727
        print("H2 Produced Annually (tonnes): ", H2_Results["Life: Annual H2 production [kg/year]"]*1E-3)
        print(
            "Max H2 hourly (tonnes): ",
            max(H2_Results["Hydrogen Hourly Production [kg/hr]"]) * 1e-3,
        )
        print(
            "Max H2 daily (tonnes): ",
            max(
                np.convolve(
                    H2_Results["Hydrogen Hourly Production [kg/hr]"], np.ones(24), mode="valid"
                )
            )
            * 1e-3,
        )
        
        prodrate = 1.0 / round(H2_Results['Rated BOL: Efficiency [kWh/kg]'],2) # kg/kWh
        roughest = power_to_electrolyzer_kw * prodrate
        print("Energy to electrolyzer (kWh): ", sum(power_to_electrolyzer_kw))
        print(
            "Energy per kg (kWh/kg): ",
            H2_Results['Sim: Total Input Power [kWh]'] / H2_Results['Sim: Total H2 Produced [kg]'],
        )
        print("Max hourly based on est kg/kWh (kg): ", max(roughest))
        print(
            "Max daily rough est (tonnes): ",
            max(np.convolve(roughest, np.ones(24), mode="valid")) * 1e-3,
        )
        print("Electrolyzer Life Average Capacity Factor: ", H2_Results["Life: Capacity Factor"])

    if save_plots or show_plots:
        N = 24 * 7 * 4
        fig, ax = plt.subplots(3, 2, sharex=True, sharey="row")

        wind_speed = [W[2] for W in wind_resource._data["data"]]

        # plt.title("4-week running average")
        pad = 5
        ax[0, 0].annotate(
            "Hourly",
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )
        ax[0, 1].annotate(
            "4-week running average",
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

        ax[0, 0].plot(wind_speed)
        convolved_wind_speed = np.convolve(wind_speed, np.ones(N) / (N), mode="valid")
        ave_x = range(N, len(convolved_wind_speed) + N)

        ax[0, 1].plot(ave_x, convolved_wind_speed)
        ax[0, 0].set(ylabel="Wind\n(m/s)", ylim=[0, 30], xlim=[0, len(wind_speed)])
        tick_spacing = 10
        ax[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        y = greenheart_config["electrolyzer"]["rating"]
        ax[1, 0].plot(energy_to_electrolyzer_kw * 1e-3)
        ax[1, 0].axhline(y=y, color="r", linestyle="--", label="Nameplate Capacity")
        ax[1, 1].plot(
            ave_x[:-1],
            np.convolve(
                energy_to_electrolyzer_kw * 1e-3, np.ones(N) / (N), mode="valid"
            ),
        )
        ax[1, 1].axhline(y=y, color="r", linestyle="--", label="Nameplate Capacity")
        ax[1, 0].set(
            ylabel="Electrolyzer \nPower (MW)", ylim=[0, 500], xlim=[0, len(wind_speed)]
        )
        # ax[1].legend(frameon=False, loc="best")
        tick_spacing = 200
        ax[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax[1, 0].text(1000, y + 0.1 * tick_spacing, "Electrolyzer Rating", color="r")

        ax[2, 0].plot(H2_Results["hydrogen_hourly_production"] * 1e-3)
        ax[2, 1].plot(
            ave_x[:-1],
            np.convolve(
                H2_Results["hydrogen_hourly_production"] * 1e-3,
                np.ones(N) / (N),
                mode="valid",
            ),
        )
        tick_spacing = 2
        ax[2, 0].set(
            xlabel="Hour",
            ylabel="Hydrogen\n(tonnes/hr)",
            # ylim=[0, 7000],
            xlim=[0, len(H2_Results["hydrogen_hourly_production"])],
        )
        ax[2, 0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax[2, 1].set(
            xlabel="Hour",
            # ylim=[0, 7000],
            xlim=[
                4 * 7 * 24 - 1,
                len(H2_Results["hydrogen_hourly_production"] + 4 * 7 * 24 + 2),
            ],
        )
        ax[2, 1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        plt.tight_layout()
        if save_plots:
            savepath = output_dir + "figures/production/"
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            plt.savefig(
                savepath + "production_overview_%i.png" % (design_scenario["id"]),
                transparent=True,
            )
        if show_plots:
            plt.show()

    return electrolyzer_physics_results


def run_electrolyzer_cost(
    electrolyzer_physics_results,
    hopp_config,
    greenheart_config,
    design_scenario,
    verbose=False,
):

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

    # run hydrogen production cost model - from hopp examples
    if design_scenario["electrolyzer_location"] == "onshore":
        offshore = 0
    else:
        offshore = 1

    if design_scenario["electrolyzer_location"] == "turbine":
        per_turb_electrolyzer_size_mw = electrolyzer_size_mw / nturbines
        per_turb_h2_annual_output = H2_Results["hydrogen_annual_output"] / nturbines
        per_turb_electrical_generation_timeseries = (
            electrical_generation_timeseries / nturbines
        )

        if electrolyzer_cost_model == "basic":
            (
                cf_h2_annuals,
                per_turb_electrolyzer_total_capital_cost,
                per_turb_electrolyzer_OM_cost,
                per_turb_electrolyzer_capex_kw,
                time_between_replacement,
                h2_tax_credit,
                h2_itc,
            ) = basic_H2_cost_model(
                greenheart_config["electrolyzer"]["electrolyzer_capex"],
                greenheart_config["electrolyzer"]["time_between_replacement"],
                per_turb_electrolyzer_size_mw,
                useful_life,
                atb_year,
                per_turb_electrical_generation_timeseries,
                per_turb_h2_annual_output,
                0.0,
                0.0,
                include_refurb_in_opex=False,
                offshore=offshore,
            )

        elif electrolyzer_cost_model == "singlitico2021":

            P_elec = per_turb_electrolyzer_size_mw * 1e-3  # [GW]
            RC_elec = greenheart_config["electrolyzer"][
                "electrolyzer_capex"
            ]  # [USD/kW]

            pem_offshore = PEMCostsSingliticoModel(elec_location=offshore)

            (
                per_turb_electrolyzer_capital_cost_musd,
                per_turb_electrolyzer_om_cost_musd,
            ) = pem_offshore.run(P_elec, RC_elec)

            per_turb_electrolyzer_total_capital_cost = (
                per_turb_electrolyzer_capital_cost_musd * 1e6
            )  # convert from M USD to USD
            per_turb_electrolyzer_OM_cost = (
                per_turb_electrolyzer_om_cost_musd * 1e6
            )  # convert from M USD to USD

        electrolyzer_total_capital_cost = (
            per_turb_electrolyzer_total_capital_cost * nturbines
        )
        electrolyzer_OM_cost = per_turb_electrolyzer_OM_cost * nturbines

    else:
        if electrolyzer_cost_model == "basic":
            (
                cf_h2_annuals,
                electrolyzer_total_capital_cost,
                electrolyzer_OM_cost,
                electrolyzer_capex_kw,
                time_between_replacement,
                h2_tax_credit,
                h2_itc,
            ) = basic_H2_cost_model(
                greenheart_config["electrolyzer"]["electrolyzer_capex"],
                greenheart_config["electrolyzer"]["time_between_replacement"],
                electrolyzer_size_mw,
                useful_life,
                atb_year,
                electrical_generation_timeseries,
                H2_Results["hydrogen_annual_output"],
                0.0,
                0.0,
                include_refurb_in_opex=False,
                offshore=offshore,
            )
        elif electrolyzer_cost_model == "singlitico2021":

            P_elec = electrolyzer_size_mw * 1e-3  # [GW]
            RC_elec = greenheart_config["electrolyzer"][
                "electrolyzer_capex"
            ]  # [USD/kW]

            pem_offshore = PEMCostsSingliticoModel(elec_location=offshore)

            electrolyzer_capital_cost_musd, electrolyzer_om_cost_musd = (
                pem_offshore.run(P_elec, RC_elec)
            )

            electrolyzer_total_capital_cost = (
                electrolyzer_capital_cost_musd * 1e6
            )  # convert from M USD to USD
            electrolyzer_OM_cost = (
                electrolyzer_om_cost_musd * 1e6
            )  # convert from M USD to USD

        else:
            raise (
                ValueError(
                    "Electrolyzer cost model must be one of['basic', 'singlitico2021'] but '%s' was given"
                    % (electrolyzer_cost_model)
                )
            )

    # package outputs for return
    electrolyzer_cost_results = {
        "electrolyzer_total_capital_cost": electrolyzer_total_capital_cost,
        "electrolyzer_OM_cost_annual": electrolyzer_OM_cost,
    }

    # print some results if desired
    if verbose:
        print("\nHydrogen Cost Results:")
        print(
            "Electrolyzer Total CAPEX $/kW: ",
            electrolyzer_total_capital_cost / (electrolyzer_size_mw * 1e3),
        )
        print(
            "Electrolyzer O&M $/kW: ",
            electrolyzer_OM_cost / (electrolyzer_size_mw * 1e3),
        )
        print(
            "Electrolyzer O&M $/kg: ",
            electrolyzer_OM_cost / H2_Results["hydrogen_annual_output"],
        )

    return electrolyzer_cost_results


def run_desal(
    plant_config, electrolyzer_physics_results, design_scenario, verbose=False
):
    if verbose:
        print("\n")
        print("Desal Results")

    if design_scenario["electrolyzer_location"] == "onshore":
        desal_results = {
            "feed_water_flowrat_m3perhr": 0,
            "desal_capex_usd": 0,
            "desal_opex_usd_per_year": 0,
            "power_for_desal_kw": 0,
            "fresh_water_capacity_m3_per_hour": 0,
            "equipment_mass_kg": 0,
            "equipment_footprint_m2": 0,
        }
    else:
        freshwater_kg_per_hr = np.mean(electrolyzer_physics_results["H2_Results"]["Water Hourly Consumption [kg/hr]"]) #to kg/hr

        if design_scenario["electrolyzer_location"] == "platform":
            (
                desal_capacity_m3_per_hour,
                feedwater_m3_per_hr,
                desal_power,
                desal_capex,
                desal_opex,
                desal_mass_kg,
                desal_size_m2,
            ) = RO_desal(freshwater_kg_per_hr, salinity="Seawater")

            # package outputs
            desal_results = {
                "fresh_water_flowrate_m3perhr": desal_capacity_m3_per_hour,
                "feed_water_flowrat_m3perhr": feedwater_m3_per_hr,
                "desal_capex_usd": desal_capex,
                "desal_opex_usd_per_year": desal_opex,
                "power_for_desal_kw": desal_power,
                "equipment_mass_kg": desal_mass_kg,
                "equipment_footprint_m2": desal_size_m2,
            }

            if verbose:
                print("Fresh water needed (m^3/hr): ", desal_capacity_m3_per_hour)

        elif design_scenario["electrolyzer_location"] == "turbine":
            nturbines = plant_config["technologies"]["wind"]["num_turbines"]

            # size for per-turbine desal #TODO consider using individual power generation time series from each turbine
            in_turb_freshwater_kg_per_hr = freshwater_kg_per_hr / nturbines

            (
                per_turb_desal_capacity_m3_per_hour,
                per_turb_feedwater_m3_per_hr,
                per_turb_desal_power,
                per_turb_desal_capex,
                per_turb_desal_opex,
                per_turb_desal_mass_kg,
                per_turb_desal_size_m2,
            ) = RO_desal(in_turb_freshwater_kg_per_hr, salinity="Seawater")

            fresh_water_flowrate = nturbines * per_turb_desal_capacity_m3_per_hour
            feed_water_flowrate = nturbines * per_turb_feedwater_m3_per_hr
            desal_capex = nturbines * per_turb_desal_capex
            desal_opex = nturbines * per_turb_desal_opex
            power_for_desal = nturbines * per_turb_desal_power

            # package outputs
            desal_results = {
                "fresh_water_flowrate_m3perhr": fresh_water_flowrate,
                "feed_water_flowrat_m3perhr": feed_water_flowrate,
                "desal_capex_usd": desal_capex,
                "desal_opex_usd_per_year": desal_opex,
                "power_for_desal_kw": power_for_desal,
                "per_turb_equipment_mass_kg": per_turb_desal_mass_kg,
                "per_turb_equipment_footprint_m2": per_turb_desal_size_m2,
            }

        if verbose:
            print(
                "Fresh water needed (m^3/hr): ",
                desal_results["fresh_water_flowrate_m3perhr"],
            )
            print("Requested fresh water (m^3/hr):", freshwater_kg_per_hr / 997)

    if verbose:
        for key in desal_results.keys():
            print("Average", key, " ", np.average(desal_results[key]))
        print("\n")

    return desal_results

def create_1MW_reference_PEM():
    pem_param_dict = {
        "eol_eff_percent_loss":10, 
        "uptime_hours_until_eol": 77600, 
        "include_degradation_penalty": True,
        "turndown_ratio":0.1,
    }
    pem = PEMClusters(cluster_size_mw = 1, 
        plant_life = 30, 
        **pem_param_dict)
    return pem

def get_electrolyzer_BOL_efficiency():
    pem_1MW = create_1MW_reference_PEM()
    bol_eff = pem_1MW.output_dict['BOL Efficiency Curve Info']['Efficiency [kWh/kg]'].values[-1]
    
    return np.round(bol_eff,2)

def size_electrolyzer_for_hydrogen_demand(hydrogen_production_capacity_required_kgphr,size_for = 'BOL',electrolyzer_degradation_power_increase = None):
    electrolyzer_energy_kWh_per_kg_estimate_BOL = get_electrolyzer_BOL_efficiency()
    if size_for == 'BOL':
        electrolyzer_capacity_MW = hydrogen_production_capacity_required_kgphr*electrolyzer_energy_kWh_per_kg_estimate_BOL/1000
    elif size_for == 'EOL':
        electrolyzer_energy_kWh_per_kg_estimate_EOL = electrolyzer_energy_kWh_per_kg_estimate_BOL*(1+electrolyzer_degradation_power_increase)
        electrolyzer_capacity_MW = hydrogen_production_capacity_required_kgphr*electrolyzer_energy_kWh_per_kg_estimate_EOL/1000
    
    return electrolyzer_capacity_MW

def check_capacity_based_on_clusters(electrolyzer_capacity_BOL_MW,cluster_cap_mw):

    if electrolyzer_capacity_BOL_MW % cluster_cap_mw  == 0:
        n_pem_clusters_max = electrolyzer_capacity_BOL_MW//cluster_cap_mw 
    else:
        n_pem_clusters_max = int(np.ceil(np.ceil(electrolyzer_capacity_BOL_MW)/cluster_cap_mw))
    electrolyzer_size_mw = n_pem_clusters_max*cluster_cap_mw
    return electrolyzer_size_mw