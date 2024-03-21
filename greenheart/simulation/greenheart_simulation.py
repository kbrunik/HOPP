# general imports
import os
from typing import Optional
import warnings
import numpy as np
import pandas as pd
from attrs import define, field

pd.options.mode.chained_assignment = None  # default='warn'

from greenheart.simulation.technologies.ammonia.ammonia import (
    run_ammonia_full_model,
)
from greenheart.simulation.technologies.steel.steel import (
    run_steel_full_model,
)

# visualization imports
import matplotlib.pyplot as plt

# HOPP imports
import greenheart.tools.eco.electrolysis as he_elec
import greenheart.tools.eco.finance as he_fin
import greenheart.tools.eco.hopp_mgmt as he_hopp
import greenheart.tools.eco.utilities as he_util
import greenheart.tools.eco.hydrogen_mgmt as he_h2

@define
class GreenHeartSimulationConfig:
    """
    Class to hold all the configuration parameters for the GreenHeart model

    Also sets up the HOPP, GreenHeart, ORBIT, and FLORIS configurations based on the
    input files and configuration parameters passed in.

    Args:
        filename_hopp_config (str): filename for the HOPP configuration
        filename_config.greenheart_config (str): filename for the GreenHeart configuration
        filename_turbine_config (str): filename for the turbine configuration
        filename_orbit_config (str): filename for the ORBIT configuration
        filename_floris_config (str): filename for the FLORIS configuration
        electrolyzer_rating_mw (Optional[float]): rating of the electrolyzer in MW
        solar_rating (Optional[float]): rating of the solar plant in MW
        battery_capacity_kw (Optional[float]): capacity of the battery in kW
        battery_capacity_kwh (Optional[float]): capacity of the battery in kWh
        wind_rating (Optional[float]): rating of the wind plant in MW
        verbose (bool): flag to print verbose output
        show_plots (bool): flag to show plots
        save_plots (bool): flag to save plots
        use_profast (bool): flag to use profast
        post_processing (bool): flag to run post processing
        storage_type (Optional[str]): type of storage
        incentive_option (int): incentive option
        plant_design_scenario (int): plant design scenario
        output_level (int): output level
        grid_connection (Optional[bool]): flag for grid connection
    """

    filename_hopp_config: str
    filename_greenheart_config: str
    filename_turbine_config: str
    filename_orbit_config: str
    filename_floris_config: str
    electrolyzer_rating_mw: Optional[float] = field(default=None)
    solar_rating: Optional[float] = field(default=None)
    battery_capacity_kw: Optional[float] = field(default=None)
    battery_capacity_kwh: Optional[float] = field(default=None)
    wind_rating: Optional[float] = field(default=None)
    verbose: bool = field(default=False)
    show_plots: bool = field(default=False)
    save_plots: bool = field(default=False)
    use_profast: bool = field(default=True)
    post_processing: bool = field(default=True)
    storage_type: Optional[str] = field(default=None)
    incentive_option: int = field(default=1)
    plant_design_scenario: int = field(default=1)
    output_level: int = field(default=1)
    grid_connection: Optional[bool] = field(default=None)

    # these are set in the __attrs_post_init__ method
    hopp_config: dict = field(init=False)
    greenheart_config: dict = field(init=False)
    orbit_config: dict = field(init=False)
    turbine_config: dict = field(init=False)
    floris_config: Optional[dict] = field(init=False)
    orbit_hybrid_electrical_export_config: dict = field(init=False)
    design_scenario: dict = field(init=False)

    def __attrs_post_init__(self):
        (
            self.hopp_config,
            self.greenheart_config,
            self.orbit_config,
            self.turbine_config,
            self.floris_config,
            self.orbit_hybrid_electrical_export_config,
        ) = he_util.get_inputs(
            self.filename_hopp_config,
            self.filename_greenheart_config,
            filename_orbit_config=self.filename_orbit_config,
            filename_floris_config=self.filename_floris_config,
            filename_turbine_config=self.filename_turbine_config,
            verbose=self.verbose,
            show_plots=self.show_plots,
            save_plots=self.save_plots,
        )

        # n scenarios, n discrete variables
        self.design_scenario = self.greenheart_config["plant_design"][
            "scenario%s" % (self.plant_design_scenario)
        ]
        self.design_scenario["id"] = self.plant_design_scenario

        # if design_scenario["h2_storage_location"] == "turbine":
        #     plant_config["h2_storage"]["type"] = "turbine"

        if self.electrolyzer_rating_mw != None:
            self.greenheart_config["electrolyzer"]["flag"] = True
            self.greenheart_config["electrolyzer"][
                "rating"
            ] = self.electrolyzer_rating_mw

        if self.solar_rating != None:
            self.hopp_config["site"]["solar"] = True
            self.hopp_config["technologies"]["pv"][
                "system_capacity_kw"
            ] = self.solar_rating

        if self.battery_capacity_kw != None:
            self.hopp_config["site"]["battery"]["flag"] = True
            self.hopp_config["technologies"]["battery"][
                "system_capacity_kw"
            ] = self.battery_capacity_kw

        if self.battery_capacity_kwh != None:
            self.hopp_config["site"]["battery"]["flag"] = True
            self.hopp_config["technologies"]["battery"][
                "system_capacity_kwh"
            ] = self.battery_capacity_kwh

        if self.storage_type != None:
            self.greenheart_config["h2_storage"]["type"] = self.storage_type

        if self.wind_rating != None:
            self.orbit_config["plant"]["capacity"] = int(self.wind_rating * 1e-3)
            self.orbit_config["plant"]["num_turbines"] = int(
                self.wind_rating * 1e-3 / self.turbine_config["turbine_rating"]
            )
            self.hopp_config["technologies"]["wind"]["num_turbines"] = (
                self.orbit_config["plant"]["num_turbines"]
            )
        
        if self.grid_connection != None:
            self.greenheart_config["project_parameters"][
                "grid_connection"
            ] = self.grid_connection
            if self.grid_connection:
                self.hopp_config["technologies"]["grid"]["interconnect_kw"] = (
                    self.orbit_config["plant"]["capacity"] * 1e6
                )

def run_simulation(config: GreenHeartSimulationConfig):
    # run orbit for wind plant construction and other costs
    ## TODO get correct weather (wind, wave) inputs for ORBIT input (possibly via ERA5)

    if config.design_scenario["wind_location"] == "offshore":
        
        if config.orbit_config["plant"]["num_turbines"] != config.hopp_config["technologies"]["wind"]["num_turbines"]:
            config.orbit_config["plant"].update(
                {"num_turbines": config.hopp_config["technologies"]["wind"]["num_turbines"]}
            )
            warnings.warn(f"'num_turbines' in the orbit_config was {config.orbit_config['plant']['num_turbines']}, but 'num_turbines' in" 
                    f"hopp_config was {config.hopp_config['technologies']['wind']['num_turbines']}. The value in the orbit_config"
                    "is being overwritten with the value from the hopp_config", UserWarning)
            
        if config.orbit_config["site"]["depth"] != config.greenheart_config["site"]["depth"]:
            config.orbit_config["site"].update(
                {"depth": config.greenheart_config["site"]["depth"]}
            )
            warnings.warn(f"site depth in the orbit_config was {config.orbit_config['site']['depth']}, but site depth in" 
                    f"greenheart_config was {config.greenheart_config['site']['depth']}. The value in the orbit_config"
                    "is being overwritten with the value from the greenheart_config", UserWarning)

        wind_config = he_fin.WindCostConfig(
            design_scenario=config.design_scenario,
            hopp_config=config.hopp_config,
            greenheart_config=config.greenheart_config,
            orbit_config=config.orbit_config,
            orbit_hybrid_electrical_export_config=config.orbit_hybrid_electrical_export_config,
        )

    if config.design_scenario["wind_location"] == "onshore":
        wind_config = he_fin.WindCostConfig(
            design_scenario=config.design_scenario,
            hopp_config=config.hopp_config,
            greenheart_config=config.greenheart_config,
            turbine_config=config.turbine_config,
        )

    wind_cost_results = he_fin.run_wind_cost_model(
        wind_cost_inputs=wind_config, verbose=config.verbose
    )
    # setup HOPP model
    hopp_config, hopp_site = he_hopp.setup_hopp(
        config.hopp_config,
        config.greenheart_config,
        config.orbit_config,
        config.turbine_config,
        wind_cost_results,
        config.floris_config,
        config.design_scenario,
        show_plots=config.show_plots,
        save_plots=config.save_plots,
    )

    # run HOPP model
    # hopp_results = he_hopp.run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, verbose=verbose)
    hopp_results = he_hopp.run_hopp(
        hopp_config,
        hopp_site,
        project_lifetime=config.greenheart_config["project_parameters"][
            "project_lifetime"
        ],
        verbose=config.verbose,
    )

    # this portion of the system is inside a function so we can use a solver to determine the correct energy availability for h2 production
    def energy_internals(
        hopp_results=hopp_results,
        wind_cost_results=wind_cost_results,
        design_scenario=config.design_scenario,
        orbit_config=config.orbit_config,
        hopp_config=hopp_config,
        greenheart_config=config.greenheart_config,
        turbine_config=config.turbine_config,
        wind_resource=hopp_site.wind_resource,
        verbose=config.verbose,
        show_plots=config.show_plots,
        save_plots=config.save_plots,
        solver=True,
        power_for_peripherals_kw_in=0.0,
        breakdown=False,
    ):

        hopp_results_internal = dict(hopp_results)

        # set energy input profile
        ### subtract peripheral power from supply to get what is left for electrolyzer
        remaining_power_profile_in = np.zeros_like(
            hopp_results["combined_hybrid_power_production_hopp"]
        )

        high_count = sum(
            np.asarray(hopp_results["combined_hybrid_power_production_hopp"])
            >= power_for_peripherals_kw_in
        )
        total_peripheral_energy = power_for_peripherals_kw_in * 365 * 24
        distributed_peripheral_power = total_peripheral_energy / high_count
        for i in range(len(hopp_results["combined_hybrid_power_production_hopp"])):
            r = (
                hopp_results["combined_hybrid_power_production_hopp"][i]
                - distributed_peripheral_power
            )
            if r > 0:
                remaining_power_profile_in[i] = r

        hopp_results_internal["combined_hybrid_power_production_hopp"] = tuple(
            remaining_power_profile_in
        )

        # run electrolyzer physics model
        electrolyzer_physics_results = he_elec.run_electrolyzer_physics(
            hopp_results_internal,
            config.greenheart_config,
            wind_resource,
            design_scenario,
            show_plots=show_plots,
            save_plots=save_plots,
            verbose=verbose,
        )

        # run electrolyzer cost model
        electrolyzer_cost_results = he_elec.run_electrolyzer_cost(
            electrolyzer_physics_results,
            hopp_config,
            config.greenheart_config,
            design_scenario,
            verbose=verbose,
        )

        desal_results = he_elec.run_desal(
            hopp_config, electrolyzer_physics_results, design_scenario, verbose
        )

        # run array system model
        h2_pipe_array_results = he_h2.run_h2_pipe_array(
            greenheart_config,
            hopp_config,
            turbine_config,
            wind_cost_results,
            electrolyzer_physics_results,
            design_scenario,
            verbose,
        )

        # compressor #TODO size correctly
        h2_transport_compressor, h2_transport_compressor_results = (
            he_h2.run_h2_transport_compressor(
                config.greenheart_config,
                electrolyzer_physics_results,
                design_scenario,
                verbose=verbose,
            )
        )

        # transport pipeline
        if design_scenario["wind_location"] == "offshore":
            h2_transport_pipe_results = he_h2.run_h2_transport_pipe(
                orbit_config,
                greenheart_config,
                electrolyzer_physics_results,
                design_scenario,
                verbose=verbose,
            )
        if design_scenario["wind_location"] == "onshore":
            h2_transport_pipe_results = {
                "total capital cost [$]": [0 * 5433290.0184895478],
                "annual operating cost [$]": [0.0],
            }

        # pressure vessel storage
        pipe_storage, h2_storage_results = he_h2.run_h2_storage(
            hopp_config,
            greenheart_config,
            turbine_config,
            electrolyzer_physics_results,
            design_scenario,
            verbose=verbose,
        )

        total_energy_available = np.sum(
            hopp_results["combined_hybrid_power_production_hopp"]
        )

        ### get all energy non-electrolyzer usage in kw
        desal_power_kw = desal_results["power_for_desal_kw"]

        h2_transport_compressor_power_kw = h2_transport_compressor_results[
            "compressor_power"
        ]  # kW

        h2_storage_energy_kwh = h2_storage_results["storage_energy"]
        h2_storage_power_kw = h2_storage_energy_kwh * (1.0 / (365 * 24))

        # if transport is not HVDC and h2 storage is on shore, then power the storage from the grid
        if (design_scenario["transportation"] == "pipeline") and (
            design_scenario["h2_storage_location"] == "onshore"
        ):
            total_accessory_power_renewable_kw = (
                desal_power_kw + h2_transport_compressor_power_kw
            )
            total_accessory_power_grid_kw = h2_storage_power_kw
        else:
            total_accessory_power_renewable_kw = (
                desal_power_kw + h2_transport_compressor_power_kw + h2_storage_power_kw
            )
            total_accessory_power_grid_kw = 0.0

        ### subtract peripheral power from supply to get what is left for electrolyzer and also get grid power
        remaining_power_profile = np.zeros_like(
            hopp_results["combined_hybrid_power_production_hopp"]
        )
        grid_power_profile = np.zeros_like(
            hopp_results["combined_hybrid_power_production_hopp"]
        )
        for i in range(len(hopp_results["combined_hybrid_power_production_hopp"])):
            r = (
                hopp_results["combined_hybrid_power_production_hopp"][i]
                - total_accessory_power_renewable_kw
            )
            grid_power_profile[i] = total_accessory_power_grid_kw
            if r > 0:
                remaining_power_profile[i] = r

        if verbose and not solver:
            print("\nEnergy/Power Results:")
            print("Supply (MWh): ", total_energy_available)
            print("Desal (kW): ", desal_power_kw)
            print("Transport compressor (kW): ", h2_transport_compressor_power_kw)
            print("Storage compression, refrigeration, etc (kW): ", h2_storage_power_kw)
            # print("Difference: ", total_energy_available/(365*24) - np.sum(remaining_power_profile)/(365*24) - total_accessory_power_renewable_kw)

        if (show_plots or save_plots) and not solver:
            fig, ax = plt.subplots(1)
            plt.plot(
                np.asarray(hopp_results["combined_hybrid_power_production_hopp"])
                * 1e-6,
                label="Total Energy Available",
            )
            plt.plot(
                remaining_power_profile * 1e-6,
                label="Energy Available for Electrolysis",
            )
            plt.xlabel("Hour")
            plt.ylabel("Power (GW)")
            plt.tight_layout()
            if save_plots:
                savepath = "figures/power_series/"
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                plt.savefig(
                    savepath + "power_%i.png" % (design_scenario["id"]),
                    transparent=True,
                )
            if show_plots:
                plt.show()
        if solver:
            if breakdown:
                return (
                    total_accessory_power_renewable_kw,
                    total_accessory_power_grid_kw,
                    desal_power_kw,
                    h2_transport_compressor_power_kw,
                    h2_storage_power_kw,
                    remaining_power_profile,
                )
            else:
                return total_accessory_power_renewable_kw
        else:
            return (
                electrolyzer_physics_results,
                electrolyzer_cost_results,
                desal_results,
                h2_pipe_array_results,
                h2_transport_compressor,
                h2_transport_compressor_results,
                h2_transport_pipe_results,
                pipe_storage,
                h2_storage_results,
                total_accessory_power_renewable_kw,
                total_accessory_power_grid_kw,
                remaining_power_profile,
            )

    # define function to provide to the brent solver
    def energy_residual_function(power_for_peripherals_kw_in):

        # get results for current design
        power_for_peripherals_kw_out = energy_internals(
            power_for_peripherals_kw_in=power_for_peripherals_kw_in,
            solver=True,
            verbose=False,
        )

        # collect residual
        power_residual = power_for_peripherals_kw_out - power_for_peripherals_kw_in

        return power_residual

    def simple_solver(initial_guess=0.0):

        # get results for current design
        (
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            desal_power_kw,
            h2_transport_compressor_power_kw,
            h2_storage_power_kw,
            remaining_power_profile,
        ) = energy_internals(
            power_for_peripherals_kw_in=initial_guess,
            solver=True,
            verbose=False,
            breakdown=True,
        )

        return (
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            desal_power_kw,
            h2_transport_compressor_power_kw,
            h2_storage_power_kw,
        )

    #################### solving for energy needed for non-electrolyzer components ####################################
    # this approach either exactly over over-estimates the energy needed for non-electrolyzer components
    solver_results = simple_solver(0)
    solver_result = solver_results[0]

    # # this is a check on the simple solver
    # print("\nsolver result: ", solver_result)
    # residual = energy_residual_function(solver_result)
    # print("\nresidual: ", residual)

    # this approach exactly sizes the energy needed for the non-electrolyzer components (according to the current models anyway)
    # solver_result = optimize.brentq(energy_residual_function, -10, 20000, rtol=1E-5)
    # OptimizeResult = optimize.root(energy_residual_function, 11E3, tol=1)
    # solver_result = OptimizeResult.x
    # solver_results = simple_solver(solver_result)
    # solver_result = solver_results[0]
    # print(solver_result)

    ##################################################################################################################

    # get results for final design
    (
        electrolyzer_physics_results,
        electrolyzer_cost_results,
        desal_results,
        h2_pipe_array_results,
        h2_transport_compressor,
        h2_transport_compressor_results,
        h2_transport_pipe_results,
        pipe_storage,
        h2_storage_results,
        total_accessory_power_renewable_kw,
        total_accessory_power_grid_kw,
        remaining_power_profile,
    ) = energy_internals(solver=False, power_for_peripherals_kw_in=solver_result)

    ## end solver loop here
    platform_results = he_h2.run_equipment_platform(
        hopp_config,
        config.greenheart_config,
        config.orbit_config,
        config.design_scenario,
        hopp_results,
        electrolyzer_physics_results,
        h2_storage_results,
        desal_results,
        verbose=config.verbose,
    )

    ################# OSW intermediate calculations" aka final financial calculations
    # does LCOE even make sense if we are only selling the H2? I think in this case LCOE should not be used, rather LCOH should be used. Or, we could use LCOE based on the electricity actually used for h2
    # I think LCOE is just being used to estimate the cost of the electricity used, but in this case we should just use the cost of the electricity generating plant since we are not selling to the grid. We
    # could build in a grid connection later such that we use LCOE for any purchased electricity and sell any excess electricity after H2 production
    # actually, I think this is what OSW is doing for LCOH

    # TODO double check full-system CAPEX
    capex, capex_breakdown = he_fin.run_capex(
        hopp_results,
        wind_cost_results,
        electrolyzer_cost_results,
        h2_pipe_array_results,
        h2_transport_compressor_results,
        h2_transport_pipe_results,
        h2_storage_results,
        hopp_config,
        config.greenheart_config,
        config.design_scenario,
        desal_results,
        platform_results,
        verbose=config.verbose,
    )

    # TODO double check full-system OPEX
    opex_annual, opex_breakdown_annual = he_fin.run_opex(
        hopp_results,
        wind_cost_results,
        electrolyzer_cost_results,
        h2_pipe_array_results,
        h2_transport_compressor_results,
        h2_transport_pipe_results,
        h2_storage_results,
        hopp_config,
        config.greenheart_config,
        desal_results,
        platform_results,
        verbose=config.verbose,
        total_export_system_cost=capex_breakdown["electrical_export_system"],
    )

    if config.verbose:
        print(
            "hybrid plant capacity factor: ",
            np.sum(hopp_results["combined_hybrid_power_production_hopp"])
            / (hopp_results["hybrid_plant"].system_capacity_kw.hybrid * 365 * 24),
        )

    steel_finance = None
    ammonia_finance = None

    if config.use_profast:
        lcoe, pf_lcoe = he_fin.run_profast_lcoe(
            config.greenheart_config,
            wind_cost_results,
            capex_breakdown,
            opex_breakdown_annual,
            hopp_results,
            config.incentive_option,
            config.design_scenario,
            verbose=config.verbose,
            show_plots=config.show_plots,
            save_plots=config.save_plots,
        )
        lcoh_grid_only, pf_grid_only = he_fin.run_profast_grid_only(
            config.greenheart_config,
            wind_cost_results,
            electrolyzer_physics_results,
            capex_breakdown,
            opex_breakdown_annual,
            hopp_results,
            config.design_scenario,
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            verbose=config.verbose,
            show_plots=config.show_plots,
            save_plots=config.save_plots,
        )
        lcoh, pf_lcoh = he_fin.run_profast_full_plant_model(
            config.greenheart_config,
            wind_cost_results,
            electrolyzer_physics_results,
            capex_breakdown,
            opex_breakdown_annual,
            hopp_results,
            config.incentive_option,
            config.design_scenario,
            total_accessory_power_renewable_kw,
            total_accessory_power_grid_kw,
            verbose=config.verbose,
            show_plots=config.show_plots,
            save_plots=config.save_plots,
        )

        hydrogen_amount_kgpy = electrolyzer_physics_results["H2_Results"][
            "hydrogen_annual_output"
        ]

        if "steel" in config.greenheart_config:
            if config.verbose:
                print("Running steel\n")

            # use lcoh from the electrolyzer model if it is not already in the config
            if "lcoh" not in config.greenheart_config["steel"]["finances"]:
                config.greenheart_config["steel"]["finances"]["lcoh"] = lcoh

            # use lcoh from the electrolyzer model if it is not already in the config
            if "lcoh" not in config.greenheart_config["steel"]["costs"]:
                config.greenheart_config["steel"]["costs"]["lcoh"] = lcoh

            # use the hydrogen amount from the electrolyzer physics model if it is not already in the config
            if (
                "hydrogen_amount_kgpy"
                not in config.greenheart_config["steel"]["capacity"]
            ):
                config.greenheart_config["steel"]["capacity"][
                    "hydrogen_amount_kgpy"
                ] = hydrogen_amount_kgpy

            _, _, steel_finance = run_steel_full_model(config.greenheart_config)

        if "ammonia" in config.greenheart_config:
            if config.verbose:
                print("Running ammonia\n")

            # use the hydrogen amount from the electrolyzer physics model if it is not already in the config
            if (
                "hydrogen_amount_kgpy"
                not in config.greenheart_config["ammonia"]["capacity"]
            ):
                config.greenheart_config["ammonia"]["capacity"][
                    "hydrogen_amount_kgpy"
                ] = hydrogen_amount_kgpy

            _, _, ammonia_finance = run_ammonia_full_model(config.greenheart_config)

    ################# end OSW intermediate calculations
    if config.post_processing:
        power_breakdown = he_util.post_process_simulation(
            lcoe,
            lcoh,
            pf_lcoh,
            pf_lcoe,
            hopp_results,
            electrolyzer_physics_results,
            hopp_config,
            config.greenheart_config,
            config.orbit_config,
            h2_storage_results,
            capex_breakdown,
            opex_breakdown_annual,
            wind_cost_results,
            platform_results,
            desal_results,
            config.design_scenario,
            config.plant_design_scenario,
            config.incentive_option,
            solver_results=solver_results,
            show_plots=config.show_plots,
            save_plots=config.save_plots,
            verbose=config.verbose,
        )  # , lcoe, lcoh, lcoh_with_grid, lcoh_grid_only)

    # return
    if config.output_level == 0:
        return 0
    elif config.output_level == 1:
        return lcoh
    elif config.output_level == 2:
        return (
            lcoh,
            lcoe,
            capex_breakdown,
            opex_breakdown_annual,
            pf_lcoh,
            electrolyzer_physics_results,
        )
    elif config.output_level == 3:
        return (
            lcoh,
            lcoe,
            capex_breakdown,
            opex_breakdown_annual,
            pf_lcoh,
            electrolyzer_physics_results,
            pf_lcoe,
            power_breakdown,
        )
    elif config.output_level == 4:
        return lcoe, lcoh, lcoh_grid_only
    elif config.output_level == 5:
        return lcoe, lcoh, lcoh_grid_only, hopp_results["hopp_interface"]
    elif config.output_level == 6:
        return hopp_results, electrolyzer_physics_results, remaining_power_profile
    elif config.output_level == 7:
        return lcoe, lcoh, steel_finance, ammonia_finance


def run_sweeps(simulate=False, verbose=True, show_plots=True, use_profast=True):

    if simulate:
        verbose = False
        show_plots = False
    if simulate:
        storage_types = ["none", "pressure_vessel", "pipe", "salt_cavern"]
        wind_ratings = [400]  # , 800, 1200] #[200, 400, 600, 800]

        for wind_rating in wind_ratings:
            ratings = np.linspace(
                round(0.2 * wind_rating, ndigits=0), 2 * wind_rating + 1, 50
            )
            for storage_type in storage_types:
                lcoh_array = np.zeros(len(ratings))
                for z in np.arange(0, len(ratings)):
                    lcoh_array[z] = run_simulation(
                        electrolyzer_rating_mw=ratings[z],
                        wind_rating=wind_rating,
                        verbose=verbose,
                        show_plots=show_plots,
                        use_profast=use_profast,
                        storage_type=storage_type,
                    )
                    print(lcoh_array)
                np.savetxt(
                    "data/lcoh_vs_rating_%s_storage_%sMWwindplant.txt"
                    % (storage_type, wind_rating),
                    np.c_[ratings, lcoh_array],
                )

    if show_plots:

        wind_ratings = [400, 800, 1200]  # [200, 400, 600, 800]
        indexes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 6))

        for i in np.arange(0, len(wind_ratings)):
            wind_rating = wind_ratings[i]
            data_no_storage = np.loadtxt(
                "data/lcoh_vs_rating_none_storage_%sMWwindplant.txt" % (wind_rating)
            )
            data_pressure_vessel = np.loadtxt(
                "data/lcoh_vs_rating_pressure_vessel_storage_%sMWwindplant.txt"
                % (wind_rating)
            )
            data_salt_cavern = np.loadtxt(
                "data/lcoh_vs_rating_salt_cavern_storage_%sMWwindplant.txt"
                % (wind_rating)
            )
            data_pipe = np.loadtxt(
                "data/lcoh_vs_rating_pipe_storage_%sMWwindplant.txt" % (wind_rating)
            )

            ax[indexes[i]].plot(
                data_pressure_vessel[:, 0] / wind_rating,
                data_pressure_vessel[:, 1],
                label="Pressure Vessel",
            )
            ax[indexes[i]].plot(
                data_pipe[:, 0] / wind_rating, data_pipe[:, 1], label="Underground Pipe"
            )
            ax[indexes[i]].plot(
                data_salt_cavern[:, 0] / wind_rating,
                data_salt_cavern[:, 1],
                label="Salt Cavern",
            )
            ax[indexes[i]].plot(
                data_no_storage[:, 0] / wind_rating,
                data_no_storage[:, 1],
                "--k",
                label="No Storage",
            )

            ax[indexes[i]].scatter(
                data_pressure_vessel[np.argmin(data_pressure_vessel[:, 1]), 0]
                / wind_rating,
                np.min(data_pressure_vessel[:, 1]),
                color="k",
            )
            ax[indexes[i]].scatter(
                data_pipe[np.argmin(data_pipe[:, 1]), 0] / wind_rating,
                np.min(data_pipe[:, 1]),
                color="k",
            )
            ax[indexes[i]].scatter(
                data_salt_cavern[np.argmin(data_salt_cavern[:, 1]), 0] / wind_rating,
                np.min(data_salt_cavern[:, 1]),
                color="k",
            )
            ax[indexes[i]].scatter(
                data_no_storage[np.argmin(data_no_storage[:, 1]), 0] / wind_rating,
                np.min(data_no_storage[:, 1]),
                color="k",
                label="Optimal ratio",
            )

            ax[indexes[i]].legend(frameon=False, loc="best")

            ax[indexes[i]].set_xlim([0.2, 2.0])
            ax[indexes[i]].set_ylim([0, 25])

            ax[indexes[i]].annotate("%s MW Wind Plant" % (wind_rating), (0.6, 1.0))

        ax[1, 0].set_xlabel("Electrolyzer/Wind Plant Rating Ratio")
        ax[1, 1].set_xlabel("Electrolyzer/Wind Plant Rating Ratio")
        ax[0, 0].set_ylabel("LCOH ($/kg)")
        ax[1, 0].set_ylabel("LCOH ($/kg)")

        plt.tight_layout()
        plt.savefig("lcoh_vs_rating_ratio.pdf", transparent=True)
        plt.show()

    return 0


def run_policy_options_storage_types(
    verbose=True, show_plots=False, save_plots=False, use_profast=True
):

    storage_types = ["pressure_vessel", "pipe", "salt_cavern", "none"]
    policy_options = [1, 2, 3, 4, 5, 6, 7]

    lcoh_array = np.zeros((len(storage_types), len(policy_options)))
    for i, storage_type in enumerate(storage_types):
        for j, poption in enumerate(policy_options):
            lcoh_array[i, j] = run_simulation(
                storage_type=storage_type,
                incentive_option=poption,
                verbose=verbose,
                show_plots=show_plots,
                use_profast=use_profast,
            )
        print(lcoh_array)

    savepath = "results/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.savetxt(
        savepath + "lcoh-with-policy.txt",
        np.c_[np.round(lcoh_array, decimals=2)],
        header="rows: %s, columns: %s"
        % ("".join(storage_types), "".join(str(p) for p in policy_options)),
        fmt="%.2f",
    )

    return 0


def run_policy_storage_design_options(
    verbose=False, show_plots=False, save_plots=False, use_profast=True
):

    design_scenarios = [1, 2, 3, 4, 5, 6, 7]
    policy_options = [1, 2, 3, 4, 5, 6, 7]
    storage_types = ["pressure_vessel", "pipe", "salt_cavern", "none"]

    design_series = []
    policy_series = []
    storage_series = []
    lcoh_series = []
    lcoe_series = []
    electrolyzer_capacity_factor_series = []
    annual_energy_breakdown_series = {
        "design": [],
        "policy": [],
        "storage": [],
        "wind_kwh": [],
        "renewable_kwh": [],
        "grid_power_kwh": [],
        "electrolyzer_kwh": [],
        "desal_kwh": [],
        "h2_transport_compressor_power_kwh": [],
        "h2_storage_power_kwh": [],
    }

    lcoh_array = np.zeros((len(design_scenarios), len(policy_options)))
    for i, design in enumerate(design_scenarios):
        for j, policy in enumerate(policy_options):
            for storage in storage_types:
                if storage != "pressure_vessel":  # and storage != "none"):
                    if design != 1 and design != 5 and design != 7:
                        print("skipping: ", design, " ", policy, " ", storage)
                        continue
                design_series.append(design)
                policy_series.append(policy)
                storage_series.append(storage)
                (
                    lcoh,
                    lcoe,
                    capex_breakdown,
                    opex_breakdown_annual,
                    pf_lcoh,
                    electrolyzer_physics_results,
                    pf_lcoe,
                    annual_energy_breakdown,
                ) = run_simulation(
                    storage_type=storage,
                    plant_design_scenario=design,
                    incentive_option=policy,
                    verbose=verbose,
                    show_plots=show_plots,
                    use_profast=use_profast,
                    output_level=3,
                )
                lcoh_series.append(lcoh)
                lcoe_series.append(lcoe)
                electrolyzer_capacity_factor_series.append(
                    electrolyzer_physics_results["capacity_factor"]
                )

                annual_energy_breakdown_series["design"].append(design)
                annual_energy_breakdown_series["policy"].append(policy)
                annual_energy_breakdown_series["storage"].append(storage)
                for key in annual_energy_breakdown.keys():
                    annual_energy_breakdown_series[key].append(
                        annual_energy_breakdown[key]
                    )

    savepath = "data/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    df = pd.DataFrame.from_dict(
        {
            "Design": design_series,
            "Storage": storage_series,
            "Policy": policy_series,
            "LCOH [$/kg]": lcoh_series,
            "LCOE [$/kWh]": lcoe_series,
            "Electrolyzer capacity factor": electrolyzer_capacity_factor_series,
        }
    )
    df.to_csv(savepath + "design-storage-policy-lcoh.csv")

    df_energy = pd.DataFrame.from_dict(annual_energy_breakdown_series)
    df_energy.to_csv(savepath + "annual_energy_breakdown.csv")
    return 0


def run_design_options(
    verbose=False, show_plots=False, save_plots=False, incentive_option=1
):

    design_options = range(1, 8)  # 8
    scenario_lcoh = []
    scenario_lcoe = []
    scenario_capex_breakdown = []
    scenario_opex_breakdown_annual = []
    scenario_pf = []
    scenario_electrolyzer_physics = []

    for design in design_options:
        (
            lcoh,
            lcoe,
            capex_breakdown,
            opex_breakdown_annual,
            pf,
            electrolyzer_physics_results,
        ) = run_simulation(
            verbose=verbose,
            show_plots=show_plots,
            use_profast=True,
            incentive_option=incentive_option,
            plant_design_scenario=design,
            output_level=2,
        )
        scenario_lcoh.append(lcoh)
        scenario_lcoe.append(lcoe)
        scenario_capex_breakdown.append(capex_breakdown)
        scenario_opex_breakdown_annual.append(opex_breakdown_annual)
        scenario_pf.append(pf)
        scenario_electrolyzer_physics.append(electrolyzer_physics_results)
    df_aggregate = pd.DataFrame.from_dict(
        {
            "Design": [int(x) for x in design_options],
            "LCOH [$/kg]": scenario_lcoh,
            "LCOE [$/kWh]": scenario_lcoe,
        }
    )
    df_capex = pd.DataFrame(scenario_capex_breakdown)
    df_opex = pd.DataFrame(scenario_opex_breakdown_annual)

    df_capex.insert(0, "Design", design_options)
    df_opex.insert(0, "Design", design_options)

    # df_aggregate = df_aggregate.transpose()
    df_capex = df_capex.transpose()
    df_opex = df_opex.transpose()

    results_path = "./combined_results/"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    df_aggregate.to_csv(results_path + "metrics.csv")
    df_capex.to_csv(results_path + "capex.csv")
    df_opex.to_csv(results_path + "opex.csv")
    return 0


def run_storage_options():
    storage_types = ["pressure_vessel", "pipe", "salt_cavern", "none"]
    lcoe_list = []
    lcoh_list = []
    lcoh_with_grid_list = []
    lcoh_grid_only_list = []
    for storage_type in storage_types:
        lcoe, lcoh, _ = run_simulation(
            verbose=False,
            show_plots=False,
            save_plots=False,
            use_profast=True,
            incentive_option=1,
            plant_design_scenario=1,
            storage_type=storage_type,
            output_level=4,
            grid_connection=False,
        )
        lcoe_list.append(lcoe)
        lcoh_list.append(lcoh)

        # with grid
        _, lcoh_with_grid, lcoh_grid_only = run_simulation(
            verbose=False,
            show_plots=False,
            save_plots=False,
            use_profast=True,
            incentive_option=1,
            plant_design_scenario=1,
            storage_type=storage_type,
            output_level=4,
            grid_connection=True,
        )
        lcoh_with_grid_list.append(lcoh_with_grid)
        lcoh_grid_only_list.append(lcoh_grid_only)

    data_dict = {
        "Storage Type": storage_types,
        "LCOE [$/MW]": np.asarray(lcoe_list) * 1e3,
        "LCOH [$/kg]": lcoh_list,
        "LCOH with Grid [$/kg]": lcoh_with_grid_list,
        "LCOH Grid Only [$/kg]": lcoh_grid_only_list,
    }
    df = pd.DataFrame.from_dict(data_dict)

    savepath = "data/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    df.to_csv(savepath + "storage-types-and-matrics.csv")
    return 0
