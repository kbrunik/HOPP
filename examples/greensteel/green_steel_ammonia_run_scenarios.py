import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import hopp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import math

from hopp.simulation.technologies.sites import SiteInfo
from hopp.utilities.keys import set_developer_nrel_gov_key

from hopp.to_organize.H2_Analysis.hopp_for_h2 import hopp_for_h2
from hopp.to_organize.H2_Analysis.run_h2a import run_h2a as run_h2a
from hopp.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import numpy as np
from lcoe.lcoe import lcoe as lcoe_calc
import warnings
warnings.filterwarnings("ignore")

#import hopp.to_organize.hopp_tools_steel as hopp_tools_steel
import hopp_tools_steel
import hopp.to_organize.inputs_py as inputs_py
import copy
from hopp.to_organize.hopp_tools_steel import hoppDict
import hopp.to_organize.run_RODeO as run_RODeO
#import hopp.to_organize.run_profast_for_hydrogen as run_profast_for_hydrogen
import run_profast_for_hydrogen
#import hopp.to_organize.distributed_pipe_cost_analysis as distributed_pipe_cost_analysis
import distributed_pipe_cost_analysis
import hopp.to_organize.H2_Analysis.LCA_single_scenario as LCA_single_scenario
from green_steel_ammonia_solar_parametric_sweep import solar_storage_param_sweep
from grid_price_profiles import grid_price_interpolation

def batch_generator_kernel(arg_list):

    # Read in arguments
    [
        policy,
        i,
        atb_year,
        site_location,
        electrolysis_scale,
        run_RODeO_selector,
        floris,
        grid_connection_scenario,
        grid_price_scenario,
        direct_coupling,
        electrolyzer_cost_case,
        electrolyzer_degradation_power_increase,
        wind_plant_degradation_power_decrease,
        steel_annual_production_rate_target_tpy,
        project_path,
        results_dir,
        fin_sum_dir,
        energy_profile_dir,
        price_breakdown_dir,
        rodeo_output_dir,
        floris_dir,
        path,
        save_hybrid_plant_yaml,
        save_model_input_yaml,
        save_model_output_yaml,
        number_pem_stacks,
        run_pv_battery_sweep,
        electrolyzer_degradation_penalty,
        pem_control_type,
        storage_capacity_multiplier,
        solar_ITC,
        grid_price_filename
    ] = arg_list

    hopp_path = os.path.dirname(os.path.abspath(hopp.__file__))

    from hopp.simulation.technologies.sites import flatirons_site as sample_site # For some reason we have to pull this inside the definition

    """
    Perform a LCOH analysis for an offshore wind + Hydrogen PEM system

    Missing Functionality:
    1. Figure out H2A Costs or ammortize cost/kw electrolyzer figure and add opex

    ~1. Offshore wind site locations and cost details (4 sites, $1300/kw capex + BOS cost which will come from Orbit Runs)~

    2. Cost Scaling Based on Year (Have Weiser et. al report with cost scaling for fixed and floating tech, will implement)
    3. Cost Scaling Based on Plant Size (Shields et. Al report)
    4. Integration Required:
    * Pressure Vessel Model~
    * HVDC Model
    * Pipeline Model

    5. Model Development Required:
    - Floating Electrolyzer Platform
    """

    #Set API key
    load_dotenv()
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env

    #Step 1: User Inputs for scenario
    resource_year = 2013

    sample_site['year'] = resource_year
    useful_life = 30
    critical_load_factor = 1
    run_reopt_flag = False
    custom_powercurve = True    #A flag that is applicable when using PySam WindPower (not FLORIS)
    storage_used = False
    battery_can_grid_charge = False
    grid_connected_hopp = False

    # grid_connected_rodeo = False
    #run_RODeO_selector = False
    user_defined_electrolyzer_EOL_eff_drop = True
    EOL_eff_drop = electrolyzer_degradation_power_increase*100
    user_defined_electrolyzer_BOL_kWh_per_kg = False
    BOL_kWh_per_kg = []
    #the electrolyzer_model_parameters are not fully tested, please
    #don't try to enable yet!
    electrolyzer_model_parameters ={
    'Modify BOL Eff':user_defined_electrolyzer_BOL_kWh_per_kg,
    'BOL Eff [kWh/kg-H2]':BOL_kWh_per_kg,
    'Modify EOL Degradation Value':user_defined_electrolyzer_EOL_eff_drop,
    'EOL Rated Efficiency Drop':EOL_eff_drop}
    # Technology sizing
    #interconnection_size_mw = wind_size_mw
    #electrolyzer_size_mw = wind_size_mw



    #solar and battery size list will be used in param sweep if
    #param swee is true
    ##Solar and Battery Parametric Sweep Inputs
    #solar_sizes_mw=[750]
    #storage_sizes_mw=[0]
    #storage_sizes_mwh = [0]
    if grid_connection_scenario == 'off-grid':
        solar_sizes_mw=[0,100,250,500,750]
        storage_sizes_mw=[0,100,100,200]
        storage_sizes_mwh = [0,100,400,400]
        #storage_sizes_mw=[0,100,100,200]
        #storage_sizes_mwh = [0,100,400,400]
    else:
        solar_sizes_mw = [0,100,250,500]
        storage_sizes_mw=[0]
        storage_sizes_mwh = [0]
        #storage_sizes_mw = [0,50,50,100]
        #storage_sizes_mwh = [0,50,200,200]

    save_param_sweep_general_info=False
    save_param_sweep_best_case=False
    #THESE ARE WORKING VARIABLES NOW
    solar_size_mw = 0
    storage_size_mw = 0
    storage_size_mwh = 0
    battery_for_minimum_electrolyzer_op=True#If true, then dispatch battery (if on) to supply minimum power for operation to PEM, otherwise use it for rated PEM power

    if electrolyzer_degradation_penalty==True:
        user_defined_stack_replacement_time = False#if true then not dependent on pem performance and set to constant
    else:
        user_defined_stack_replacement_time = True
    use_optimistic_pem_efficiency = False
    if electrolysis_scale=='Centralized':
        default_n_pem_clusters=25
    else:
        default_n_pem_clusters = 1
    if number_pem_stacks == 'None':
        n_pem_clusters = default_n_pem_clusters
    else:
        n_pem_clusters = number_pem_stacks
    scenario_choice = 'Green Steel Ammonia Analysis'

    scenario = dict()

    #Site lat and lon will be set by data loaded from Orbit runs

    # Financial inputs
    discount_rate = 0.07
    debt_equity_split = 60

    # Wind costs input from ORBIT analysis
    h2_model ='Simple'  #Basic cost model based on H2a and HFTO program record for PEM electrolysis
    # h2_model = 'H2A'

    # These inputs are not used in this analysis (no solar or storage)
    solar_cost_kw = 9999 #THESE ARE OVERWRITTEN LATER
    solar_om_cost_kw=9999

    filename_st_xl = os.path.join(project_path, 'H2_Analysis', 'storage_costs_ATB.csv')
    st_xl=pd.read_csv(filename_st_xl,index_col=0)
    storage_costs=st_xl[str(atb_year)]
    storage_cost_kwh=storage_costs['Battery Energy Capital Cost ($/kWh)']
    storage_cost_kw=storage_costs['Battery Power Capital Cost ($/kW)']
    storage_om_percent = 0.025 #percent of capex
    renewable_plant_cost = {}



    # Flags (TODO: remove them and update documentation)
    forced_sizes = True
    force_electrolyzer_cost = False


    # Enable Ability to purchase/sell electricity to/from grid. Price Defined in $/kWh
    if grid_connection_scenario != "off-grid":
        sell_price = 0.025
        buy_price = 0.025
    else:
        sell_price = False
        buy_price = False

    #print('Parent path = ', parent_path)

    # Site specific turbine information
    xl = pd.ExcelFile(path)

    save_outputs_dict = inputs_py.establish_save_output_dict()
    save_all_runs = list()

    # which plots to show
    plot_power_production = False
    plot_battery = False
    plot_grid = False
    plot_h2 = False
    plot_desal = False
    plot_wind = False
    plot_hvdcpipe = False
    plot_hvdcpipe_lcoh = False



    # Read in gams exe and license location
    # Create a .txt file in notepad with the locations of the gams .exe file, the .gms RODeO
    # version that you want to use, and the location of the gams license file. The text
    # should look something like this:
    # "C:\\GAMS\\win64\\24.8\\gams.exe" ..\\RODeO\\Storage_dispatch_SCS license=C:\\GAMS\\win64\\24.8\\gamslice.txt
    # Do not push this file to the remote repository because it will be different for every user
    # and for every machine, depending on what version of gams they are using and where it is installed
    if run_RODeO_selector == True:
        with open('gams_exe_license_locations.txt') as f:
            gams_locations_rodeo_version = f.readlines()
        f.close()
    else:
        gams_locations_rodeo_version = "none"
    

    hopp_dict = hoppDict(save_model_input_yaml, save_model_output_yaml)

    sub_dict = {
        'policy': policy[i],
        'atb_year': atb_year,
        'site_location': site_location,
        'parent_path': project_path,
        # 'load': load,
        #'kw_continuous': kw_continuous,
        'sample_site': sample_site,
        'discount_rate': discount_rate,
        'forced_sizes': forced_sizes,
        'force_electrolyzer_cost': force_electrolyzer_cost,
        #'wind_size': wind_size_mw,
        'solar_size': solar_size_mw,
        'storage_size_mw': storage_size_mw,
        'storage_size_mwh': storage_size_mwh,
        'solar_cost_kw': solar_cost_kw,
        'storage_cost_kw': storage_cost_kw,
        'storage_cost_kwh': storage_cost_kwh,
        'debt_equity_split': debt_equity_split,
        'useful_life': useful_life,
        'critical_load_factor': critical_load_factor,
        'run_reopt_flag': run_reopt_flag,
        'custom_powercurve': custom_powercurve,
        'storage_used': storage_used,
        'battery_can_grid_charge': battery_can_grid_charge,
        'grid_connected_hopp': grid_connected_hopp,
        #'interconnection_size_mw': interconnection_size_mw,
        #'electrolyzer_size_mw': electrolyzer_size_mw,
        'scenario':
            {
                'Useful Life': useful_life,
                'Debt Equity': debt_equity_split,
                'discount_rate': discount_rate,
            },
        'sell_price': sell_price,
        'buy_price': buy_price,
        'h2_model': h2_model,
        'results_dir': results_dir,
        'scenario_choice': scenario_choice,
    }

    hopp_dict.add('Configuration', sub_dict)

    plot_dict = {
        'plot':
            {
                'plot_power_production': False,
                'plot_battery': False,
                'plot_grid': False,
                'plot_h2': False,
                'plot_desal': True,
                'plot_wind': True,
                'plot_hvdcpipe': True,
                'plot_hvdcpipe_lcoh': True,
            }
    }

    hopp_dict.add('Configuration', plot_dict)

    # set policy values
    hopp_dict, scenario, policy_option = hopp_tools_steel.set_policy_values(hopp_dict, scenario, policy, i)
    #print(scenario['Wind PTC'])

    scenario_df = xl.parse()
    scenario_df.set_index(["Parameter"], inplace = True)

    site_df = scenario_df[site_location]

    turbine_model = str(site_df['Turbine Rating'])+'MW'

    turbine_rating = site_df['Turbine Rating']

    # set turbine values
    hopp_dict, scenario, nTurbs, floris_config = hopp_tools_steel.set_turbine_model(hopp_dict, turbine_model, scenario, project_path,floris_dir, floris,site_location,grid_connection_scenario)


# Establish wind farm and electrolyzer sizing

    # Calculate target hydrogen and electricity demand
    hydrogen_consumption_for_steel = 0.06596 # metric tonnes of hydrogen/metric tonne of steel production

    # Annual hydrogen production target to meet steel production target
    steel_ammonia_plant_cf = 0.9
    hydrogen_production_target_kgpy = steel_annual_production_rate_target_tpy*1000*hydrogen_consumption_for_steel/steel_ammonia_plant_cf

    hydrogen_demand_kgphr = hydrogen_production_target_kgpy/8760

    # Calculate equivalent ammona production target
    hydrogen_consumption_for_ammonia = 0.197284403              # kg of hydrogen/kg of ammonia production
    ammonia_production_target_kgpy = hydrogen_production_target_kgpy/hydrogen_consumption_for_ammonia*steel_ammonia_plant_cf

    electrolyzer_energy_kWh_per_kg_estimate_BOL = 54.61 # Eventually need to re-arrange things to get this from set_electrolyzer_info 54.55

    electrolyzer_energy_kWh_per_kg_estimate_EOL = electrolyzer_energy_kWh_per_kg_estimate_BOL*(1+electrolyzer_degradation_power_increase)

    # Annual electricity target to meet hydrogen production target - use this to calculate renewable plant sizing
    #electricity_production_target_MWhpy = hydrogen_production_target_kgpy*electrolyzer_energy_kWh_per_kg_estimate_BOL/1000

    # Estimate required electrolyzer capacity
    if floris == False:
        if grid_connection_scenario =='off-grid':
            # For PySAM, use probable wind capacity factors by location if off-grid
            if site_location == 'Site 1':
                cf_estimate = 0.402
            elif site_location == 'Site 2':
                cf_estimate = 0.492
            elif site_location == 'Site 3':
                cf_estimate = 0.395
            elif site_location == 'Site 4':
                cf_estimate = 0.303
            elif site_location == 'Site 5':
                cf_estimate = 0.399

        else:
            # If grid-connected, base capacity off of constant full-power operation (steel/ammonia plant CF is incorporated above)
            cf_estimate = 1

        # Electrolyzer rated hydrogen production capacity - independent of degradation
        hydrogen_production_capacity_required_kgphr = hydrogen_production_target_kgpy/(8760*cf_estimate)

        # Annual electricity requirement estimate
        electricity_production_target_MWhpyr = hydrogen_production_target_kgpy*electrolyzer_energy_kWh_per_kg_estimate_EOL/1000

        # Electrolyzer power requirement at BOL - namplate capacity in MWe?
        electrolyzer_capacity_BOL_MW = hydrogen_production_capacity_required_kgphr*electrolyzer_energy_kWh_per_kg_estimate_BOL/1000

        # Electrolyzer power requirement at EOL
        electrolyzer_capacity_EOL_MW = hydrogen_production_capacity_required_kgphr*electrolyzer_energy_kWh_per_kg_estimate_EOL/1000

        # Size wind plant for providing power to electrolyzer at EOL. Do not size wind plant here to consider wind degradation
        # because we are not actually modeling wind plant degradation; if we size it in here we will have more wind generation
        # than we would in reality becaue the model does not take into account degradation. Wind plant degradation can be factored
        # into capital cost later.
        n_turbines = int(np.ceil(np.ceil(electrolyzer_capacity_EOL_MW)/turbine_rating))
        wind_size_mw = n_turbines*turbine_rating
        #wind_size_mw = electrolyzer_capacity_EOL_MW
        #wind_size_mw = electrolyzer_capacity_EOL_MW*1.08

        # # End of life required electrolyzer capacity in MW
        # electrolyzer_capacity_EOL_MW = electricity_production_target_MWhpy/(8760*cf_estimate)
        # # Electrolyzer hydrogen production rated capacity
        # hydrogen_production_rated_capacity_kgphr = electrolyzer_capacity_EOL_MW/(electrolyzer_energy_kWh_per_kg_estimate_BOL/1000)
        # # End-of-life electrolyzer electrical capacity taking into account stack degradation
        # electrolyzer_capacity_EOL_MW = electrolyzer_capacity_BOL_MW*1.13
        # # Wind plant size taking into account both electrolyzer and turbine degradation. NOTE: unclear if we should take degradation
        # # into account here (where it will influence amount of electricity available for hydrogen production)
        # wind_size_mw = electrolyzer_capacity_BOL_MW
        # #wind_size_mw = electrolyzer_capacity_EOL_MW*1.08
    else:
        wind_size_mw = nTurbs*turbine_rating
        electrolyzer_capacity_EOL_MW = wind_size_mw
        electrolyzer_capacity_BOL_MW = electrolyzer_capacity_EOL_MW/(1+electrolyzer_degradation_power_increase)

        # if grid_connection_scenario != 'off-grid':
        #     hydrogen_production_capacity_required_kgphr = hydrogen_production_target_kgpy/(8760)
        # else:
        hydrogen_production_capacity_required_kgphr = electrolyzer_capacity_BOL_MW*1000/electrolyzer_energy_kWh_per_kg_estimate_BOL


    interconnection_size_mw = wind_size_mw # this makes sense because wind_size_mw captures extra electricity needed by electrolzyer at end of life
    #electrolyzer_size_mw = np.ceil(electrolyzer_capacity_EOL_MW)
    #electrolyzer_size_mw = np.ceil(electrolyzer_capacity_BOL_MW)
    cluster_cap_mw = 40
    n_pem_clusters_max = int(np.ceil(np.ceil(electrolyzer_capacity_BOL_MW)/cluster_cap_mw))
    electrolyzer_size_mw = n_pem_clusters_max*cluster_cap_mw

    #n_pem_clusters = 12
    if electrolysis_scale == 'Distributed':
        n_pem_clusters = 1
    elif electrolysis_scale == 'Centralized':
        n_pem_clusters = n_pem_clusters_max



        # if grid_connection_scenario == 'off-grid':


        #     cluster_size_mw = np.ceil(electrolyzer_size_mw/number_pem_stacks/cluster_cap_mw)*cluster_cap_mw
        #     n_pem_clusters = int(electrolyzer_size_mw/cluster_size_mw)
        # else:
        #     n_pem_clusters = number_pem_stacks

    kw_continuous = electrolyzer_size_mw * 1000
    load = [kw_continuous for x in
            range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
    if battery_for_minimum_electrolyzer_op:
        battery_dispatch_load = list(0.1*np.array(load))
    else:
        battery_dispatch_load = list(np.array(load))

    # Add things to hopp_dict that we couldn't add before getting wind and electrolyzer size
    sub_dict = {
        'wind_size': wind_size_mw,
        'kw_continuous': kw_continuous,
        'interconnection_size_mw': interconnection_size_mw,
        'electrolyzer_size_mw': electrolyzer_size_mw,
        }

    hopp_dict.add('Configuration', sub_dict)

    scenario['Useful Life'] = useful_life

    # financials
    hopp_dict, scenario = hopp_tools_steel.set_financial_info(hopp_dict, scenario, debt_equity_split, discount_rate)

    # set electrolyzer information
    hopp_dict, electrolyzer_capex_kw, electrolyzer_component_costs_kw,capex_ratio_dist, electrolyzer_energy_kWh_per_kg, time_between_replacement =  hopp_tools_steel.set_electrolyzer_info(hopp_dict, atb_year,electrolysis_scale,electrolyzer_cost_case,electrolyzer_degradation_power_increase,grid_connection_scenario,turbine_rating,direct_coupling)


    electrolyzer_installation_factor = 12/100
    electrolyzer_direct_cost_kw = electrolyzer_capex_kw*(1+electrolyzer_installation_factor)
    #
    # Extract Scenario Information from ORBIT Runs
    # Load Excel file of scenarios
    # OSW sites and cost file including turbines 8/16/2022

    # site info
    # solar_size_mw=0
    hopp_dict, site_df, sample_site = hopp_tools_steel.set_site_info(hopp_dict, site_df, sample_site)
    site_name = site_df['State']
    #fixed_or_floating_wind = site_df['Substructure technology']
    site = SiteInfo(sample_site, hub_height=scenario['Tower Height'])

    #Assign scenario cost details
    if atb_year == 2020:
        total_capex = site_df['2020 CapEx']
        wind_om_cost_kw = site_df['2020 OpEx ($/kw-yr)']*(1+wind_plant_degradation_power_decrease)
    if atb_year == 2025:
        total_capex = site_df['2025 CapEx']
        wind_om_cost_kw = site_df['2025 OpEx ($/kw-yr)']*(1+wind_plant_degradation_power_decrease)
    if atb_year == 2030:
        total_capex = site_df['2030 CapEx']
        wind_om_cost_kw = site_df['2030 OpEx ($/kw-yr)']*(1+wind_plant_degradation_power_decrease)
    if atb_year == 2035:
        total_capex = site_df['2035 CapEx']
        wind_om_cost_kw = site_df['2035 OpEx ($/kw-yr)']*(1+wind_plant_degradation_power_decrease)

    hopp_dict.add('Configuration', {'site': site})
    if grid_connection_scenario !='grid-only':
        if run_pv_battery_sweep:

            inputs_for_sweep=[atb_year,policy_option,policy[i],hopp_dict,\
            electrolysis_scale,scenario,project_path,results_dir,\
            grid_connected_hopp,grid_connection_scenario,grid_price_scenario,\
            site_df,sample_site,site,site_location,\
            turbine_model,wind_size_mw,nTurbs,floris_config,floris,\
            sell_price,buy_price,discount_rate,debt_equity_split,\
            electrolyzer_size_mw,electrolyzer_capacity_EOL_MW,n_pem_clusters,pem_control_type,hydrogen_demand_kgphr,\
            electrolyzer_capex_kw,electrolyzer_component_costs_kw,wind_plant_degradation_power_decrease,electrolyzer_energy_kWh_per_kg,time_between_replacement,\
            user_defined_stack_replacement_time,use_optimistic_pem_efficiency,electrolyzer_degradation_penalty,storage_capacity_multiplier,hydrogen_production_capacity_required_kgphr,\
            electrolyzer_model_parameters,electricity_production_target_MWhpyr,turbine_rating,electrolyzer_degradation_power_increase,cluster_cap_mw,interconnection_size_mw,solar_ITC,grid_price_filename,\
            gams_locations_rodeo_version,rodeo_output_dir,run_RODeO_selector]
            #if solar and battery size lists are set to 'None' then defaults will be used
            #
            lcoh,hopp_dict,best_result_data,param_sweep_tracker,combined_pv_wind_power_production_hopp,combined_pv_wind_storage_power_production_hopp,\
            combined_pv_wind_curtailment_hopp,energy_shortfall_hopp,energy_to_electrolyzer,hybrid_plant,solar_size_mw,wind_size_mw,\
            storage_size_mw,storage_size_mwh,renewable_plant_cost,lcoe,cost_to_buy_from_grid, profit_from_selling_to_grid,\
            cf_wind_annuals,cf_solar_annuals,wind_itc_total=solar_storage_param_sweep(project_path,inputs_for_sweep,save_param_sweep_best_case,save_param_sweep_general_info,solar_sizes_mw,storage_sizes_mw,storage_sizes_mwh)
            []

            # Might not need everything below
            capex_multiplier = site_df['CapEx Multiplier']
            wind_cost_kw = copy.deepcopy(total_capex) * capex_multiplier
            hopp_dict.main_dict['Configuration']['wind_om_cost_kw']=wind_om_cost_kw
            hopp_dict.main_dict['Configuration']['wind_cost_kw']=wind_cost_kw
            renewable_plant_cost['wind']={'o&m_per_kw':wind_om_cost_kw,'capex_per_kw':wind_cost_kw,'size_mw':wind_size_mw}
            #renewable_plant_cost['pv']={'o&m_per_kw':solar_om_cost_kw,'capex_per_kw':solar_cost_kw,'size_mw':solar_size_mw}
            if solar_size_mw>0:
                solar_om_cost_kw=site_df[str(atb_year) + ' PV OpEx']/1.34
                solar_capex_multiplier=site_df['PV Capex Multiplier']
                solar_capex=site_df[str(atb_year) + ' PV base installed cost']/1.34
                solar_cost_kw=solar_capex * solar_capex_multiplier
                hopp_dict.main_dict['Configuration']['solar_size']=solar_size_mw
                hopp_dict.main_dict['Configuration']['solar_cost_kw']=solar_cost_kw
                hopp_dict.main_dict['Configuration']['solar_om_cost_kw']=solar_om_cost_kw
            renewable_plant_cost['pv']={'o&m_per_kw':solar_om_cost_kw,'capex_per_kw':solar_cost_kw,'size_mw':solar_size_mw}

            if storage_size_mw>0:
                storage_hours = storage_size_mwh/storage_size_mw
            else:
                storage_hours=0
            renewable_plant_cost['battery']={'capex_per_kw':storage_cost_kw,
                    'capex_per_kwh':storage_cost_kwh,
                    'o&m_percent':storage_om_percent,
                    'size_mw':storage_size_mw,
                    'size_mwh':storage_size_mwh,
                    'storage_hours':storage_hours}
            if storage_size_mw>0:
                hopp_dict.main_dict['Configuration']['storage_size_mw']=storage_size_mw
                hopp_dict.main_dict['Configuration']['storage_size_mwh']=storage_size_mwh
                hopp_dict.main_dict['Configuration']['battery_cost_kw']=storage_cost_kw
                hopp_dict.main_dict['Configuration']['battery_cost_kwh']=storage_cost_kwh

        elif run_pv_battery_sweep==False:

            capex_multiplier = site_df['CapEx Multiplier']
            wind_cost_kw = copy.deepcopy(total_capex) * capex_multiplier*(1+wind_plant_degradation_power_decrease)
            hopp_dict.main_dict['Configuration']['wind_om_cost_kw']=wind_om_cost_kw
            hopp_dict.main_dict['Configuration']['wind_cost_kw']=wind_cost_kw
            renewable_plant_cost['wind']={'o&m_per_kw':wind_om_cost_kw,'capex_per_kw':wind_cost_kw,'size_mw':wind_size_mw}
            #renewable_plant_cost['pv']={'o&m_per_kw':solar_om_cost_kw,'capex_per_kw':solar_cost_kw,'size_mw':solar_size_mw}
            if solar_size_mw>0:
                solar_om_cost_kw=site_df[str(atb_year) + ' PV OpEx']/1.34
                solar_capex_multiplier=site_df['PV Capex Multiplier']
                solar_capex=site_df[str(atb_year) + ' PV base installed cost']/1.34
                solar_cost_kw=solar_capex * solar_capex_multiplier
                hopp_dict.main_dict['Configuration']['solar_size']=solar_size_mw
                hopp_dict.main_dict['Configuration']['solar_cost_kw']=solar_cost_kw
                hopp_dict.main_dict['Configuration']['solar_om_cost_kw']=solar_om_cost_kw
            renewable_plant_cost['pv']={'o&m_per_kw':solar_om_cost_kw,'capex_per_kw':solar_cost_kw,'size_mw':solar_size_mw}
            if storage_size_mw>0:
                storage_hours = storage_size_mwh/storage_size_mw
            else:
                storage_hours=0
            renewable_plant_cost['battery']={'capex_per_kw':storage_cost_kw,
                    'capex_per_kwh':storage_cost_kwh,
                    'o&m_percent':storage_om_percent,
                    'size_mw':storage_size_mw,
                    'size_mwh':storage_size_mwh,
                    'storage_hours':storage_hours}

            #Plot Wind Data to ensure offshore data is sound
            wind_data = site.wind_resource._data['data']
            wind_speed = [W[2] for W in wind_data]
            #plot_results.plot_wind_results(wind_data, site_name, site_df['Representative coordinates'], results_dir, plot_wind)

            #Plot Wind Cost Contributions
            # Plot a nested pie chart of results
            # TODO: Remove export system from pieplot
            # plot_results.plot_pie(site_df, site_name, turbine_model, results_dir)
            #start for-loop!
            if storage_size_mw>0:
                storage_hours = storage_size_mwh/storage_size_mw
            else:
                storage_hours=0
            renewable_plant_cost['battery']={'capex_per_kw':storage_cost_kw,
                    'capex_per_kwh':storage_cost_kwh,
                    'o&m_percent':storage_om_percent,
                    'size_mw':storage_size_mw,
                    'size_mwh':storage_size_mwh,
                    'storage_hours':storage_hours}
            run_wind_plant=True
            if storage_size_mw>0:
                hopp_dict.main_dict['Configuration']['storage_size_mw']=storage_size_mw
                hopp_dict.main_dict['Configuration']['storage_size_mwh']=storage_size_mwh
                hopp_dict.main_dict['Configuration']['battery_cost_kw']=storage_cost_kw
                hopp_dict.main_dict['Configuration']['battery_cost_kwh']=storage_cost_kwh

            # ## skip running renewables if grid-only
            #if True: #grid_connection_scenario != 'grid-only':
                # Run HOPP
            hopp_dict, combined_pv_wind_power_production_hopp, energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, hybrid_plant, wind_size_mw, solar_size_mw, lcoe = \
                hopp_tools_steel.run_HOPP(
                            project_path,
                            hopp_dict,
                            scenario,
                            site,
                            sample_site,
                            forced_sizes,
                            solar_size_mw,
                            wind_size_mw,
                            storage_size_mw,
                            storage_size_mwh,
                            wind_cost_kw,
                            solar_cost_kw,
                            storage_cost_kw,
                            storage_cost_kwh,
                            kw_continuous,
                            load,
                            electrolyzer_size_mw,
                            wind_om_cost_kw,
                            solar_om_cost_kw,
                            nTurbs,
                            floris_config,
                            floris,
                            run_wind_plant
                    )

            cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
            if solar_size_mw >0:
                cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
            else:
                cf_solar_annuals = np.zeros(30)
            wind_itc_total = hybrid_plant.wind._financial_model.Outputs.itc_total
            # if floris:
            # #ACTUAL WIND SIZE
            #     hopp_dict.main_dict['Configuration']['n_Turbs']=hybrid_plant.wind._system_model.nTurbs
            #     hopp_dict.main_dict['Configuration']['turb_rating_kw']=hybrid_plant.wind._system_model.turb_rating
            #     hopp_dict.main_dict['Configuration']['wind_size_mw']=hybrid_plant.wind._system_model.nTurbs*hybrid_plant.wind._system_model.turb_rating*(1/1000)
            #     wind_size_mw=hybrid_plant.wind._system_model.nTurbs*hybrid_plant.wind._system_model.turb_rating*(1/1000)
            #     renewable_plant_cost['wind']['size_mw']=wind_size_mw
            # energy_shortfall_hopp = [x - y for x, y in
            #                     zip(battery_dispatch_load,combined_pv_wind_power_production_hopp)]
            # energy_shortfall_hopp = [x if x > 0 else 0 for x in energy_shortfall_hopp]
            # combined_pv_wind_curtailment_hopp = [x - y for x, y in
            #                     zip(combined_pv_wind_power_production_hopp,load)]
            # combined_pv_wind_curtailment_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_hopp]
            # combined_pv_wind_curtailment_hopp[0]=0

            generation_summary_df = pd.DataFrame({'Generation profile (kW)': hybrid_plant.grid.generation_profile[0:8760] })
        #generation_summary_df.to_csv(os.path.join(results_dir, 'Generation Summary_{}_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model,scenario['Powercurve File'])))


        #Step 4: Plot HOPP Results
        # plot_results.plot_HOPP(combined_pv_wind_power_production_hopp,
        #                         energy_shortfall_hopp,
        #                         combined_pv_wind_curtailment_hopp,
        #                         load,
        #                         results_dir,
        #                         site_name,
        #                         atb_year,
        #                         turbine_model,
        #                         hybrid_plant,
        #                         plot_power_production)

            #Step 5: Run Simple Dispatch Model
            hopp_dict, combined_pv_wind_storage_power_production_hopp, battery_SOC, battery_used, excess_energy = \
                hopp_tools_steel.run_battery(
                    hopp_dict,
                    energy_shortfall_hopp,
                    combined_pv_wind_curtailment_hopp,
                    combined_pv_wind_power_production_hopp
                )

            # plot_results.plot_battery_results(
            #     combined_pv_wind_curtailment_hopp,
            #     energy_shortfall_hopp,
            #     combined_pv_wind_storage_power_production_hopp,
            #     combined_pv_wind_power_production_hopp,
            #     battery_SOC,
            #     battery_used,
            #     results_dir,
            #     site_name,atb_year,turbine_model,
            #     load,
            #     plot_battery,
            # )

            # grid information
            hopp_dict, cost_to_buy_from_grid, profit_from_selling_to_grid, energy_to_electrolyzer = hopp_tools_steel.grid(
                hopp_dict,
                combined_pv_wind_storage_power_production_hopp,
                sell_price,
                excess_energy,
                buy_price,
                kw_continuous,
                plot_grid,
            )

    # else:
    elif grid_connection_scenario == 'grid-only':

        wind_cost_kw = 0
        lcoe = 0
        wind_size_mw = 0
        solar_size_mw = 0
        storage_size_mw = 0
        storage_hours = 0
        cf_wind_annuals = np.zeros(30)
        cf_solar_annuals = np.zeros(30)
        wind_itc_total = 0

        combined_pv_wind_storage_power_production_hopp = np.zeros(len(load))
        combined_pv_wind_curtailment_hopp = np.zeros(len(load))
        energy_shortfall_hopp = load
        excess_energy = np.zeros(len(load))
        hybrid_plant = 0

        # grid information
        hopp_dict, cost_to_buy_from_grid, profit_from_selling_to_grid, energy_to_electrolyzer = hopp_tools_steel.grid(
            hopp_dict,
            combined_pv_wind_storage_power_production_hopp,
            sell_price,
            excess_energy,
            buy_price,
            kw_continuous,
            plot_grid,
        )

    # Calculate capacity factor of electricity. For now  basing off wind size because we are setting electrolyzer capacity = wind capacity,
    # but in future may want to adjust this
    cf_electricity = sum(energy_to_electrolyzer)/(electrolyzer_size_mw*8760*1000)
    if solar_size_mw > 0:
        cf_solar = hybrid_plant.pv.capacity_factor/100
        solar_annual_energy_MWh = hybrid_plant.annual_energies['pv']/1000
    else:
        cf_solar = 0
        solar_annual_energy_MWh = 0

    if wind_size_mw > 0:
        cf_wind = hybrid_plant.wind.capacity_factor/100
        wind_annual_energy_MWh = hybrid_plant.annual_energies['wind']/1000
    else:
        cf_wind = 0
        wind_annual_energy_MWh = 0


    # Step #: Calculate hydrogen pipe costs for distributed case
    if electrolysis_scale == 'Distributed':

        # Add losses back into distributed case
        energy_to_electrolyzer_new = []
        for energy_hr in energy_to_electrolyzer:
            energy_to_electrolyzer_new.append(energy_hr*(1+0.0424/(1-0.1283)))

        energy_to_electrolyzer = energy_to_electrolyzer_new

        # High level estimate of max hydrogen flow rate. Doesn't have to be perfect, but should be slightly conservative (higher efficiency)
        hydrogen_max_hourly_production_kg = max(energy_to_electrolyzer)/electrolyzer_energy_kWh_per_kg

        # Run pipe cost analysis module
        pipe_network_cost_total_USD,pipe_network_costs_USD,pipe_material_cost_bymass_USD =\
            distributed_pipe_cost_analysis.hydrogen_steel_pipeline_cost_analysis(project_path,turbine_model,hydrogen_max_hourly_production_kg,site_name)

        pipeline_material_cost = pipe_network_costs_USD['Total material cost ($)'].sum()

        # Eventually replace with calculations

        if site_name == 'TX':
            cabling_material_cost = 50196586
        if site_name == 'IA':
            cabling_material_cost = 55183961
        if site_name == 'IN':
            cabling_material_cost = 66955046
        if site_name == 'WY':
            cabling_material_cost = 50196586
        if site_name == 'MS':
            cabling_material_cost = 108416091
        if site_name == 'MN':
            cabling_material_cost = 66955046
        transmission_cost = 0

        cabling_vs_pipeline_cost_difference = cabling_material_cost - pipeline_material_cost

        turbine_power_electronics_savings = 13

    elif electrolysis_scale == 'Centralized':
        cabling_vs_pipeline_cost_difference = 0
        if grid_connection_scenario == 'hybrid-grid' or grid_connection_scenario == 'grid-only':

            # Upload the right transmission cost CSV. Note, only works up to 1049 MW (files only go up to 1000 MW)
            plant_step_size = 100
            nearest_interconnect_size = int(np.round(interconnection_size_mw/plant_step_size))*plant_step_size
            transmission_cost_df = pd.read_csv(os.path.join(project_path,'H2_Analysis','Transmission_costs',str(nearest_interconnect_size)+'MW_plant_transmission_costs.csv'),index_col = None,header = 0)

            # Find the closest lat-lon in the transmission cost df
            lat_lon = (site_df['Lat'],site_df['Lon'])
            transmission_cost_lat_lons = [(x,y) for x,y in zip(transmission_cost_df['latitude'],transmission_cost_df['longitude'])]
            transmission_cost_lat,transmission_cost_lon = transmission_cost_lat_lons[cdist([lat_lon],transmission_cost_lat_lons).argmin()]

            trans_cap_cost_per_mw = transmission_cost_df.loc[(transmission_cost_df['latitude']==transmission_cost_lat) & (transmission_cost_df['longitude']==transmission_cost_lon),'trans_cap_cost_per_mw'].tolist()[0]
            reinforcement_cost_per_mw = transmission_cost_df.loc[(transmission_cost_df['latitude']==transmission_cost_lat) & (transmission_cost_df['longitude']==transmission_cost_lon),'reinforcement_cost_per_mw'].tolist()[0]
            transmission_cost = (trans_cap_cost_per_mw + reinforcement_cost_per_mw)*interconnection_size_mw
            []
        else:
            transmission_cost = 0

        turbine_power_electronics_savings = 0

    if grid_connection_scenario != 'grid-only':
        revised_renewable_cost = hybrid_plant.grid.total_installed_cost - cabling_vs_pipeline_cost_difference - turbine_power_electronics_savings*wind_size_mw*1000 + transmission_cost
        renewable_plant_cost['wind_savings_dollars']={'turbine_power_electronics_savings_dollars':-1*turbine_power_electronics_savings*wind_size_mw*1000,
        'tranmission_cost_dollars':transmission_cost,'cabling_vs_pipeline_cost_difference_dollars':-1*cabling_vs_pipeline_cost_difference}
    else:
        revised_renewable_cost = 0.0


    # Step 6: Run RODeO or Profast for hydrogen

    #if run_RODeO_selector == True:
        # rodeo_scenario,lcoh,electrolyzer_capacity_factor,hydrogen_storage_duration_hr,hydrogen_storage_capacity_kg,\
        #     hydrogen_annual_production,water_consumption_hourly,RODeO_summary_results_dict,hydrogen_hourly_results_RODeO,\
        #         electrical_generation_timeseries,electrolyzer_installed_cost_kw,hydrogen_storage_cost_USDprkg\
        #     = run_RODeO.run_RODeO(atb_year,site_name,turbine_model,electrolysis_scale,policy_option,policy,i,wind_size_mw,solar_size_mw,electrolyzer_size_mw,\
        #             energy_to_electrolyzer,electrolyzer_energy_kWh_per_kg,hybrid_plant,renewable_plant_cost,electrolyzer_capex_kw,capex_ratio_dist,wind_om_cost_kw,useful_life,time_between_replacement,\
        #             grid_connection_scenario,grid_price_scenario,gams_locations_rodeo_version,rodeo_output_dir)


        # hydrogen_lifecycle_emissions = LCA_single_scenario.hydrogen_LCA_singlescenario(grid_connection_scenario,atb_year,site_name,turbine_model,electrolysis_scale,\
        #                                                                             policy_option,grid_price_scenario,electrolyzer_energy_kWh_per_kg,hydrogen_hourly_results_RODeO)

        # # Max hydrogen production rate [kg/hr]
        # max_hydrogen_production_rate_kg_hr = hydrogen_hourly_results_RODeO['Electrolyzer hydrogen production [kg/hr]'].max()
        # max_hydrogen_delivery_rate_kg_hr = hydrogen_hourly_results_RODeO['Product Sold (units of product)'].max()

        # electrolyzer_capacity_factor = RODeO_summary_results_dict['input capacity factor']

    #else:
    # If not running RODeO, run H2A via ProFAST
        # Currently only works for offgrid
        #grid_string = 'offgrid'
        #scenario_name = 'steel_'+str(atb_year)+'_'+ site_location.replace(' ','-') +'_'+turbine_model+'_'+grid_string

    #Run the H2_PEM model to get hourly hydrogen output, capacity factor, water consumption, etc.
    h2_model = 'Simple'
    h2_model = 'Simple'
    hopp_dict, H2_Results, electrical_generation_timeseries = hopp_tools_steel.run_H2_PEM_sim(
        hopp_dict,
        #hybrid_plant,
        energy_to_electrolyzer,
        scenario,
        # wind_size_mw,
        # solar_size_mw,
        electrolyzer_size_mw,
        electrolysis_scale,
        n_pem_clusters,
        pem_control_type,
        electrolyzer_direct_cost_kw,
        electrolyzer_model_parameters,
        electrolyzer_degradation_penalty,
        grid_connection_scenario,
        hydrogen_production_capacity_required_kgphr
        # kw_continuous,
        # electrolyzer_capex_kw,
        # lcoe,
    )

    h2_hourly_production = H2_Results['hydrogen_hourly_production'].tolist()
    # fig, ax = plt.subplots(1,1)
    # ax.plot(h2_hourly_production)
    # plt.show()

    h2_hourly_production.sort()
    h2_hourly_production = [math.floor(x) for x in h2_hourly_production]
    #h2_hourly_production = h2_hourly_production.remove(0)
    h2_min_production_rate_nonzero = []
    for j in range(len(h2_hourly_production)):
        if h2_hourly_production[j] > 0:
            h2_min_production_rate_nonzero.append(h2_hourly_production[j])

    #Step 6b: Run desal model
    hopp_dict, desal_capex, desal_opex = hopp_tools_steel.desal_model(
        hopp_dict,
        H2_Results,
        electrolyzer_size_mw,
        electrical_generation_timeseries,
        useful_life,
    )

    hydrogen_annual_production = H2_Results['hydrogen_annual_output']

    # hydrogen_max_hourly_production_kg = max(H2_Results['hydrogen_hourly_production'])

    # Calculate required storage capacity to meet a flat demand profile. In the future, we could customize this to
    # work with any demand profile

    # Storage costs as a function of location
    if site_location == 'Site 1':
        storage_type = 'Buried pipes'
    elif site_location == 'Site 2':
        storage_type = 'Salt cavern'
    elif site_location == 'Site 3':
        storage_type = 'Buried pipes'
    elif site_location == 'Site 4':
        storage_type = 'Salt cavern'
    elif site_location == 'Site 5':
        storage_type = 'Lined rock cavern' 

    hydrogen_production_storage_system_output_kgprhr,hydrogen_storage_capacity_kg,hydrogen_storage_capacity_MWh_HHV,hydrogen_storage_duration_hr,hydrogen_storage_cost_USDprkg,storage_compressor_total_capacity_kW,storage_compressor_total_installed_cost_USD,storage_status_message\
        = hopp_tools_steel.hydrogen_storage_capacity_cost_calcs(H2_Results,electrolyzer_size_mw,storage_type,hydrogen_demand_kgphr)

    #Make sure there is enough wind capacity for storage compressor; if not, add wind capacity. Because we round up
    #to the nearest turbine size it is possible that there is already enough wind capacity; this bit just makes sure
    # that there will always be enough.
    wind_capacity_required_MW = electrolyzer_capacity_EOL_MW + storage_compressor_total_capacity_kW/1000
    if wind_capacity_required_MW > wind_size_mw:
        n_turbines = int(np.ceil(np.ceil(wind_capacity_required_MW)/turbine_rating))
        wind_size_mw = turbine_rating*n_turbines
        # renewable_plant_cost.update({'wind': {'size_mw': wind_size_mw}})
                                    
        if "wind" not in renewable_plant_cost.keys():
            renewable_plant_cost["wind"] = {}
        renewable_plant_cost['wind']['size_mw']=wind_size_mw



    # Apply storage multiplier
    hydrogen_storage_capacity_kg = hydrogen_storage_capacity_kg*storage_capacity_multiplier
    #print(storage_status_message)

    # Run ProFAST to get LCOH

    # Municipal water rates and wastewater treatment rates combined ($/gal)
    if site_location == 'Site 1': # Site 1 - Indiana
        #water_cost = 0.00612
        water_cost = 0.0045
    elif site_location == 'Site 2': # Site 2 - Texas
        #water_cost = 0.00811
        water_cost = 0.00478
    elif site_location == 'Site 3': # Site 3 - Iowa
        #water_cost = 0.00634
        water_cost = 0.00291
    elif site_location == 'Site 4': # Site 4 - Mississippi
        #water_cost = 0.00844
        water_cost = 0.00409
    elif site_location =='Site 5': # Site 5 - MN, assuming same as IA for now
        #water_cost=0.00634 
        water_cost = 0.00291

    electrolyzer_efficiency_while_running = []
    water_consumption_while_running = []
    hydrogen_production_while_running = []
    for j in range(len(H2_Results['electrolyzer_total_efficiency'])):
        if H2_Results['hydrogen_hourly_production'][j] > 0:
            electrolyzer_efficiency_while_running.append(H2_Results['electrolyzer_total_efficiency'][j])
            water_consumption_while_running.append(H2_Results['water_hourly_usage'][j])
            hydrogen_production_while_running.append(H2_Results['hydrogen_hourly_production'][j])
    # water_consumption_while_running=H2_Results['water_hourly_usage']
    # hydrogen_production_while_running=H2_Results['hydrogen_hourly_production']
    # Specify grid cost year for ATB year
    if atb_year == 2020:
        grid_year = 2025
    elif atb_year == 2025:
        grid_year = 2030
    elif atb_year == 2030:
        grid_year = 2035
    elif atb_year == 2035:
        grid_year = 2040

    # Read in csv for grid prices
    grid_prices = pd.read_csv(os.path.join(project_path, "H2_Analysis", grid_price_filename),index_col = None,header = 0)
    elec_price = grid_prices.loc[grid_prices['Year']==grid_year,site_name].tolist()[0]
    grid_prices_interpolated_USDperkwh = grid_price_interpolation(grid_prices,site_name,atb_year,useful_life,'kWh')


    # if site_name =='WY':
    #     elec_price = grid_prices.loc[grid_prices['Year']==grid_year,'TX'].tolist()[0]
    # else:
    #     elec_price = grid_prices.loc[grid_prices['Year']==grid_year,site_name].tolist()[0]

    # electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid\
    #     = LCA_single_scenario_ProFAST.hydrogen_LCA_singlescenario_ProFAST(grid_connection_scenario,atb_year,site_name,policy_option,hydrogen_production_while_running,\
    #                                                       electrolyzer_energy_kWh_per_kg,solar_size_mw,storage_size_mw,hopp_dict)

    h2_solution,h2_summary,profast_h2_price_breakdown,lcoh_breakdown,electrolyzer_installed_cost_kw,elec_cf,ren_frac,electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid,H2_PTC,Ren_PTC,h2_production_capex,\
        hydrogen_storage_cost_USDprkg,hydrogen_storage_duration_hr,hydrogen_storage_capacity_kg,electrolyzer_size_mw = run_profast_for_hydrogen.run_profast_for_hydrogen(hopp_dict,electrolyzer_size_mw,H2_Results,\
                                    electrolyzer_capex_kw,time_between_replacement,electrolyzer_energy_kWh_per_kg,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,storage_compressor_total_capacity_kW,storage_compressor_total_installed_cost_USD,hydrogen_storage_duration_hr,\
                                    desal_capex,desal_opex,useful_life,water_cost,wind_size_mw,solar_size_mw,storage_size_mw,renewable_plant_cost,wind_om_cost_kw,grid_connected_hopp,\
                                    grid_connection_scenario,atb_year, site_name, policy_option, policy[i],electrical_generation_timeseries, combined_pv_wind_storage_power_production_hopp,combined_pv_wind_curtailment_hopp,\
                                    energy_shortfall_hopp,elec_price,grid_prices_interpolated_USDperkwh, grid_price_scenario,user_defined_stack_replacement_time,use_optimistic_pem_efficiency,wind_annual_energy_MWh,solar_annual_energy_MWh,solar_ITC,gams_locations_rodeo_version,rodeo_output_dir,run_RODeO_selector)

    lcoh = h2_solution['price']

    electrolysis_total_EI_policy_grid_firstyear = electrolysis_total_EI_policy_grid[atb_year+5]
    electrolysis_total_EI_policy_offgrid_firstyear = electrolysis_total_EI_policy_offgrid[atb_year+5]
    H2_PTC_firstyear = H2_PTC[atb_year+5]
    Ren_PTC_firstyear = Ren_PTC[atb_year+5]
    ren_frac = ren_frac


    # # Max hydrogen production rate [kg/hr]
    max_hydrogen_production_rate_kg_hr = np.max(H2_Results['hydrogen_hourly_production'])
    max_hydrogen_delivery_rate_kg_hr  = np.mean(H2_Results['hydrogen_hourly_production'])

    electrolyzer_capacity_factor = H2_Results['cap_factor']

    # Calculate hydrogen transmission cost and add to LCOH
    hopp_dict,h2_transmission_economics_from_profast,h2_transmission_economics_summary,h2_transmission_price,h2_transmission_price_breakdown = hopp_tools_steel.levelized_cost_of_h2_transmission(hopp_dict,max_hydrogen_production_rate_kg_hr,
    max_hydrogen_delivery_rate_kg_hr,electrolyzer_capacity_factor,atb_year,site_name,grid_price_filename)

    lcoh = lcoh + h2_transmission_price
    print(f"LCOH including h2_transmission: {lcoh}")
    #print(grid_connection_scenario, ' LCOH without policy:', lcoh)
    # Policy impacts on LCOH

    # if run_RODeO_selector == True:
    #     lcoh,lcoh_reduction_Ren_PTC,lcoh_reduction_H2_PTC, = hopp_tools_steel.policy_implementation_for_RODeO(grid_connection_scenario, atb_year, site_name, turbine_model, electrolysis_scale, policy_option, grid_price_scenario, electrolyzer_energy_kWh_per_kg, hydrogen_hourly_results_RODeO, RODeO_summary_results_dict, hydrogen_annual_production, useful_life, lcoh)

        #print('LCOH with policy:', lcoh)

    # Step 7: Calculate break-even cost of steel production without oxygen and heat integration
    lime_unit_cost = site_df['Lime ($/metric tonne)'] + site_df['Lime Transport ($/metric tonne)']
    carbon_unit_cost = site_df['Carbon ($/metric tonne)'] + site_df['Carbon Transport ($/metric tonne)']
    iron_ore_pellets_unit_cost = site_df['Iron Ore Pellets ($/metric tonne)'] + site_df['Iron Ore Pellets Transport ($/metric tonne)']
    o2_heat_integration = 0
    hopp_dict,steel_economics_from_profast, steel_economics_summary, profast_steel_price_breakdown,steel_breakeven_price, steel_annual_production_mtpy,steel_production_capacity_margin_pc,steel_price_breakdown = hopp_tools_steel.steel_LCOS(hopp_dict,lcoh,hydrogen_annual_production,steel_annual_production_rate_target_tpy,
                                                                                                            lime_unit_cost,
                                                                                                            carbon_unit_cost,
                                                                                                            iron_ore_pellets_unit_cost,
                                                                                                            o2_heat_integration,atb_year,site_name,grid_price_filename,electrolyzer_cost_case)


    # Calcualte break-even price of steel WITH oxygen and heat integration
    o2_heat_integration = 1
    hopp_dict,steel_economics_from_profast_integration, steel_economics_summary_integration, profast_steel_price_breakdown_integration,steel_breakeven_price_integration, steel_annual_production_mtpy_integration,steel_production_capacity_margin_pc_integration,steel_price_breakdown_integration = hopp_tools_steel.steel_LCOS(hopp_dict,lcoh,hydrogen_annual_production,steel_annual_production_rate_target_tpy,
                                                                                                            lime_unit_cost,
                                                                                                            carbon_unit_cost,
                                                                                                            iron_ore_pellets_unit_cost,
                                                                                                            o2_heat_integration,atb_year,site_name,grid_price_filename,electrolyzer_cost_case)


    # Calculate break-even price of ammonia
    cooling_water_cost = water_cost#0.000113349938601175 # $/Gal
    iron_based_catalyst_cost = 23.19977341 # $/kg
    #oxygen_cost = 0.0285210891617726       # $/kg
    oxygen_cost = 0
    hopp_dict,ammonia_economics_from_profast, ammonia_economics_summary, profast_ammonia_price_breakdown,ammonia_breakeven_price, ammonia_annual_production_kgpy,ammonia_production_capacity_margin_pc,ammonia_price_breakdown = hopp_tools_steel.levelized_cost_of_ammonia(hopp_dict,lcoh,hydrogen_annual_production,ammonia_production_target_kgpy,
                                                                                                            cooling_water_cost,
                                                                                                            iron_based_catalyst_cost,
                                                                                                            oxygen_cost,
                                                                                                            atb_year,site_name,grid_price_filename)

    # Step 7: Write outputs to file

    total_h2export_system_cost=0
    opex_pipeline=0
    total_export_system_cost=0
    total_export_om_cost=0

    # if run_RODeO_selector == True:
    #     policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
    #     scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
    #     discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
    #     wind_itc_total, total_itc_hvdc = hopp_tools_steel.write_outputs_RODeO(electrical_generation_timeseries,\
    #                         hybrid_plant,
    #                         total_export_system_cost,
    #                         total_export_om_cost,
    #                         cost_to_buy_from_grid,
    #                         electrolyzer_capex_kw,
    #                         electrolyzer_installed_cost_kw,
    #                         hydrogen_storage_cost_USDprkg,
    #                         time_between_replacement,
    #                         profit_from_selling_to_grid,
    #                         useful_life,
    #                         atb_year,
    #                         policy_option,
    #                         scenario,
    #                         wind_cost_kw,
    #                         solar_cost_kw,
    #                         discount_rate,
    #                         solar_size_mw,
    #                         results_dir,
    #                         fin_sum_dir,
    #                         site_name,
    #                         turbine_model,
    #                         electrolysis_scale,
    #                         scenario_choice,
    #                         lcoe,
    #                         run_RODeO_selector,
    #                         grid_connection_scenario,
    #                         grid_price_scenario,
    #                         elec_price,
    #                         lcoh,
    #                         h2_transmission_price,
    #                         lcoh_reduction_Ren_PTC,
    #                         lcoh_reduction_H2_PTC,
    #                         electrolyzer_capacity_factor,
    #                         hydrogen_storage_duration_hr,
    #                         hydrogen_storage_capacity_kg,
    #                         hydrogen_annual_production,
    #                         water_consumption_hourly,
    #                         RODeO_summary_results_dict,
    #                         steel_annual_production_mtpy,
    #                         steel_breakeven_price,
    #                         steel_price_breakdown,
    #                         steel_breakeven_price_integration,
    #                         ammonia_annual_production_kgpy,
    #                         ammonia_breakeven_price,
    #                         ammonia_price_breakdown)
    # else:
    policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
    scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
    discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
    wind_itc_total, total_itc_hvdc = hopp_tools_steel.write_outputs_ProFAST(electrical_generation_timeseries,\
                        cf_wind_annuals,
                        cf_solar_annuals,
                        wind_itc_total,
                        total_export_system_cost,
                        total_export_om_cost,
                        cost_to_buy_from_grid,
                        electrolyzer_capex_kw,
                        electrolyzer_installed_cost_kw,
                        electrolyzer_cost_case,
                        hydrogen_storage_cost_USDprkg,
                        time_between_replacement,
                        profit_from_selling_to_grid,
                        useful_life,
                        atb_year,
                        policy_option,
                        scenario,
                        wind_cost_kw,
                        solar_cost_kw,
                        wind_size_mw,
                        solar_size_mw,
                        storage_size_mw,
                        storage_hours,
                        electrolyzer_size_mw,
                        discount_rate,
                        results_dir,
                        fin_sum_dir,
                        energy_profile_dir,
                        price_breakdown_dir,
                        site_name,
                        turbine_model,
                        electrolysis_scale,
                        scenario_choice,
                        lcoe,
                        cf_electricity,
                        cf_wind,
                        cf_solar,
                        wind_annual_energy_MWh,
                        solar_annual_energy_MWh,
                        run_RODeO_selector,
                        grid_connection_scenario,
                        grid_price_scenario,
                        elec_price,
                        lcoh,
                        h2_transmission_price,
                        h2_production_capex,
                        H2_Results,
                        elec_cf,
                        ren_frac,
                        electrolysis_total_EI_policy_grid_firstyear,
                        electrolysis_total_EI_policy_offgrid_firstyear,
                        H2_PTC_firstyear,
                        Ren_PTC_firstyear,
                        run_pv_battery_sweep,
                        electrolyzer_degradation_penalty,
                        user_defined_stack_replacement_time,
                        pem_control_type,
                        n_pem_clusters,
                        storage_capacity_multiplier,
                        floris,
                        hydrogen_storage_duration_hr,
                        hydrogen_storage_capacity_kg,
                        lcoh_breakdown,
                        steel_annual_production_mtpy,
                        steel_production_capacity_margin_pc,
                        steel_breakeven_price,
                        steel_price_breakdown,
                        steel_breakeven_price_integration,
                        ammonia_annual_production_kgpy,
                        ammonia_production_capacity_margin_pc,
                        ammonia_breakeven_price,
                        ammonia_price_breakdown,
                        profast_h2_price_breakdown,
                        profast_steel_price_breakdown,
                        profast_ammonia_price_breakdown,
                        hopp_dict)
    
    return policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
    scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
    discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
    wind_itc_total, total_itc_hvdc, hopp_dict