import numpy as np
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
from tools.analysis.bos.cost_calculator import create_cost_calculator

class Failure:



    def __init__(self,
                 power_sources: dict,
                 electrolyzer: bool,
                 project_life: int,
                 generation_profile: dict,
                 load: list, 
                 failure_distribution: bool): 
        """
        Base class for adding technology failure.
        Target components: gearbox (wind), inverter (solar), 
            stack (battery), module (electrolyzer) 

        :param power_sources: tuple of strings, float pairs
            names of power sources to include and their kw sizes
            choices include: ('pv', 'wind', 'battery')
        :param electrolyzer: bool, 
            if True electrolyzer is included in failure model
        :param project_life: integer,
            duration of hybrid project in years
        :param generation_profile: arrays,
            generation_profile for each technology ('wind','pv')
        :param load: list
            absolute desired load profile [kWe]
        :param failure_distribution: bool,
            if True distribution curves will be used instead of 
            mean time between failures (MTBF)
            TODO: Allow choice of distribution or MTBF for each tech
        """
    
        self.power_sources = power_sources
        self.electrolyzer = electrolyzer
        self.generation = generation_profile
        self.project_life = project_life
        self.load = load
        self.failure_distribution = failure_distribution
        temp = list(power_sources.keys())
        for k in temp:
            power_sources[k.lower()] = power_sources.pop(k)



    def simulate_generation_failure(self):
        if 'pv' in self.power_sources.keys():
            self.inverter_MTTR = 9 #days
            self.pv_repair = []

            if 'pv' in self.generation.keys():
                pv_failed_generation = self.generation['pv']
            else:
                pv_failed_generation = self.generation.pv
            
            if self.failure_distribution:
                inverter_failure = np.random.gumbel()   #Need help getting mu and beta. Is this for entire life span or need multiple distributions?

            else:
                #TODO: Limit replacement after X year?
                self.inverter_MTBF = 4 #years
                #when_fail = np.random.choice(8760,1) will impliment different failure point in the year
                counter = 1
                for year in range(0,self.project_life):
                    if year == 0:
                        self.pv_repair = np.append(self.pv_repair, [0])

                    elif counter % self.inverter_MTBF == 0:
                        pv_repair_start = year * 8760
                        pv_repair_end = pv_repair_start + (self.inverter_MTTR *24)
                        pv_failed_generation[pv_repair_start:pv_repair_end] = [0] * (self.inverter_MTTR *24)
                        self.pv_repair = np.append(self.pv_repair, [1])

                    else:
                        self.pv_repair = np.append(self.pv_repair, [0])
                    counter += 1
            self.pv_failed_generation = pv_failed_generation

        if 'wind' in self.power_sources.keys():
            self.gearbox_MTTR = 7 #days
            self.wind_repair = []
            if 'wind' in self.generation.keys():
                self.wind_failed_generation = self.generation['wind']
            else:
                self.wind_failed_generation = self.generation.wind

            if self.failure_distribution:
                gearbox_failure = np.random.exponential()   #Need help getting mu and beta. Is this for entire life span or need multiple distributions?

            else:
                #TODO: Limit replacement after X year?
                self.gearbox_MTBF = 4 #years (range 3-5 years uptower)
                counter = 1
                for year in range(0,self.project_life):
                    if year == 0:
                        self.wind_repair = np.append(self.wind_repair, [0])

                    elif counter % self.gearbox_MTBF == 0:
                        wind_repair_start = year * 8760
                        wind_repair_end = wind_repair_start + (self.gearbox_MTTR *24)
                        self.wind_failed_generation[wind_repair_start:wind_repair_end] = [0] * (self.gearbox_MTTR *24)
                        self.wind_repair = np.append(self.wind_repair, [1])

                    else:
                        self.wind_repair = np.append(self.wind_repair, [0])
                    counter += 1

        self.hybrid_failure_generation = np.add(self.pv_failed_generation, self.wind_failed_generation)

    def simulate_battery_failure(self,input_battery_use:bool):
        """
        param: battery_used: bool,
            if True, assign self.battery_used with user input
            of battery use
        """
        #TODO: modify restart after repair

        if input_battery_use:
            self.battery_used = []
            #put the failure on the battery used.
        else:
            self.hybrid_failure_generation

            # energy specific metrics required for battery model
            self.energy_shortfall = [x - y for x, y in
                                zip(self.load,self.hybrid_failure_generation)]
            self.energy_shortfall = [x if x > 0 else 0 for x in self.energy_shortfall]
            self.combined_pv_wind_curtailment = [x - y for x, y in
                                zip(self.hybrid_failure_generation,self.load)]
            self.combined_pv_wind_curtailment = [x if x > 0 else 0 for x in self.combined_pv_wind_curtailment]
            bat_model = SimpleDispatch()
            bat_model.Nt = len(self.energy_shortfall)
            bat_model.curtailment = self.combined_pv_wind_curtailment
            bat_model.shortfall = self.energy_shortfall
            battery_used, excess_energy, battery_SOC = bat_model.run()
            self.battery_used = battery_used
        


        if 'battery' in self.power_sources.keys():
            self.stack_MTTR = 7 #days
            pass
    
    def simulate_electrolyzer_failure(self):
        if self.electrolyzer:
            self.module_MTTR = 21 #days
            pass



if __name__ == '__main__': 
    from pathlib import Path
    import matplotlib.pyplot as plt
    from hybrid.sites import SiteInfo, flatirons_site
    from hybrid.hybrid_simulation import HybridSimulation
    from hybrid.log import hybrid_logger as logger
    from hybrid.keys import set_nrel_key_dot_env

    examples_dir = Path(__file__).resolve().parents[1]
    plot_degradation = True

    # Set API key
    set_nrel_key_dot_env()

    # Set wind, solar, and interconnection capacities (in MW)
    solar_size_mw = 50
    wind_size_mw = 50
    interconnection_size_mw = 50        #Required by HybridSimulation() not currently being used for calculations.
    battery_capacity_mw = 20
    battery_capacity_mwh = battery_capacity_mw * 4 
    electrolyzer_capacity_mw = 40
    useful_life = 30
    load = [electrolyzer_capacity_mw*1000] * useful_life * 8760

    technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000
                },
                'wind': {
                    'num_turbines': 10,
                    'turbine_rating_kw': 2000},
                'battery': {
                    'system_capacity_kwh': battery_capacity_mwh * 1000,
                    'system_capacity_kw': battery_capacity_mw * 1000
                    }
                }

    # Get resource
    lat = flatirons_site['lat']
    lon = flatirons_site['lon']
    site = SiteInfo(flatirons_site)

    # Create model
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)
    

    hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
    hybrid_plant.ppa_price = 0.1
    # hybrid_plant.pv.dc_degradation = [0] * 25
    # hybrid_plant.wind._system_model.value("env_degrad_loss", 20)
    hybrid_plant.simulate(useful_life)

    # Save the outputs
    generation_profile = hybrid_plant.generation_profile

    hybrid_failure = Failure(technologies,False, useful_life, generation_profile, False)

    hybrid_failure.simulate_generation_failure()
    print("Number of pv repairs: ", hybrid_failure.pv_repair)
    print("Lifetime pv power generation (original): ", np.sum(hybrid_plant.pv.generation_profile)/1000, "[MW]")
    print("Lifetime pv power generation (inc. failure): ", np.sum(hybrid_failure.generation.pv)/1000, "[MW]")
    print("Number of wind repairs: ", hybrid_failure.wind_repair)
    print("Lifetime wind power generation (original): ", np.sum(hybrid_plant.wind.generation_profile)/1000, "[MW]")
    print("Lifetime wind power generation (inc. failure): ", np.sum(hybrid_failure.generation.wind)/1000, "[MW]")


    
    # if plot_degradation:
    #     plt.figure(figsize=(10,6))
    #     plt.subplot(311)
    #     plt.title("Max power generation vs degraded power generation")
    #     plt.plot(hybrid_degradation.wind_degraded_generation[175200:175344],label="degraded wind")
    #     plt.plot(hybrid_plant.wind.generation_profile[175200:175344],label="max generation")
    #     plt.ylabel("Power Production (kW)")
    #     plt.legend()
        
    #     plt.subplot(312)
    #     plt.plot(hybrid_degradation.pv_degraded_generation[175200:175344],label="degraded pv")
    #     plt.plot(hybrid_plant.pv.generation_profile[175200:175344],label="max generation")
    #     plt.ylabel("Power Production (kW)")
    #     plt.legend()

    #     plt.subplot(313)
    #     plt.plot(hybrid_degradation.hybrid_degraded_generation[175200:175344], label="degraded hybrid generation")
    #     plt.plot(load[175200:175344], label = "load profile")
    #     plt.ylabel("Power Production (kW)")
    #     plt.xlabel("Time (hour)")
    #     plt.legend()
    #     plt.show()