from attrs import define
from greenheart.simulation.technologies.steel.enthalpy_functions import (
    h2_enthalpy,
    h2o_enthalpy,
    sio2_enthalpy,
    al2o3_enthalpy,
    feo_enthalpy,
    fe_enthalpy,
)


@define
class MassModelConfig:
    """Configuration for the HDRI mass model.

    Attributes:
        steel_output_desired (float): The desired amount of steel output.
        metallization_rate (float): The metallization rate of DRI, default is 0.94. [%] [2]
        mol_weight_h2 (float): The molecular weight of hydrogen, default is 2.01588 g/mol. [1]
        mol_weight_h2o (float): The molecular weight of water, default is 18.0153 g/mol. [1]
        mol_weight_fe (float): The molecular weight of iron, default is 55.845 g/mol. [1]
        mol_weight_fe2o3 (float): The molecular weight of iron(III) oxide, default is 159.69 g/mol. [1]
        fe2o3_pure (float): The percentage of Fe2O3 in raw material, default is 0.95. [%] Accounts for 5% impurities in raw material. [2]
        sio2_percent (float): The percentage of SiO2 in raw materials, default is 0.03. [%] [2]
        al2o3_percent (float): The percentage of Al2O3 in raw materials, default is 0.02. [%] [2]
        h2_per_mol (float): The stoichiometric ratio of 3 moles of H2 per 2 moles of Fe, default is 1.5. [no units]
        lambda_h2 (float): The multiplier for extra H2 needed, default is 1.2. [no units] Lambda is the ratio of actual over stoichiometric requirement.

    Sources:
        [1]: Chase, M.W., Jr. 1998. "NIST-JANAF Thermochemical Tables, Fourth Edition." J. Phys. Chem. Ref. Data, Monograph 9 1-1951. doi:https://doi.org/10.18434/T4D303.
        [2]: Midrex Technologies, Inc. 2018. "DRI Products + Applications: Providing flexibility for steelmaking." April. Accessed May 2, 2022. https://www.midrex.com/wp-content/uploads/MIdrexDRI_ProductsBrochure_4-12-18.pdf.
    """

    steel_output_desired: float
    metallization_rate: float = 0.94
    mol_weight_h2: float = 2.01588
    mol_weight_h2o: float = 18.0153
    mol_weight_fe: float = 55.845
    mol_weight_fe2o3: float = 159.69
    fe2o3_pure: float = 0.95
    sio2_percent: float = 0.03
    al2o3_percent: float = 0.02
    h2_per_mol: float = 3 / 2
    lambda_h2: float = 1.2


@define
class MassModelOutputs:
    """Outputs from the HDRI mass model.

    Attributes:
        iron_ore_mass_needed (float): The mass of iron ore needed in kg.
        hydrogen_gas_needed (float): The amount of hydrogen gas needed in kg.
        hydrogen_gas_leaving_shaft (float): The amount of hydrogen gas leaving the shaft in kg.
        water_leaving_shaft (float): The amount of water leaving the shaft in kg.
        pure_iron_leaving_shaft (float): The amount of pure iron leaving the shaft in kg.
        total_mass_h2_h2o_leaving_shaft (float): The total mass of hydrogen and water leaving the shaft in kg.
        iron_ore_leaving_shaft (float): The amount of iron ore leaving the shaft in kg.
        silicon_dioxide_mass (float): The mass of silicon dioxide in kg.
        aluminium_oxide_mass (float): The mass of aluminium oxide in kg.
    """

    iron_ore_mass_needed: float
    hydrogen_gas_needed: float
    hydrogen_gas_leaving_shaft: float
    water_leaving_shaft: float
    pure_iron_leaving_shaft: float
    total_mass_h2_h2o_leaving_shaft: float
    iron_ore_leaving_shaft: float
    silicon_dioxide_mass: float
    aluminium_oxide_mass: float


def mass_model(config: MassModelConfig) -> MassModelOutputs:
    """Calculates the masses inputted and outputted of an HDRI oxidation system of iron ore in the green steel process for the output of one tonne of liquid steel.

    Args:
        config (MassModelConfig): Configuration for the HDRI mass model.

    Returns:
        MassModelOutputs: The calculated mass outputs from the HDRI mass model.

    Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022.
        "Decarbonizing primary steel production: Techno-economic assessment of a hydrogen-based green steel production plant in Norway." 
        Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.
    """
    fe_o_ratio = (
        2 * config.mol_weight_fe
    ) / config.mol_weight_fe2o3  # Ratio of Fe weight to Fe2O3 weight

    m1 = config.steel_output_desired / (
        config.fe2o3_pure * fe_o_ratio * config.metallization_rate
    )  # kg Fe2O3 amount of raw material needed for desired steel output
    m2_feo = (
        m1 * config.fe2o3_pure * fe_o_ratio * (1 - config.metallization_rate)
    )  # kg Fe2O3 amount of iron(III) oxide not reduced

    sio2_percent = config.sio2_percent  # Percent SiO2 in raw materials
    al2o3_percent = config.al2o3_percent  # Percent Al2O3 in raw materials

    m1_sio2 = sio2_percent * m1  # kg SiO2 mass of SiO2 in raw materials
    m1_al2o3 = al2o3_percent * m1  # kg Al2O3 mass of Al2O3 in raw materials

    m2_fe = (
        m1 - (m1_sio2 + m1_al2o3 + m2_feo)
    ) * fe_o_ratio  # kg Fe mass of metallic iron out of DRI per tonne steel

    h2_weight_per_mol = config.h2_per_mol * config.mol_weight_h2  # g H2 per 1 mol Fe
    mol_per_input_fe = (
        config.steel_output_desired * 1000
    ) / config.mol_weight_fe  # mols of iron in the input kg of Fe
    m4_stoich = (
        h2_weight_per_mol * mol_per_input_fe
    ) / 1000  # kg H2 minimum mass needed per tonne steel

    m4 = (
        m4_stoich * config.lambda_h2
    )  # kg H2 mass of hydrogen inputted into DRI per tonne steel

    water_mass = (
        (3 * config.mol_weight_h2o) / (2 * config.mol_weight_fe)
    ) * config.steel_output_desired  # water produced per tonne liquid steel

    m5_h2 = m4_stoich * (
        config.lambda_h2 - 1
    )  # mass of hydrogen remaining in exhaust after DRI in kg
    m5_h2o = water_mass  # mass of water produced in DRI in kg
    m5 = m5_h2 + m5_h2o  # total mass of exhaust in kg

    return MassModelOutputs(
        iron_ore_mass_needed=m1,
        hydrogen_gas_needed=m4,
        hydrogen_gas_leaving_shaft=m5_h2,
        water_leaving_shaft=m5_h2o,
        pure_iron_leaving_shaft=m2_fe,
        total_mass_h2_h2o_leaving_shaft=m5,
        iron_ore_leaving_shaft=m2_feo,
        silicon_dioxide_mass=m1_sio2,
        aluminium_oxide_mass=m1_al2o3,
    )


@define
class EnergyModelConfig:
    """Configuration for the HDRI energy model.

    Attributes:
        mass_inputs (MassModelConfig): Configuration for the mass inputs.
        h2_temp_in (float): The temperature of hydrogen gas entering the shaft in Kelvin. Default is 1173 K. [3]
        h2_temp_out (float): The temperature of the hydrogen/water stream exiting the DRI and entering the recuperator in Kelvin. Default is 573 K. [3]
        h_activation (float): The activation energy of hydrogen in kJ/mol. Default is 35 kJ/mol. [4]
        h_endothermic (float): The reaction energy absorbed by hydrogen in kJ/mol. Default is 99.5 kJ/mol. [4]
        stream_temp_out (float): The temperature of 95% reduced iron exiting the DRI and entering the EAF in Kelvin, assuming no heat losses. Default is 973 K. [3]

    Sources:
        [3]: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.
        [4]: haskar, Abhinav, Assadi Mohsen, and Somehsaraei Nikpey Homam. 2020. "Decarbonization of the Iron and Steel Industry with Direct Reduction of Iron Ore with Green Hydrogen." Energies 13 (3): 758. doi: https://doi.org/10.3390/en13030758
    """

    mass_inputs: MassModelConfig
    h2_temp_in: float = 1173
    h2_temp_out: float = 573
    h_activation: float = 35
    h_endothermic: float = 99.5
    stream_temp_out: float = 973


@define
class EnergyModelOutputs:
    """Outputs from the HDRI energy model.

    Attributes:
        shaft_energy_balance (float): The energy balance of the shaft in kWh.
    """

    shaft_energy_balance: float  # kWh
    enthalpy_out: float


def energy_model(config: EnergyModelConfig) -> EnergyModelOutputs:
    """Calculates the energy balance of the HDRI shaft. Negative values indicate heat leaving the system, while positive values indicate heat entering the system.

    Note: Energy balance values are typically expected to be negative, reflecting heat loss from the system.

    Args:
        config (EnergyModelConfig): Configuration settings for the energy model, including temperature, activation energy, and stream temperatures.

    Returns:
        EnergyModelOutputs: An object containing the calculated energy balance of the HDRI shaft in kWh.

    Sources:
        Model derived from Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022.
        "Decarbonizing primary steel production: Techno-economic assessment of a hydrogen-based green steel production plant in Norway."
        Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.
    """
    mass_inputs = config.mass_inputs
    m = mass_model(mass_inputs)

    enthalpy_h2_input = (
        h2_enthalpy(config.h2_temp_in) * m.hydrogen_gas_needed * 1000
    )  # kJ

    h5_h20 = (
        m.water_leaving_shaft * h2o_enthalpy(config.h2_temp_out) * 1000
    )  # kJ enthalpy of H2O in exhaust
    h5_h2 = (
        m.hydrogen_gas_leaving_shaft * h2_enthalpy(config.h2_temp_out) * 1000
    )  # kJ enthalpy of H2 in exhaust

    enthalpy_out_stream = h5_h20 + h5_h2  # kJ enthalpy of exhaust

    h_reaction = config.h_activation + config.h_endothermic  # energy needed per 1 mol
    h_reaction_total = (
        h_reaction * m.iron_ore_mass_needed * 1000 * config.mass_inputs.fe2o3_pure
    ) / config.mass_inputs.mol_weight_fe2o3  # total energy per tonne steel

    enthalpy_stream_output = 1000 * (
        fe_enthalpy(config.stream_temp_out) * m.pure_iron_leaving_shaft
        + feo_enthalpy(config.stream_temp_out) * m.iron_ore_leaving_shaft
        + sio2_enthalpy(config.stream_temp_out) * m.silicon_dioxide_mass
        + al2o3_enthalpy(config.stream_temp_out) * m.aluminium_oxide_mass
    )  # Enthalpy of metallic stream exiting DRI/exiting EAF in kJ/kg

    energy_balance = (
        enthalpy_h2_input
        - (enthalpy_out_stream + enthalpy_stream_output + h_reaction_total)
    ) / 3600  # kWh

    return EnergyModelOutputs(shaft_energy_balance=energy_balance,
                              enthalpy_out=enthalpy_out_stream)

e.entha


def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """
    save_outputs_dict = dict()

    save_outputs_dict["Steel Output Desired (kg)"] = list()
    save_outputs_dict["Iron Ore Mass Needed (kg)"] = list()
    save_outputs_dict["Hydrogen Gas Needed (kg)"] = list()
    save_outputs_dict["Hydrogen Gas Leaving Shaft (kg)"] = list()
    save_outputs_dict["Mass H2O Leaving Shaft (kg)"] = list()
    save_outputs_dict["Mass Pure Iron Leaving Shaft (kg)"] = list()
    save_outputs_dict["Total Mass H2 H2O Leaving Shaft (kg)"] = list()
    save_outputs_dict["Mass Iron Ore Leaving Shaft (kg)"] = list()
    save_outputs_dict["Energy Balance of Shaft (kWh)"] = list()
    save_outputs_dict["Shaft Total Capital Cost (Mil USD)"] = list()
    save_outputs_dict["Shaft Operational Cost (Mil USD per year)"] = list()
    save_outputs_dict["Shaft Maintenance Cost (Mil USD per year)"] = list()
    save_outputs_dict["Shaft Depreciation Cost (Mil USD per year)"] = list()
    save_outputs_dict["Total Iron Ore Cost (Mil USD per year)"] = list()
    save_outputs_dict["Electricity Needed for Heater (kWh per desired output)"] = list()
    save_outputs_dict["Total Labor Cost (Mil USD per year)"] = list()

    return save_outputs_dict


class hdri_model:
    """
    Author: Charles Kiefer
    Date: 8/14/23

    This class holds functions relating to the modeling of a Hydrogen Direct Reduced Iron Shaft
    in relation to the production of Green Steel.  Iron ore enters the shaft and is reduced by hydrogen
    gas.  Pure iron is the result.

    Sources will be in each function

    Args:
    steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output
    steel_prod_yr (ton/yr): (float) plant capacity steel produced per year. For financial outputs

    returns:
    save_outputs_dict saves returns, other returns are for EAF Model

    Steel Output Desired (kg):
        ['Iron Ore Mass Needed (kg)']
        ['Hydrogen Gas Needed (kg)']
        ['Hydrogen Gas Leaving Shaft (kg)']
        ['Mass H2O Leaving Shaft (kg)']
        ['Mass Pure Iron Leaving Shaft (kg)']
        ['Total Mass H2 H2O Leaving Shaft (kg)']
        ['Mass Iron Ore Leaving Shaft (kg)']
        ['Energy Balance of Shaft (kWh)']
        ['Shaft Total Capital Cost (Mil USD)']
        ['Shaft Operational Cost (Mil USD per year)']
        ['Shaft Maintenance Cost (Mil USD per year)']
        ['Shaft Depreciation Cost (Mil USD per year)']
        ['Total Iron Ore Cost (USD per year)']
        ['Electricity Needed for Heater (kWh per desired output)']:
            -Total electricity needed for whatever the desired output steel value is
    """

    def __init__(self):
        """
        Initializes and stores needed data for functions relating to HDRI Shaft furnace
        for Green Steel

        Sources:
        [1]: Chase, M.W., Jr. 1998. "NIST-JANAF Themochemical Tables, Fourth Edition." J. Phys. Chem. Ref. Data, Monograph 9 1-1951. doi:https://doi.org/10.18434/T4D303.
        [2]: Midrex Technologies, Inc. 2018. "DRI Products + Applications: Providing flexibility for steelmaking." April. Accessed May 2, 2022. https://www.midrex.com/wp-content/uploads/MIdrexDRI_ProductsBrochure_4-12-18.pdf.
        [3]: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.
        [4]: haskar, Abhinav, Assadi Mohsen, and Somehsaraei Nikpey Homam. 2020. "Decarbonization of the Iron and Steel Industry with Direct Reduction of Iron Ore with Green Hydrogen." Energies 13 (3): 758. doi: https://doi.org/10.3390/en13030758
        """

        self.mol_weight_fe = 55.845  # (grams/mol) [1]
        self.mol_weight_fe2o3 = 159.69  # (grams/mol) [1]
        self.mol_weight_h2 = 2.01588  # (grams/mol) [1]
        self.mol_weight_h2o = 18.0153  # (grams/mol) [1]

        self.metallization_rate = 0.94  # (%) Metallization rate of DRI [2]
        self.lambda_h2 = 1.2  # (No Units) Multiplier Extra H2 is needed. Lambda is ratio of actual over stoichiometric requirement

        self.fe2o3_pure = 0.95  # (%) ammount of Fe2O3 in raw material. accounts for 5% impurities in raw material [2]
        self.sio2_percent = 0.03  # (%) percent sio2 in raw materials [2]
        self.al2o3_percent = 0.02  # (%) percent al2o3 in raw materials [2]

        self.h2_per_mol = 3 / 2  # (no units) 3 mol of H2 per 2 mol Fe stoichiometry

        self.steel_out_desired = None  # (kg) standard is 1000 kg or one metric tonne

        self.mass_iron_ore_input = None  # (kg)
        self.mass_pure_fe_output = None  # (kg)
        self.mass_h2_input = None  # (kg)
        self.mass_h2_output = None  # (kg)
        self.mass_h2o_output = None  # (kg)
        self.mass_h2_h2o_output = None  # (kg)
        self.mass_iron_ore_output = None  # (kg)
        self.mass_sio2 = None  # (kg)
        self.mass_al2o3 = None  # (kg)

        self.h2_temp_in = 1173  # (K) h2 temp needed into shaft [3]
        self.h2_temp_out = 573  # (K) H2/H20 stream exiting DRI/entering recuperator [3]
        self.stream_temp_out = 973  # (K) 95% reduced Iron exiting DRI/Entering EAF !!Assuming 0 heat Losses!! [3]
        self.h2_temp_elec = 343  # (K) Temp of hydrogen leaving electrolyzer [3]
        self.temp_stream_exit_recup = (
            393  # (K) Stream leaving recuperator/entering condenser [3]
        )
        self.temp_input_heater = 413  # (K) H2 entering heater from recuperator [3]

        self.enthalpy_h2_input = None  # (kJ/kg)
        self.enthalpy_out_stream = None  # (kJ/kg)
        self.enthalpy_stream_output = None  # (kJ/kg)

        self.h_activation = 35  # (kJ/mol) activation energy of hydrogen [4]
        self.h_endothermic = 99.5  # (kJ/mol) reaction energy absorbed of hydrogen [4]
        self.energy_balance = (
            None  # (kWh) Energy Equation (Negative denotes heat leaving system)
        )

        self.eta_el_heater = 0.6  # (%) efficiency of electricl heater
        self.el_needed_heater = None  # (kWh) electricity need by heater

        self.lang_factor = 3  # (no units) Capital cost multiplier [3]
        self.plant_life = 40  # (years) [3]
        self.hdri_total_capital_cost = None  # (Million USD)
        self.hdri_cost_per_ton_yr = 80  # (USD/tls/yr)
        self.hdri_op_cost_tls = 13  # (tls/yr of dri)
        self.hdri_operational_cost_yr = None  # (Million USD)
        self.maintenance_cost_percent = 0.015  # (% of capital cost)[3]
        self.hdri_maintenance_cost_yr = None  # (Million USD)
        self.depreciation_cost = None  # (Million USD)
        self.iron_ore_cost_tls = 90  # (USD/ton) [3]
        self.iron_ore_total_cost_yr = None  # (Million USD)
        self.total_labor_cost_yr = None  # (Million USD)
        self.labor_cost_tls = (
            20  # (USD/ton/year) [3] $40 is flat rate for hdri and eaf together
        )

    def mass_model(self, steel_out_desired):
        """
        Mass model calculates the masses inputted and outputted of
        an HDRI oxidation system of iron ore in the greensteel process for output of tonne liquid steel

        Args:
        steel_out_desired (kg) or (kg/hr): (float) resulting desired steel output

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.
        """

        m3 = steel_out_desired

        self.steel_out_desired = m3  # kg

        alpha = self.metallization_rate  # Metallization rate of DRI

        fe_o_ratio = (
            2 * self.mol_weight_fe
        ) / self.mol_weight_fe2o3  # ratio of fe weight to fe2o3 weight

        m1 = m3 / (
            self.fe2o3_pure * fe_o_ratio * alpha
        )  # kg fe2o3   amount of raw material needed for tonne steel, kg
        m2_feo = (
            m1 * self.fe2o3_pure * fe_o_ratio * (1 - alpha)
        )  # kg fe2o3 amount of IronOxide not ocidized

        sio2_percent = self.sio2_percent  # percent sio2 in raw materials
        al2o3_percent = self.al2o3_percent  # percent al2o3 in raw materials

        m1_sio2 = (
            sio2_percent * m1
        )  # kg sio2 mass of sio2 in raw materials kg per input
        m1_al2o3 = (
            al2o3_percent * m1
        )  # kg al2o3 mass of al2o3 in raw materials kg per desired output

        m2_fe = (
            m1 - (m1_sio2 + m1_al2o3 + m2_feo)
        ) * fe_o_ratio  # kg fe mass metallic iron out of DRI per tonne steel

        self.mass_sio2 = m1_sio2
        self.mass_al2o3 = m1_al2o3
        self.mass_iron_ore_output = m2_feo
        self.mass_pure_fe_output = m2_fe
        self.mass_iron_ore_input = m1

        h2_weight_per_mol = self.h2_per_mol * self.mol_weight_h2  # g h2 per 1 mol fe
        mol_per_input_fe = (
            m3 * 1000
        ) / self.mol_weight_fe  # mols of iron in a input kg of fe
        m4_stoich = (
            h2_weight_per_mol * mol_per_input_fe
        ) / 1000  # kg h2stoich minimum mass of H2 needed per tonne steel

        m4 = (
            m4_stoich * self.lambda_h2
        )  # kg h2 mass of hydrogen inputted into DRI in Kg per tonne steel

        self.mass_h2_input = m4

        water_mass = (
            (3 * self.mol_weight_h2o) / (2 * self.mol_weight_fe)
        ) * m3  # water produced per tonne liquid steel

        m5_h2 = m4_stoich * (
            self.lambda_h2 - 1
        )  # mass of hyrdogen remaing in exhaust after DRI in kg
        m5_h2o = water_mass  # Mass water produced in DRI in kg
        m5 = m5_h2 + m5_h2o  # Total Mass of exhaust in kg

        self.mass_h2_output = m5_h2
        self.mass_h2o_output = m5_h2o
        self.mass_h2_h2o_output = m5

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict["Steel Output Desired (kg)"].append(self.steel_out_desired)
        save_outputs_dict["Iron Ore Mass Needed (kg)"].append(self.mass_iron_ore_input)
        save_outputs_dict["Hydrogen Gas Needed (kg)"].append(self.mass_h2_input)
        save_outputs_dict["Hydrogen Gas Leaving Shaft (kg)"].append(self.mass_h2_output)
        save_outputs_dict["Mass H2O Leaving Shaft (kg)"].append(self.mass_h2o_output)
        save_outputs_dict["Mass Pure Iron Leaving Shaft (kg)"].append(
            self.mass_pure_fe_output
        )
        save_outputs_dict["Total Mass H2 H2O Leaving Shaft (kg)"].append(
            self.mass_h2_h2o_output
        )
        save_outputs_dict["Mass Iron Ore Leaving Shaft (kg)"].append(
            self.mass_iron_ore_output
        )

        return (
            save_outputs_dict,
            m2_fe,
            m2_feo,
            self.steel_out_desired,
            self.mass_iron_ore_input,
            self.mass_h2_input,
            self.mass_h2_output,
            self.mass_h2o_output,
            self.mass_pure_fe_output,
            self.mass_h2_h2o_output,
            self.mass_iron_ore_output,
        )

    def energy_model(self, steel_out_desired):
        """
        This function calculates the energy balance of the hdri shaft. Negative values designate
        heat leaving the system. Positive values designate heat needs to enter the system.

        Energy belance values should be negative.

        Args:
        steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.


        """
        # from greenheart.simulation.technologies.steel.enthalpy_functions import h2_enthalpy, h2_enthalpy, h2o_enthalpy, sio2_enthalpy, al2o3_enthalpy, feo_enthalpy ,fe_enthalpy

        hdri_model.mass_model(self, steel_out_desired)

        mass_h2_input = self.mass_h2_input
        mass_h2o_output = self.mass_h2o_output
        mass_h2_output = self.mass_h2_output

        h4 = h2_enthalpy(self.h2_temp_in) * mass_h2_input * 1000  # kJ
        # h4_kwh = h4/3600  #3600 kJ in 1 kWh

        self.enthalpy_h2_input = h4  # kJ

        h5_h20 = (
            mass_h2o_output * h2o_enthalpy(self.h2_temp_out) * 1000
        )  # kJ Enthalpyof h20 in exhaust
        h5_h2 = (
            mass_h2_output * h2_enthalpy(self.h2_temp_out) * 1000
        )  # kJ Enthalpy of H2 in exhaust

        h5 = h5_h20 + h5_h2  # kJ Enthalpy of exhaust
        # h5_kwh = h5/3600                          #Conversion Kj to Kwh

        self.enthalpy_out_stream = h5

        h_reaction = self.h_activation + self.h_endothermic  # energy needed per 1 mol
        h_reaction_total = (
            h_reaction * self.mass_iron_ore_input * 1000 * self.fe2o3_pure
        ) / self.mol_weight_fe2o3  # total energy per tonne steel
        # h_reaction_total_kwh=h_reaction_total/3600  #kJ to kwh conversion

        h2 = 1000 * (
            (fe_enthalpy(self.stream_temp_out) * self.mass_pure_fe_output)
            + (feo_enthalpy(self.stream_temp_out) * self.mass_iron_ore_output)
            + (sio2_enthalpy(self.stream_temp_out) * self.mass_sio2)
            + (al2o3_enthalpy(self.stream_temp_out) * self.mass_al2o3)
        )  # Enthalpy of metallic stream exiting DRI/exiting EAF in kj/kg

        # h2_kwh=h2/3600      #conversion kj to kwh

        self.enthalpy_stream_output = h2

        q_dri = (
            (self.enthalpy_h2_input)
            - (self.enthalpy_out_stream + h2 + h_reaction_total)
        ) / 3600  # Enthalpy of Hydrogen minus enthalpy of exhaust, metal stream, reaction enthalpy

        self.energy_balance = q_dri  # kwh

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict["Energy Balance of Shaft (kWh)"].append(self.energy_balance)

        return (save_outputs_dict, self.energy_balance)

    def financial_model(self, steel_prod_yr):
        """
        This function returns the financials for a rated capacity plant for HDRI shaft

        Args:
        steel_prod_yr (ton/yr): (float) plant capacity steel produced per year. For financial outputs

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

        """
        hdri_model.mass_model(self, steel_prod_yr)

        self.hdri_total_capital_cost = (
            (self.hdri_cost_per_ton_yr * steel_prod_yr) / 10**6
        ) * self.lang_factor  # Mil USD

        self.hdri_operational_cost_yr = (
            self.hdri_op_cost_tls * steel_prod_yr / (10**6)
        )  # Mil USD per year

        self.hdri_maintenance_cost_yr = (
            self.maintenance_cost_percent * self.hdri_total_capital_cost
        )  # Mil USD per year

        self.depreciation_cost = (
            self.hdri_total_capital_cost / self.plant_life
        )  # Mil USD per year

        total_iron_ore = self.mass_iron_ore_input  # tonne FeO

        self.iron_ore_total_cost_yr = (
            self.iron_ore_cost_tls * total_iron_ore / 10**6
        )  # USD per ton

        self.total_labor_cost_yr = self.labor_cost_tls * steel_prod_yr / 10**6

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict["Shaft Total Capital Cost (Mil USD)"].append(
            self.hdri_total_capital_cost
        )
        save_outputs_dict["Shaft Operational Cost (Mil USD per year)"].append(
            self.hdri_operational_cost_yr
        )
        save_outputs_dict["Shaft Maintenance Cost (Mil USD per year)"].append(
            self.hdri_maintenance_cost_yr
        )
        save_outputs_dict["Shaft Depreciation Cost (Mil USD per year)"].append(
            self.depreciation_cost
        )
        save_outputs_dict["Total Iron Ore Cost (Mil USD per year)"].append(
            self.iron_ore_total_cost_yr
        )
        save_outputs_dict["Total Labor Cost (Mil USD per year)"].append(
            self.total_labor_cost_yr
        )

        # labour costs will need to be done separately as they are flat rate

        return (
            save_outputs_dict,
            self.hdri_total_capital_cost,
            self.hdri_operational_cost_yr,
            self.hdri_maintenance_cost_yr,
            self.depreciation_cost,
            self.iron_ore_total_cost_yr,
            self.total_labor_cost_yr,
        )

    def recuperator_mass_energy_model(self, steel_out_desired):
        """
        Accessory process for heat exchanger.  Currently has no outputs as recuperator doesn't change masses

        Args:
        steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

        """
        # from greenheart.simulation.technologies.steel.enthalpy_functions import h2o_enthalpy, h2_enthalpy

        hdri_model.mass_model(self, steel_out_desired)
        hdri_model.energy_model(self, steel_out_desired)

        m10 = self.mass_h2_input  # hydrogen from electrolyzer = hydrogen into DRI

        m12_h2o = (
            self.mass_h2o_output
        )  # Mass h2o of exhaust stream = mass h2o to condensor
        m12_h2 = self.mass_h2_output  # Mass h2 of exhaust stream = mass h2
        # m13 = m12_h2o     #Mass h2o from condenser to electrolyzer

        h12_h2o = (
            m12_h2o * h2o_enthalpy(self.temp_stream_exit_recup) * 1000
        )  # exit h20 in recuperator to condenser stream
        h12_h2 = (
            m12_h2 * h2_enthalpy(self.temp_stream_exit_recup) * 1000
        )  # exit h2 in recuperator to condenser stream
        h12 = h12_h2 + h12_h2o  # total enthalpy in exit of recuperator to condensor
        h12_kwh = h12 / 3600  # conversion kj to kwh

        h10 = h2_enthalpy(self.h2_temp_elec) * m10 * 1000  # electrolyzer to recuperator
        h10_kwh = h10 / 3600  # conversion kj to kwh

        h11 = ((self.enthalpy_out_stream - h12) + h10) / 3600

        save_outputs_dict = establish_save_output_dict()

        return (save_outputs_dict, h11)

    def heater_mass_energy_model(self, steel_out_desired):
        """
        Function returns the energy needed to heat up gas to needed temp for oxidation

        Args:
        steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

        """
        # from hopp.simulation.technologies.steel.enthalpy_functions import h2_enthalpy

        hdri_model.mass_model(self, steel_out_desired)
        hdri_model.energy_model(self, steel_out_desired)

        T11_heat_in = (
            self.temp_input_heater
        )  # Assumes 30 degree heat loss from recuperator to heater
        m11_heat_in = (
            self.mass_h2_input
        )  # mass of hydrogen into heater = mass hydrogen into recuperator

        h11_heat_in = (
            h2_enthalpy(T11_heat_in) * m11_heat_in * 1000
        )  # enthalpy of stream into heater
        # h11_heat_in_kwh = (h11_heat_in/3600)      #kj to kwh conversion

        # eta_rec = ((h11_heat_in - h10)/(h5-h12))    #recuperator efficiency
        q_heater = (
            self.enthalpy_h2_input - h11_heat_in
        ) / 3600  # energy needed to be provided to heater

        eta_el_heater = self.eta_el_heater  # Efficiency of heater
        el_heater = q_heater / eta_el_heater  # electricity need at heater

        self.el_needed_heater = el_heater  # kWh or kw

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict[
            "Electricity Needed for Heater (kWh per desired output)"
        ].append(self.el_needed_heater)

        return (save_outputs_dict, self.el_needed_heater)


if __name__ == "__main__":

    model_instance = hdri_model()

    steel_output_desired = 1000  # (kg or kg/hr)

    # mass_outputs = model_instance.mass_model(steel_output_desired)
    # energy_outputs = model_instance.energy_model(steel_output_desired)
    recuperator_outputs = model_instance.recuperator_mass_energy_model(steel_output_desired)
    # heater_outputs = model_instance.heater_mass_energy_model(steel_output_desired)

    # steel_output_desired_yr = 2000000 #(ton/yr)

    # financial_outputs = model_instance.financial_model(steel_output_desired_yr)
    print()
    print(recuperator_outputs)

    config = MassModelConfig(steel_output_desired=1000)
    out = mass_model(config)
    print()
    print(out)

    config1 = EnergyModelConfig(mass_inputs=MassModelConfig(steel_output_desired=1000))
    out1 = energy_model(config1)
    print()
    print(out1)
