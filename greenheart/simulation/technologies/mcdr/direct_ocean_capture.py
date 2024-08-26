"""
Initial model of electrochemical mCC system (Time-Dependent Pre-Calculation Version with Generic Data & CSV Output) 
Date: 6/11/2024
Author: James Niffenegger
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from attrs import define, field


@define
class ElectroDialysisInputs:
    """
    A class to represent the inputs for an Electrodialysis (ED) unit.

    Attributes:
        P_ed1 (float): Power needed for one ED unit in watts (W). Default is 50e6.
        Q_ed1 (float): Flow rate for one ED unit in cubic meters per second (m^3/s). Default is 1.
        N_edMin (int): Minimum number of ED units. Default is 1.
        N_edMax (int): Maximum number of ED units. Default is 5.
        E_HCl (float): Energy required by the ED unit to process HCl in kilowatt-hours per mole (kWh/mol). Default is 0.05.
        E_NaOH (float): Energy required by the ED unit to process NaOH in kilowatt-hours per mole (kWh/mol). Default is 0.05.
        y_ext (float): CO2 extraction efficiency (%). Default is 0.9.
        y_vac (float): Vacuum efficiency (%). Default is 0.3.
        co2_mm (float): Molar mass of CO2 in grams per mole (g/mol). Default is 44.01.
    """

    P_ed1: float = 50 * 10**6
    Q_ed1: float = 1
    N_edMin: int = 1
    N_edMax: int = 5
    E_HCl: float = 0.05
    E_NaOH: float = 0.05
    y_ext: float = 0.9
    y_vac = 0.3
    co2_mm: float = field(init=False, default=44.01)  # g/mol molar mass of CO2
    P_minED: float = field(init=False)

    def __attrs_post_init__(self):
        self.P_minED = self.P_ed1 * self.N_edMin  # Min ED Power (MW)


@define
class SeaWaterInputs:
    """
    A class to represent the initial inputs for seawater chemistry.

    Attributes:
        dic_i (float): Initial concentration of dissolved inorganic carbon in mol/L.
            Default is 2.2e-3. (3.12 mM in Instant Ocean, 2.2 mM in seawater)
        pH_i (float): Initial pH of seawater. Default is 8.1.
        pH_eq2 (float): Second equivalence point of seawater pH. Default is 4.3.
        k1 (float): First dissociation constant of carbonic acid in mol/L.
            Default is 1.38e-6 (10^-5.86).
        k2 (float): Second dissociation constant of carbonic acid in mol/L.
            Default is 1.2e-9 (10^-8.92).
        kw (float): Water dissociation constant at 25°C and a salinity of 3 in mol/L.
            Default is 6.02e-14 (10^-13.22).
    """

    dic_i: float = 2.2 * 10**-3
    pH_i: float = 8.1
    pH_eq2: float = 4.3
    k1: float = 10**-5.86
    k2: float = 10**-8.92
    kw: float = 10**-13.22

    h_i: float = field(init=False)
    h_eq2: float = field(init=False)
    ta_i: float = field(init=False)

    def __attrs_post_init__(self):
        self.h_i = 10**-self.pH_i
        self.h_eq2 = 10**-self.pH_eq2

        # Initial TA (total alkalinity concentration) (mol/L)
        self.ta_i = findTA(self, self.dic_i,self.h_i)

def findTA(seawater: SeaWaterInputs, dic, h):
    k1 = seawater.k1
    k2 = seawater.k2
    kw = seawater.kw
    ta = (dic/(1+(h/k1)+(k2/h))) + (2*dic/(1+(h/k2)+((h**2)/(k1*k2)))) + kw/h - h
    return ta

# Find H+ from TA and DIC (mol/L)
def findH_TA(seawater: SeaWaterInputs, dic, ta, ph_min, ph_max, step):
    ph_range = np.arange(ph_min, ph_max + step, step)
    ta_error = np.zeros(len(ph_range))
    for i in range(len(ph_range)):
        h_est = 10**-ph_range[i]
        ta_est = findTA(seawater, dic,h_est)
        ta_error[i] = abs(ta - ta_est)
    for i in range(len(ph_range)):
        if ta_error[i] == min(ta_error):
            i_ph = i
    ph_f = ph_range[i_ph]
    h_f = 10**-ph_f
    return h_f

# Find CO2 conc from DIC and H+ Function (mol/L)
def findCO2(seawater: SeaWaterInputs, dic,h):
    k1 = seawater.k1
    k2 = seawater.k2
    co2 = dic/(1+(k1/h)+((k1*k2)/(h**2)))
    return co2

# Find H+ from CO2 conc and DIC (mol/L)
def findH_CO2(seawater: SeaWaterInputs, dic, co2, ph_min, ph_max, step):
    ph_range = np.arange(ph_min, ph_max + step, step)
    co2_error = np.zeros(len(ph_range))
    for i in range(len(ph_range)):
        h_est = 10**-ph_range[i]
        co2_est = findCO2(seawater, dic,h_est)
        co2_error[i] = abs(co2 - co2_est)
    for i in range(len(ph_range)):
        if co2_error[i] == min(co2_error):
            i_ph = i
    ph_f = ph_range[i_ph]
    h_f = 10**-ph_f
    return h_f

@define
class PumpInputs:
    """
    A class to define the input parameters for various pumps in the system.

    Attributes:
        y_pump (float): The constant efficiency of the pump. Default is 0.9.
        p_o_min_bar (float): The minimum pressure (in bar) for seawater intake with filtration. Default is 1.
        p_o_max_bar (float): The maximum pressure (in bar) for seawater intake with filtration. Default is 2.
        p_ed_min_bar (float): The minimum pressure (in bar) for ED (Electrodialysis) units. Default is 0.5.
        p_ed_max_bar (float): The maximum pressure (in bar) for ED (Electrodialysis) units. Default is 1.
        p_a_min_bar (float): The minimum pressure (in bar) for pumping acid. Default is 0.5.
        p_a_max_bar (float): The maximum pressure (in bar) for pumping acid. Default is 1.
        p_i_min_bar (float): The minimum pressure (in bar) for pumping seawater for acid addition. Default is 0.
        p_i_max_bar (float): The maximum pressure (in bar) for pumping seawater for acid addition. Default is 0.
        p_co2_min_bar (float): The minimum pressure (in bar) for pumping seawater for CO2 extraction. Default is 0.5.
        p_co2_max_bar (float): The maximum pressure (in bar) for pumping seawater for CO2 extraction. Default is 1.
        p_asw_min_bar (float): The minimum pressure (in bar) for pumping seawater for base addition. Default is 0.
        p_asw_max_bar (float): The maximum pressure (in bar) for pumping seawater for base addition. Default is 0.
        p_b_min_bar (float): The minimum pressure (in bar) for pumping base. Default is 0.5.
        p_b_max_bar (float): The maximum pressure (in bar) for pumping base. Default is 1.
        p_f_min_bar (float): The minimum pressure (in bar) for released seawater. Default is 0.
        p_f_max_bar (float): The maximum pressure (in bar) for released seawater. Default is 0.
    """

    y_pump: float = 0.9
    p_o_min_bar: float = 1
    p_o_max_bar: float = 2
    p_ed_min_bar: float = 0.5
    p_ed_max_bar: float = 1
    p_a_min_bar: float = 0.5
    p_a_max_bar: float = 1
    p_i_min_bar: float = 0
    p_i_max_bar: float = 0
    p_co2_min_bar: float = 0.5
    p_co2_max_bar: float = 1
    p_asw_min_bar: float = 0
    p_asw_max_bar: float = 0
    p_b_min_bar: float = 0.5
    p_b_max_bar: float = 1
    p_f_min_bar: float = 0
    p_f_max_bar: float = 0


@define
class Pump:
    """
    A class to represent a pump with specific flow rate and pressure characteristics.

    Attributes:
        Q_min (float): Minimum flow rate (m³/s).
        Q_max (float): Maximum flow rate (m³/s).
        p_min_bar (float): Minimum pressure (bar).
        p_max_bar (float): Maximum pressure (bar).
        eff (float): Efficiency of the pump.
        Q (float): Instantaneous flow rate (m³/s), initially set to zero.
    """

    Q_min: float  # Minimum flow rate (m³/s)
    Q_max: float  # Maximum flow rate (m³/s)
    p_min_bar: float  # Minimum pressure (bar)
    p_max_bar: float  # Maximum pressure (bar)
    eff: float  # Efficiency
    Q: float = field(default=0)  # Instantaneous flow rate (m³/s), initially set to zero

    def pumpPower(self, Q):
        """
        Calculate the power required for the pump based on the flow rate.

        Args:
            Q (float): Flow rate (m³/s).

        Returns:
            float: Power required for the pump (W).

        Raises:
            ValueError: If the flow rate is out of the specified range or if the minimum pressure is greater than the maximum pressure.
        """
        if Q == 0:
            P_pump = 0
        elif Q < self.Q_min or Q > self.Q_max:
            raise ValueError("Flow Rate Out of Range Provided for Pump Power")
        elif self.p_min_bar > self.p_max_bar:
            raise ValueError(
                "Minimum Pressure Must Be Less Than or Equal to Maximum Pressure for Pump"
            )
        elif self.Q_min == self.Q_max:
            p_bar = (
                self.p_min_bar + self.p_max_bar
            ) / 2  # average pressure used if the flow rate is constant
            p = p_bar * 100000  # convert from bar to Pa
            P_pump = Q * p / self.eff
        else:
            perc_range = (Q - self.Q_min) / (self.Q_max - self.Q_min)
            p_bar = (self.p_max_bar - self.p_min_bar) * perc_range + self.p_min_bar
            p = p_bar * 100000  # convert from bar to Pa
            P_pump = Q * p / self.eff
        return P_pump

    @property
    def P_min(self):
        """
        Calculate the minimum power required for the pump.

        Returns:
            float: Minimum power required for the pump (W).
        """
        return self.pumpPower(self.Q_min)

    @property
    def P_max(self):
        """
        Calculate the maximum power required for the pump.

        Returns:
            float: Maximum power required for the pump (W).
        """
        return self.pumpPower(self.Q_max)

    def power(self):
        """
        Calculate the power required for the pump based on the current flow rate.

        Returns:
            float: Power required for the pump (W).
        """
        return self.pumpPower(self.Q)


@define
class PumpOutputs:
    """
    A class to represent the outputs of initialized pumps.

    Attributes:
        pumpO (Pump): Seawater intake pump.
        pumpED (Pump): ED pump.
        pumpA (Pump): Acid pump.
        pumpI (Pump): Pump for seawater acidification.
        pumpCO2ex (Pump): Pump for CO2 extraction.
        pumpASW (Pump): Pump for pH restoration.
        pumpB (Pump): Base pump.
        pumpF (Pump): Seawater output pump.
        pumpED4 (Pump): ED pump for S4.
    """

    pumpO: Pump
    pumpED: Pump
    pumpA: Pump
    pumpI: Pump
    pumpCO2ex: Pump
    pumpASW: Pump
    pumpB: Pump
    pumpF: Pump
    pumpED4: Pump


def initialize_pumps(
    ed_config: ElectroDialysisInputs, pump_config: PumpInputs
) -> PumpOutputs:
    """Initialize a list of Pump instances based on the provided configurations.

    Args:
        ed_config (ElectroDialysisInputs): The electro-dialysis inputs.
        pump_config (PumpInputs): The pump inputs.

    Returns:
        PumpOutputs: An instance of PumpOutputs containing all initialized pumps.
    """
    Q_ed1 = ed_config.Q_ed1
    N_edMin = ed_config.N_edMin
    N_edMax = ed_config.N_edMax
    p = pump_config
    pumpO = Pump(
        Q_ed1 * ed_config.N_edMin * 100,
        Q_ed1 * N_edMax * 100,
        p.p_o_min_bar,
        p.p_o_max_bar,
        p.y_pump,
    )  # features of seawater intake pump
    pumpED = Pump(
        Q_ed1 * N_edMin, Q_ed1 * N_edMax, p.p_ed_min_bar, p.p_ed_max_bar, p.y_pump
    )  # features of ED pump
    pumpA = Pump(
        pumpED.Q_min / 2, pumpED.Q_max / 2, p.p_a_min_bar, p.p_a_max_bar, p.y_pump
    )  # features of acid pump
    pumpI = Pump(
        pumpO.Q_min - pumpED.Q_min, pumpO.Q_max, p.p_i_min_bar, p.p_i_max_bar, p.y_pump
    )  # features of pump for seawater acidification
    pumpCO2ex = Pump(
        pumpI.Q_min + pumpA.Q_min,
        pumpI.Q_max + pumpA.Q_max,
        p.p_co2_min_bar,
        p.p_co2_max_bar,
        p.y_pump,
    )  # features of pump for CO2 extraction
    pumpASW = Pump(
        pumpCO2ex.Q_min, pumpCO2ex.Q_max, p.p_asw_min_bar, p.p_asw_max_bar, p.y_pump
    )  # features of pump for pH restoration
    pumpB = Pump(
        pumpED.Q_min - pumpA.Q_min,
        pumpED.Q_max - pumpA.Q_max,
        p.p_b_min_bar,
        p.p_b_max_bar,
        p.y_pump,
    )  # features of base pump
    pumpF = Pump(
        pumpASW.Q_min + pumpB.Q_min,
        pumpASW.Q_max + pumpB.Q_max,
        p.p_f_min_bar,
        p.p_f_max_bar,
        p.y_pump,
    )  # features of seawater output pump (note min can be less if all acid and base are used)
    pumpED4 = Pump(
        Q_ed1 * N_edMin, Q_ed1 * N_edMax, p.p_o_min_bar, p.p_o_max_bar, p.y_pump
    )  # features of ED pump for S4 (pressure of intake is used here)
    return PumpOutputs(
        pumpO=pumpO,
        pumpED=pumpED,
        pumpA=pumpA,
        pumpI=pumpI,
        pumpCO2ex=pumpCO2ex,
        pumpASW=pumpASW,
        pumpB=pumpB,
        pumpF=pumpF,
        pumpED4=pumpED4,
    )


@define
class Vacuum:
    """
    A class to represent a vacuum system for carbon capture with specific flow rate and pressure characteristics.

    Attributes:
        mCC_min (float): Minimum flow rate (m³/s).
        mCC_max (float): Maximum flow rate (m³/s).
        p_min_bar (float): Minimum pressure (bar).
        p_max_bar (float): Maximum pressure (bar).
        eff (float): Efficiency of the vacuum system.
        mCC (float): Instantaneous flow rate (m³/s), initially set to zero.
        co2_mm (float): Molar mass of CO2 (g/mol), default is 44.01.
    """

    mCC_min: float  # Minimum flow rate (m³/s)
    mCC_max: float  # Maximum flow rate (m³/s)
    p_min_bar: float  # Minimum pressure (bar)
    p_max_bar: float  # Maximum pressure (bar)
    eff: float  # Efficiency
    mCC: float = field(
        default=0
    )  # Instantaneous flow rate (m³/s), initially set to zero
    co2_mm: float = field(init=False, default=44.01)  # g/mol molar mass of CO2

    def vacPower(self, mCC: float) -> float:
        """
        Calculate the power required for the vacuum system based on the flow rate.

        Args:
            mCC (float): Flow rate (m³/s).

        Returns:
            float: Power required for the vacuum system (W).

        Raises:
            ValueError: If the flow rate is out of the specified range or if the minimum pressure is greater than the maximum pressure.
        """
        if mCC == 0:
            return 0

        if mCC < self.mCC_min or mCC > self.mCC_max:
            raise ValueError(
                "Carbon Capture Rate Out of Range Provided for Vacuum Power"
            )
        if self.p_min_bar > self.p_max_bar:
            raise ValueError(
                "Minimum Pressure Must Be Less Than or Equal to Maximum Pressure for Vacuum"
            )

        if self.mCC_min == self.mCC_max:
            p_bar = (
                self.p_min_bar + self.p_max_bar
            ) / 2  # average pressure used if the flow rate is constant
        else:
            perc_range = (mCC - self.mCC_min) / (self.mCC_max - self.mCC_min)
            p_bar = (self.p_max_bar - self.p_min_bar) * perc_range + self.p_min_bar

        R_gc = 8.314  # (J/mol*K) universal gas constant
        temp = 298  # (K) assuming temperature inside is 25C
        deltaP = p_bar * 100000  # gauge pressure in Pa
        p = (
            p_bar + 1
        ) * 100000  # convert from gauge pressure to absolute and from bar to Pa
        n_co2 = mCC / (self.co2_mm * 3600 / 10**6)  # convert tCO2/hr to mol/s
        Q_co2 = n_co2 * R_gc * temp / p  # (m³/s) flow rate using ideal gas law
        P_vac = Q_co2 * deltaP / self.eff

        return P_vac

    @property
    def P_min(self) -> float:
        """
        Calculate the minimum power required for the vacuum system.

        Returns:
            float: Minimum power required for the vacuum system (W).
        """
        return self.vacPower(self.mCC_min)

    @property
    def P_max(self) -> float:
        """
        Calculate the maximum power required for the vacuum system.

        Returns:
            float: Maximum power required for the vacuum system (W).
        """
        return self.vacPower(self.mCC_max)

    def power(self) -> float:
        """
        Calculate the instantaneous power required for the vacuum system based on the current flow rate.

        Returns:
            float: Instantaneous power required for the vacuum system (W).
        """
        return self.vacPower(self.mCC)


@define
class ElectrodialysisOutputs:
    N_ed: list  # Number of ED units active
    # P_mCC = np.zeros(len(exTime))
    P_xs: list  # (W) Excess power at each time
    # P_ed = np.zeros(len(exTime))
    pH_f: list  # Final pH at each time
    dic_f: list  # (mol/L) Final DIC at each time
    mCC_t: list  # tCO2/hr at each time
    volAcid: list  # (m^3) Volume of acid added or removed from tanks at each time
    volBase: list  # (m^3) Volume of base added  or removed from tanks at each time
    Q_in: list  # (m^3/s) Intake flow rate at each time step
    Q_out: list  # (m^3/s) Outtake flow rate at each time step
    ca_t: list  # (mol/L) Acid concentration at each time step
    cb_t: list  # (mol/L) Base concentration at each time step
    S_t: list  # Scenario for each time step
    nON: int = (
        0  # Timesteps when capture occurs (S1-3) used to determine capacity factor
    )


def initialize_power_chemical_ranges(
        ed_config: ElectroDialysisInputs, 
        pump_config: PumpInputs, 
        seawater_config: SeaWaterInputs
        ) -> ElectrodialysisOutputs:
    
    N_edMin = ed_config.N_edMin
    N_edMax = ed_config.N_edMax
    P_ed1 = ed_config.P_ed1
    Q_ed1 = ed_config.Q_ed1
    E_HCl =ed_config.E_HCl
    E_NaOH = ed_config.E_NaOH
    y_ext=ed_config.y_ext
    co2_mm = ed_config.co2_mm
    dic_i =seawater_config.dic_i
    h_i = seawater_config.h_i
    ta_i = seawater_config.ta_i
    kw=seawater_config.kw
    h_eq2 = seawater_config.h_eq2
    pH_eq2 = seawater_config.pH_eq2
    pH_i = seawater_config.pH_i



    
    # Define the range sizes
    N_range = N_edMax - N_edMin + 1
    S2_tot_range = (N_range * (N_range + 1)) // 2

    S2_ranges = np.zeros((S2_tot_range, 2))

    # Fill the ranges array
    k = 0
    for i in range(N_range):
        for j in range(N_range - i):
            S2_ranges[k, 0] = i + N_edMin
            S2_ranges[k, 1] = j
            k += 1

    # Define the array names
    keys = [
        "volAcid",
        "volBase",
        "mCC",
        "pH_f",
        "dic_f",
        "c_a",
        "c_b",
        "Qin",
        "Qout",
    ]

    # Initialize the dictionaries
    S1 = {key: np.zeros(N_range) for key in keys}
    S2 = {key: np.zeros(S2_tot_range) for key in keys}
    S3 = {key: np.zeros(N_range) for key in keys}
    S4 = {key: np.zeros(N_range) for key in keys}

    p = initialize_pumps(
        ed_config=ed_config, pump_config=pump_config
    )

################################ Chemical Ranges: S1, S3, S4 #####################################
    for i in range(N_range):
############################### S1: Chem Ranges: Tank Filled #####################################
        P_EDi = (i+N_edMin)*P_ed1 # ED unit power requirements
        p.pumpED.Q = (i+N_edMin)*Q_ed1 # Flow rates for ED Units
        p.pumpO.Q = 100*p.pumpED.Q # Intake is 100x larger than ED unit
        S1['Qin'][i] = p.pumpO.Q # (m3/s) Intake

        # Acid and Base Concentrations
        p.pumpA.Q = p.pumpED.Q/2 # Acid flow rate
        C_a = (1/p.pumpA.Q)*(P_EDi/(3600*(E_HCl*1000))-(p.pumpED.Q*h_i*1000)) # (mol/m3) Acid concentration from ED units
        S1['c_a'][i] = C_a/1000 # (mol/L) Acid concentration from ED units

        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q # Base flow rate
        C_b = (1/p.pumpB.Q)*(P_EDi/(3600*(E_NaOH*1000))-(p.pumpED.Q*(kw/h_i)*1000)) # (mol/m3) Base concentration from ED units
        S1['c_b'][i] = C_b/1000 # (mol/L) Base concentration from ED units

        # Acid Addition
        p.pumpI.Q = p.pumpO.Q - p.pumpED.Q # Intake remaining after diversion to ED
        n_a = p.pumpA.Q * C_a # mole rate of acid (mol HCl/s)
        n_tai = ta_i * 1000 * p.pumpI.Q # mole rate of total alkalinity (mol TA/s)

        if n_a >= n_tai:
            Q_a1 = n_tai/C_a # flow rate needed to reach equivalence point (m3/s)
            Q_a2 = p.pumpA.Q - Q_a1 # remaining flow rate (m3/s)
            H_af = (h_eq2*1000*(p.pumpI.Q + Q_a1) + C_a*Q_a2)/(p.pumpA.Q + p.pumpI.Q) # (mol/m3) concentration after acid addition
        elif n_a < n_tai:
            n_TAaf = n_tai - n_a # (mol/s) remaining mole rate of total alkalinity
            TA_af = n_TAaf/(p.pumpI.Q + p.pumpA.Q) # (mol/m3) remaining concentration of total alkalinity

            H_af = (findH_TA(dic_i,TA_af/1000,pH_eq2,pH_i, 0.01)) * 1000 # (mol/m3) Function result is mol/L need mol/m3

        # Find CO2 Extracted
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q # post extraction
        CO2_af = (findCO2(seawater_config,dic_i,H_af/1000)) * 1000 # (mol/m3) Concentration of aqueous CO2 in the acidified seawater
        n_co2_h2o = CO2_af * p.pumpASW.Q # mole rate of CO2 in the water
        n_co2_ext = n_co2_h2o * y_ext # mole rate of CO2 extracted (mol/s)
        S1['mCC'][i] = n_co2_ext * co2_mm*3600/10**6 # tCO2/hr 

        # Find pH After CO2 Extraction & Before Base Addition
        CO2_bi = (1-y_ext)*CO2_af # (mol/m3) CO2 conc before base add and after CO2 extraction
        DIC_f = dic_i * 1000 - (y_ext*CO2_af) # (mol/m3) dic conc before base add and after CO2 extraction
        S1['dic_f'][i] = DIC_f/1000 # convert final DIC to mol/L
        
        H_bi = (findH_CO2(seawater_config,DIC_f/1000, CO2_bi/1000, -np.log10(H_af/1000), pH_i, 0.01)) * 1000 # (mol/m3) acidity after CO2 extraction (note min of search is the acidified seawater pH)

        # Find TA Before Base Addition
        TA_bi = (findTA(seawater_config, DIC_f/1000,H_bi/1000)) * 1000 # (mol/m3)

        # Find TA After Base Addition
        TA_bf = (TA_bi * p.pumpASW.Q + C_b * p.pumpB.Q)/(p.pumpASW.Q + p.pumpB.Q) # (mol/m3)

        # Find pH After Base Addition
        H_bf = (findH_TA(seawater_config, DIC_f/1000, TA_bf/1000, -np.log10(H_bi/1000), -np.log10(kw), 0.01)) * 1000 # (mol/m3) acidity after base addition
        S1['pH_f'][i] = -np.log10(H_bf/1000)

        # Outtake
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q # (m3/s) Outtake flow rate
        S1['Qout'][i] = p.pumpF.Q # (m3/s) Outtake

############################### S3: Chem Ranges: ED not active, tanks not zeros ##################
        P_EDi = 0 # ED Unit is off
        p.pumpED.Q = 0 # ED Unit is off
        p.pumpO.Q = 100*(i+N_edMin)*Q_ed1 # Flow rates for intake based on equivalent ED units that would be active
        S3['Qin'][i] = p.pumpO.Q
        p.pumpI.Q = p.pumpO.Q # since no flow is going to the ED unit
        p.pumpA.Q = (i+N_edMin)*Q_ed1/2 # Flow rate for acid pump based on equivalent ED units that would be active
        p.pumpB.Q = p.pumpA.Q
        
        # Change in volume due to acid and base use
        S3['volAcid'][i] = -p.pumpA.Q * 3600 # (m3) volume of acid lost by the tank
        S3['volBase'][i] = -p.pumpB.Q * 3600 # (m3) volume of base lost by the tank
        
        # The concentration of acid and base produced does not vary with flow rate since Q_a = Q_b = Q_ed/2
        # Also does not vary with power since the power for the ED units scale directly with the flow rate
        C_a = (1/p.pumpA.Q_min)*(P_ed1*N_edMin/(3600*(E_HCl*1000))-(p.pumpED.Q_min*h_i*1000)) # (mol/m3) Acid concentration from ED units
        S3['c_a'][i] = C_a/1000 # (mol/L) Acid concentration used in S3
        C_b = (1/p.pumpB.Q_min)*(P_ed1*N_edMin/(3600*(E_NaOH*1000))-(p.pumpED.Q_min*(kw/h_i)*1000)) # (mol/m3) Base concentration from ED units
        S3['c_b'][i] = C_b/1000 # (mol/L) Base concentration used in S3
        
        # Acid addition
        n_a = p.pumpA.Q * C_a # mole rate of acid (mol HCl/s)
        n_tai = ta_i * 1000 * p.pumpI.Q # mole rate of total alkalinity (mol TA/s)
        if n_a >= n_tai:
            Q_a1 = n_tai/C_a # flow rate needed to reach equivalence point (m3/s)
            Q_a2 = p.pumpA.Q - Q_a1 # remaining flow rate (m3/s)
            H_af = (h_eq2*1000*(p.pumpI.Q + Q_a1) + C_a*Q_a2)/(p.pumpA.Q + p.pumpI.Q) # (mol/m3) concentration after acid addition
        elif n_a < n_tai:
            n_TAaf = n_tai - n_a # (mol/s) remaining mole rate of total alkalinity
            TA_af = n_TAaf/(p.pumpI.Q + p.pumpA.Q) # (mol/m3) remaining concentration of total alkalinity
            H_af = (findH_TA(dic_i,TA_af/1000,pH_eq2,pH_i, 0.01)) * 1000 # (mol/m3) Function result is mol/L need mol/m3
        
        # Find CO2 Extracted
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q # post extraction
        CO2_af = (findCO2(seawater_config, dic_i,H_af/1000)) * 1000 # (mol/m3) Concentration of aqueous CO2 in the acidified seawater
        n_co2_h2o = CO2_af * p.pumpASW.Q # mole rate of CO2 in the water
        n_co2_ext = n_co2_h2o * y_ext # mole rate of CO2 extracted
        S3['mCC'][i] = n_co2_ext * co2_mm * 3600/10**6 # tCO2/step

        # Find pH After CO2 Extraction & Before Base Addition
        CO2_bi = (1-y_ext)*CO2_af # (mol/m3) CO2 conc before base add and after CO2 extraction
        DIC_f = dic_i * 1000 - (y_ext*CO2_af) # (mol/m3) dic conc before base add and after CO2 extraction
        S3['dic_f'][i] = DIC_f/1000 # (mol/L)
        H_bi = (findH_CO2(seawater_config,DIC_f/1000, CO2_bi/1000, -np.log10(H_af/1000), pH_i, 0.01)) * 1000 # (mol/m3) acidity after CO2 extraction (note min of search is the acidified seawater pH)
        # Find TA Before Base Addition
        TA_bi = (findTA(seawater_config, DIC_f/1000,H_bi/1000)) * 1000 # (mol/m3)
        # Find TA After Base Addition
        TA_bf = (TA_bi * p.pumpASW.Q + C_b * p.pumpB.Q)/(p.pumpASW.Q + p.pumpB.Q) # (mol/m3)
        # Find pH After Base Addition
        H_bf = (findH_TA(seawater_config, DIC_f/1000, TA_bf/1000, -np.log10(H_bi/1000), -np.log10(kw), 0.01)) * 1000 # (mol/m3) acidity after base addition
        S3['pH_f'][i] = -np.log10(H_bf/1000)
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q # Outtake flow rate
        S3['Qout'][i] = p.pumpF.Q

        # Define vacuum pump based on mCC ranges from S1 and S3 (S3 has a slightly higher mCC)
        vacCO2 = Vacuum(min(np.concatenate([S1['mCC'], S3['mCC']])), max(np.concatenate([S1['mCC'], S3['mCC']])), pump_config.p_co2_min_bar, pump_config.p_co2_max_bar, ed_config.y_vac)


############################### S4: Chem Ranges: ED active, no capture ###########################
        P_EDi = (i+N_edMin)*P_ed1 # ED unit power requirements
        p.pumpED.Q = 0 # Regular ED pump is inactive here
        p.pumpED4.Q = (i+N_edMin)*Q_ed1 # ED pump with filtration pressure
        
        # Acid and base concentrations
        p.pumpA.Q = p.pumpED4.Q/2 # Acid flow rate
        p.pumpB.Q = p.pumpED4.Q - p.pumpA.Q # Base flow rate
        C_a = (1/p.pumpA.Q)*(P_EDi/(3600*(E_HCl*1000))-(p.pumpED4.Q*h_i*1000)) # (mol/m3) Acid concentration from ED units
        S4['c_a'][i] = C_a/1000 # (mol/L) Acid concentration from ED units
        C_b = (1/p.pumpB.Q)*(P_EDi/(3600*(E_NaOH*1000))-(p.pumpED4.Q*(kw/h_i)*1000)) # (mol/m3) Base concentration from ED units
        S4['c_b'][i] = C_b/1000 # (mol/L) Base concentration from ED units

        # Acid added to the tank
        n_aT = C_a*p.pumpA.Q # (mol/s) rate of acid moles added to tank
        S4['volAcid'][i] = p.pumpA.Q * 3600 # volume of acid in tank after time step

        # Base added to the tank
        n_bT = C_b*p.pumpB.Q # (mol/s) rate of base moles added to tank
        S4['volBase'][i] = p.pumpB.Q * 3600 # volume of base in tank after time step

        # Intake (ED4 pump not O pump is used)
        p.pumpO.Q = 0 # Need intake for ED & min CC
        S4['Qin'][i] = p.pumpED4.Q # (m3/s) Intake
        
        # Other pumps not used
        p.pumpI.Q = 0 # Intake remaining after diversion to ED
        p.pumpCO2ex.Q = 0 # Acid addition
        p.pumpASW.Q = 0 # post extraction
        
        # Outtake
        p.pumpF.Q = 0 # Outtake flow rate
        S4['Qout'][i] = p.pumpF.Q # (m3/s) Outtake

        # Since no capture is conducted the final DIC and pH is the same as the initial
        S4['pH_f'][i] = pH_i 
        S4['dic_f'][i] = dic_i # (mol/L)

################################ Chemical Ranges: S2 #############################################
    for i in range(S2_tot_range):
        # ED Unit Characteristics
        N_edi = S2_ranges[i,0] + S2_ranges[i,1] 
        P_EDi = (N_edi)*P_ed1 # ED unit power requirements
        p.pumpED.Q = (N_edi)*Q_ed1 # Flow rates for ED Units
        
        # Acid and Base Creation
        p.pumpA.Q = p.pumpED.Q/2 # Acid flow rate
        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q # Base flow rate
        C_a = (1/p.pumpA.Q)*(P_EDi/(3600*(E_HCl*1000))-(p.pumpED.Q*h_i*1000)) # (mol/m3) Acid concentration from ED units
        S2['c_a'][i] = C_a/1000 # (mol/L) Acid concentration from ED units
        C_b = (1/p.pumpB.Q)*(P_EDi/(3600*(E_NaOH*1000))-(p.pumpED.Q*(kw/h_i)*1000)) # (mol/m3) Base concentration from ED units
        S2['c_b'][i] = C_b/1000 # (mol/L) Base concentration from ED units

        # Amount of acid added for mCC
        Q_aMCC = S2_ranges[i,0] * Q_ed1/2 # flow rate used for mCC

        # Acid addition to tank (base volume will be the same)
        Q_aT = p.pumpA.Q - Q_aMCC # (m3/s) flow rate of acid to tank
        n_aT = C_a*Q_aT # (mol/s) rate of acid moles added to tank
        S2['volAcid'][i] = Q_aT * 3600 # (m3) acid added to tank

        # Seawater Intake
        p.pumpO.Q = Q_aMCC * 2 * 100 + (p.pumpED.Q - (Q_aMCC * 2)) # total seawater intake
        S2['Qin'][i] = p.pumpO.Q # (m3/s) intake
        
        # Acid addition to seawater
        p.pumpI.Q = p.pumpO.Q - p.pumpED.Q # seawater that will recieve acid

        # Acid Flow Rate for mCC Chemistry Calcs
        p.pumpA.Q = Q_aMCC # flow rate remaining after adding acid to tank
        n_a = p.pumpA.Q * C_a # mole rate of acid (mol HCl/s)
        n_tai = ta_i * 1000 * p.pumpI.Q # mole rate of total alkalinity (mol TA/s)
        if n_a >= n_tai:
            Q_a1 = n_tai/C_a # flow rate needed to reach equivalence point (m3/s)
            Q_a2 = p.pumpA.Q - Q_a1 # remaining flow rate (m3/s)
            H_af = (h_eq2*1000*(p.pumpI.Q + Q_a1) + C_a*Q_a2)/(p.pumpA.Q + p.pumpI.Q) # (mol/m3) concentration after acid addition
        
        elif n_a < n_tai:
            n_TAaf = n_tai - n_a # (mol/s) remaining mole rate of total alkalinity
            TA_af = n_TAaf/(p.pumpI.Q + p.pumpA.Q) # (mol/m3) remaining concentration of total alkalinity

            H_af = (findH_TA(dic_i,TA_af/1000,pH_eq2,pH_i, 0.01)) * 1000 # (mol/m3) Function result is mol/L need mol/m3
            
        # Find CO2 Extracted
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q # post extraction
        CO2_af = (findCO2(seawater_config, dic_i,H_af/1000)) * 1000 # (mol/m3) Concentration of aqueous CO2 in the acidified seawater
        n_co2_h2o = CO2_af * p.pumpASW.Q # mole rate of CO2 in the water
        n_co2_ext = n_co2_h2o * y_ext # mole rate of CO2 extracted
        S2['mCC'][i] = n_co2_ext * co2_mm * 3600/10**6 # tCO2/step

        # Find pH After CO2 Extraction & Before Base Addition
        CO2_bi = (1-y_ext)*CO2_af # (mol/m3) CO2 conc before base add and after CO2 extraction
        DIC_f = dic_i * 1000 - (y_ext*CO2_af) # (mol/m3) dic conc before base add and after CO2 extraction
        S2['dic_f'][i] = DIC_f/1000 # convert final DIC to mol/L
        H_bi = (findH_CO2(seawater_config, DIC_f/1000, CO2_bi/1000, -np.log10(H_af/1000), pH_i, 0.01)) * 1000 # (mol/m3) acidity after CO2 extraction (note min of search is the acidified seawater pH)
        
        # Find TA Before Base Addition
        TA_bi = (findTA(seawater_config, DIC_f/1000,H_bi/1000)) * 1000 # (mol/m3)
        
        # Add Additional Base to Tank
        # Amount of base added for mCC
        Q_bMCC = Q_aMCC # flow rate used for minimal mCC

        # Base addition to tank
        Q_bT = p.pumpB.Q - Q_bMCC # (m3/s) flow rate of base to tank
        n_bT = C_b*Q_bT # (mol/s) rate of base moles added to tank
        S2['volBase'][i] = Q_bT * 3600 # (m3) base added to tank

        # Base Flow Rate Adjusted to Minimum for Chemistry Calcs
        p.pumpB.Q = Q_bMCC # flow rate remaining after adding base to tank

        # Find TA After Base Addition
        TA_bf = (TA_bi * p.pumpASW.Q + C_b * p.pumpB.Q)/(p.pumpASW.Q + p.pumpB.Q) # (mol/m3)

        # Find pH After Base Addition
        H_bf = (findH_TA(seawater_config, DIC_f/1000, TA_bf/1000, -np.log10(H_bi/1000), -np.log10(kw), 0.01)) * 1000 # (mol/m3) acidity after base addition
        S2['pH_f'][i] = -np.log10(H_bf/1000)

        # Seawater Outtake
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q # Outtake flow rate
        S2['Qout'][i] = p.pumpF.Q # (m3/s) Outtake 

# def initialize_scenarios():

if __name__ == "__main__":
    pumps = initialize_pumps(
        ed_config=ElectroDialysisInputs(), pump_config=PumpInputs()
    )

    print(pumps.pumpA.Q_max)
    print(SeaWaterInputs())


    res = initialize_power_chemical_ranges(
        ed_config = ElectroDialysisInputs(), 
        pump_config= PumpInputs(), 
        seawater_config= SeaWaterInputs()
        )