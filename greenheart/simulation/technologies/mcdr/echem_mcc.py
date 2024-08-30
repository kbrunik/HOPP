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
from numpy import ndarray


@define
class ElectrodialysisInputs:
    """
    A class to represent the inputs for an Electrodialysis (ED) marine carbon capture unit.

    Attributes:
        P_ed1 (float): Power needed for one ED unit in watts (W). Default is 50e6 (50 MW).
        Q_ed1 (float): Flow rate for one ED unit in cubic meters per second (m³/s). Default is 1.
        N_edMin (int): Minimum number of ED units. Default is 1.
        N_edMax (int): Maximum number of ED units. Default is 5.
        E_HCl (float): Energy required by the ED unit to process HCl in kilowatt-hours per mole (kWh/mol). Default is 0.05.
        E_NaOH (float): Energy required by the ED unit to process NaOH in kilowatt-hours per mole (kWh/mol). Default is 0.05.
        y_ext (float): CO2 extraction efficiency as a percentage. Default is 90% (0.9).
        y_vac (float): Vacuum efficiency as a percentage. Default is 30% (0.3).
        store_hours (float): Hours the tanks can provide minimum flow rate for capture. Default is 6.
        co2_mm (float): Molar mass of CO2 in grams per mole (g/mol). Default is 44.01.

    Attributes (calculated):
        P_minED (float): Minimum power required for the ED units, calculated as the product of P_ed1 and N_edMin.
    """

    P_ed1: float = 50 * 10**6
    Q_ed1: float = 1
    N_edMin: int = 1
    N_edMax: int = 5
    E_HCl: float = 0.05
    E_NaOH: float = 0.05
    y_ext: float = 0.9
    y_vac: float = 0.3
    store_hours: float = 6
    co2_mm: float = field(init=False, default=44.01)  # g/mol molar mass of CO2
    P_minED: float = field(init=False)

    def __attrs_post_init__(self):
        self.P_minED = self.P_ed1 * self.N_edMin  # Min ED Power (MW)


@define
class SeaWaterInputs:
    """
    A class to represent the initial inputs for seawater chemistry.

    Attributes:
        dic_i (float): Initial concentration of dissolved inorganic carbon (DIC) in mol/L.
            Default is 2.2e-3 (2.2 mM, typical for seawater; 3.12 mM in Instant Ocean).
        pH_i (float): Initial pH of seawater. Default is 8.1.
        pH_eq2 (float): Second equivalence point of seawater pH. Default is 4.3.
        k1 (float): First dissociation constant of carbonic acid in mol/L.
            Default is 1.38e-6 (10^-5.86).
        k2 (float): Second dissociation constant of carbonic acid in mol/L.
            Default is 1.2e-9 (10^-8.92).
        kw (float): Water dissociation constant at 25°C and a salinity of 3 in mol/L.
            Default is 6.02e-14 (10^-13.22).

    Attributes (calculated):
        h_i (float): Initial concentration of hydrogen ions (H+) in mol/L, calculated from the initial pH.
        h_eq2 (float): Concentration of hydrogen ions (H+) at the second equivalence point in mol/L, calculated from pH_eq2.
        ta_i (float): Initial total alkalinity concentration (TA) in mol/L, calculated using DIC and h_i.
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
        self.ta_i = findTA(self, self.dic_i, self.h_i)


def findTA(seawater: SeaWaterInputs, dic, h):
    """
    Calculate the total alkalinity (TA) of seawater.

    This function computes the total alkalinity based on the initial concentration of dissolved inorganic carbon (DIC),
    the hydrogen ion concentration (H+), and the seawater dissociation constants (k1, k2, kw).

    Args:
        seawater (SeaWaterInputs): An instance of the SeaWaterInputs class containing seawater chemistry constants.
        dic (float): The concentration of dissolved inorganic carbon (DIC) in mol/L.
        h (float): The concentration of hydrogen ions (H+) in mol/L.

    Returns:
        float: The total alkalinity (TA) in mol/L.
    """

    k1 = seawater.k1
    k2 = seawater.k2
    kw = seawater.kw
    ta = (
        (dic / (1 + (h / k1) + (k2 / h)))
        + (2 * dic / (1 + (h / k2) + ((h**2) / (k1 * k2))))
        + kw / h
        - h
    )
    return ta


def findH_TA(seawater: SeaWaterInputs, dic, ta, ph_min, ph_max, step):
    """
    Estimate the hydrogen ion concentration (H+) that corresponds to a given total alkalinity (TA).

    This function iteratively calculates the total alkalinity over a range of pH values to find the hydrogen ion
    concentration that minimizes the difference between the estimated and the provided total alkalinity.

    Args:
        seawater (SeaWaterInputs): An instance of the SeaWaterInputs class containing seawater chemistry constants.
        dic (float): The concentration of dissolved inorganic carbon (DIC) in mol/L.
        ta (float): The target total alkalinity (TA) in mol/L.
        ph_min (float): The minimum pH value in the search range.
        ph_max (float): The maximum pH value in the search range.
        step (float): The step size for the pH values in the search range.

    Returns:
        float: The hydrogen ion concentration (H+) in mol/L that corresponds to the target total alkalinity.
    """
    ph_range = np.arange(ph_min, ph_max + step, step)
    ta_error = np.zeros(len(ph_range))
    for i in range(len(ph_range)):
        h_est = 10 ** -ph_range[i]
        ta_est = findTA(seawater, dic, h_est)
        ta_error[i] = abs(ta - ta_est)
    for i in range(len(ph_range)):
        if ta_error[i] == min(ta_error):
            i_ph = i
    ph_f = ph_range[i_ph]
    h_f = 10**-ph_f
    return h_f


def findCO2(seawater: SeaWaterInputs, dic, h):
    """
    Calculate the concentration of dissolved carbon dioxide (CO2) from the dissolved inorganic carbon (DIC)
    and hydrogen ion concentration (H+).

    This function computes the concentration of CO2 based on the provided DIC, the hydrogen ion concentration,
    and the dissociation constants of carbonic acid.

    Args:
        seawater (SeaWaterInputs): An instance of the SeaWaterInputs class containing seawater chemistry constants.
        dic (float): The concentration of dissolved inorganic carbon (DIC) in mol/L.
        h (float): The concentration of hydrogen ions (H+) in mol/L.

    Returns:
        float: The concentration of dissolved carbon dioxide (CO2) in mol/L.
    """
    k1 = seawater.k1
    k2 = seawater.k2
    co2 = dic / (1 + (k1 / h) + ((k1 * k2) / (h**2)))
    return co2


def findH_CO2(seawater: SeaWaterInputs, dic, co2, ph_min, ph_max, step):
    """
    Estimate the hydrogen ion concentration (H+) that corresponds to a given concentration of dissolved carbon dioxide (CO2).

    This function iteratively calculates the CO2 concentration over a range of pH values to find the hydrogen ion concentration
    that minimizes the difference between the estimated and the provided CO2 concentration.

    Args:
        seawater (SeaWaterInputs): An instance of the SeaWaterInputs class containing seawater chemistry constants.
        dic (float): The concentration of dissolved inorganic carbon (DIC) in mol/L.
        co2 (float): The target concentration of dissolved carbon dioxide (CO2) in mol/L.
        ph_min (float): The minimum pH value in the search range.
        ph_max (float): The maximum pH value in the search range.
        step (float): The step size for the pH values in the search range.

    Returns:
        float: The hydrogen ion concentration (H+) in mol/L that corresponds to the target CO2 concentration.
    """
    ph_range = np.arange(ph_min, ph_max + step, step)
    co2_error = np.zeros(len(ph_range))
    for i in range(len(ph_range)):
        h_est = 10 ** -ph_range[i]
        co2_est = findCO2(seawater, dic, h_est)
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

    Q_min: float
    Q_max: float
    p_min_bar: float
    p_max_bar: float
    eff: float
    Q: float = field(default=0)

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
    ed_config: ElectrodialysisInputs, pump_config: PumpInputs
) -> PumpOutputs:
    """Initialize a list of Pump instances based on the provided configurations.

    Args:
        ed_config (ElectrodialysisInputs): The electro-dialysis inputs.
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

    mCC_min: float
    mCC_max: float
    p_min_bar: float
    p_max_bar: float
    eff: float
    mCC: float = field(default=0)
    co2_mm: float = field(init=False, default=44.01)

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
class ElectrodialysisRangeOutputs:
    """
    A class to represent the electrodialysis device power and chemical ranges under each scenario.

    Attributes:
        S1 (dict): Chemical and power ranges for scenario 1 (e.g., tank filled).
            - "volAcid": Volume of acid (L).
            - "volBase": Volume of base (L).
            - "mCC": Molarity of carbon capture solution (mol/L).
            - "pH_f": Final pH of the solution.
            - "dic_f": Final dissolved inorganic carbon concentration (mol/L).
            - "c_a": Concentration of acid (mol/L).
            - "c_b": Concentration of base (mol/L).
            - "Qin": Flow rate into the system (m³/s).
            - "Qout": Flow rate out of the system (m³/s).
            - "pwrRanges": Power range for the scenario (MW).

        S2 (dict): Chemical and power ranges for scenario 2 (e.g., variable power and chemical ranges).
            - Same keys as in S1.

        S3 (dict): Chemical and power ranges for scenario 3 (e.g., ED not active, tanks not zeros).
            - Same keys as in S1.

        S4 (dict): Chemical and power ranges for scenario 4 (e.g., ED active, no capture).
            - Same keys as in S1.

        S5 (dict): Chemical and power ranges for scenario 5 (reserved for special cases).
            - Same keys as in S1.

        P_minS1_tot (float): Minimum power for scenario 1.
        P_minS2_tot (float): Minimum power for scenario 2.
        P_minS3_tot (float): Minimum power for scenario 3.
        P_minS4_tot (float): Minimum power for scenario 4.
        V_aT_max (float): Maximum volume of acid in the tank.
        V_bT_max (float): Maximum volume of base in the tank.
        V_a3_min (float): Minimum volume of acid required for S3.
        V_b3_min (float): Minimum volume of base required for S3.
        N_range (int): Number of ED units active in S1, S3, S4.
        S2_tot_range (int): Number of ED units active in S2.
        pump_power_min (float): Minimum pump power in MW.
        pump_power_max (float): Maximum pump power in MW.
        vacuum_power_min (float): Minimum vacuum power in MW.
        vacuum_power_max (float): Maximum vacuum power in MW.
    """

    S1: dict
    S2: dict
    S3: dict
    S4: dict
    S5: dict
    P_minS1_tot: float
    P_minS2_tot: float
    P_minS3_tot: float
    P_minS4_tot: float
    V_aT_max: float
    V_bT_max: float
    V_a3_min: float
    V_b3_min: float
    N_range: int
    S2_tot_range: int
    S2_ranges: ndarray
    pump_power_min: float
    pump_power_max: float
    vacuum_power_min: float
    vacuum_power_max: float


def initialize_power_chemical_ranges(
    ed_config: ElectrodialysisInputs,
    pump_config: PumpInputs,
    seawater_config: SeaWaterInputs,
) -> ElectrodialysisRangeOutputs:
    """
    Initialize the power and chemical ranges for an electrodialysis system under various scenarios.

    This function calculates the power and chemical usage ranges for an electrodialysis system across five distinct scenarios:

    1. Scenario 1: Tank filled

    2. Scenario 2: Capture CO2 & Fill Tank

    3. Scenario 3: ED not active, tanks not zeros*

    4. Scenario 4: ED active, no capture

    5. Scenario 5: All input power is excess

    Args:
        ed_config (ElectrodialysisInputs): Configuration parameters for the electrodialysis system.
        pump_config (PumpInputs): Configuration parameters for the pumping system.
        seawater_config (SeaWaterInputs): Configuration parameters for the seawater used in the process.

    Returns:
        ElectrodialysisRangeOutputs: An object containing the power and chemical ranges for each scenario.
    """

    N_edMin = ed_config.N_edMin
    N_edMax = ed_config.N_edMax
    P_ed1 = ed_config.P_ed1
    Q_ed1 = ed_config.Q_ed1
    E_HCl = ed_config.E_HCl
    E_NaOH = ed_config.E_NaOH
    y_ext = ed_config.y_ext
    co2_mm = ed_config.co2_mm
    dic_i = seawater_config.dic_i
    h_i = seawater_config.h_i
    ta_i = seawater_config.ta_i
    kw = seawater_config.kw
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
        "pwrRanges",
    ]

    # Initialize the dictionaries
    S1 = {key: np.zeros(N_range) for key in keys}
    S2 = {key: np.zeros(S2_tot_range) for key in keys}
    S3 = {key: np.zeros(N_range) for key in keys}
    S4 = {key: np.zeros(N_range) for key in keys}
    S5 = {key: np.zeros(1) for key in keys}

    p = initialize_pumps(ed_config=ed_config, pump_config=pump_config)

    ########################## Chemical & Power Ranges: S1, S3, S4 ###################################
    for i in range(N_range):
        ############################### S1: Chem Ranges: Tank Filled #####################################
        P_EDi = (i + N_edMin) * P_ed1  # ED unit power requirements
        p.pumpED.Q = (i + N_edMin) * Q_ed1  # Flow rates for ED Units
        p.pumpO.Q = 100 * p.pumpED.Q  # Intake is 100x larger than ED unit
        S1["Qin"][i] = p.pumpO.Q  # (m3/s) Intake

        # Acid and Base Concentrations
        p.pumpA.Q = p.pumpED.Q / 2  # Acid flow rate
        C_a = (1 / p.pumpA.Q) * (
            P_EDi / (3600 * (E_HCl * 1000)) - (p.pumpED.Q * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        S1["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units

        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q  # Base flow rate
        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        S1["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units

        # Acid Addition
        p.pumpI.Q = p.pumpO.Q - p.pumpED.Q  # Intake remaining after diversion to ED
        n_a = p.pumpA.Q * C_a  # mole rate of acid (mol HCl/s)
        n_tai = ta_i * 1000 * p.pumpI.Q  # mole rate of total alkalinity (mol TA/s)

        if n_a >= n_tai:
            Q_a1 = n_tai / C_a  # flow rate needed to reach equivalence point (m3/s)
            Q_a2 = p.pumpA.Q - Q_a1  # remaining flow rate (m3/s)
            H_af = (h_eq2 * 1000 * (p.pumpI.Q + Q_a1) + C_a * Q_a2) / (
                p.pumpA.Q + p.pumpI.Q
            )  # (mol/m3) concentration after acid addition
        elif n_a < n_tai:
            n_TAaf = n_tai - n_a  # (mol/s) remaining mole rate of total alkalinity
            TA_af = n_TAaf / (
                p.pumpI.Q + p.pumpA.Q
            )  # (mol/m3) remaining concentration of total alkalinity

            H_af = (
                findH_TA(dic_i, TA_af / 1000, pH_eq2, pH_i, 0.01)
            ) * 1000  # (mol/m3) Function result is mol/L need mol/m3

        # Find CO2 Extracted
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q  # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q  # post extraction
        CO2_af = (
            findCO2(seawater_config, dic_i, H_af / 1000)
        ) * 1000  # (mol/m3) Concentration of aqueous CO2 in the acidified seawater
        n_co2_h2o = CO2_af * p.pumpASW.Q  # mole rate of CO2 in the water
        n_co2_ext = n_co2_h2o * y_ext  # mole rate of CO2 extracted (mol/s)
        S1["mCC"][i] = n_co2_ext * co2_mm * 3600 / 10**6  # tCO2/hr

        # Find pH After CO2 Extraction & Before Base Addition
        CO2_bi = (
            1 - y_ext
        ) * CO2_af  # (mol/m3) CO2 conc before base add and after CO2 extraction
        DIC_f = dic_i * 1000 - (
            y_ext * CO2_af
        )  # (mol/m3) dic conc before base add and after CO2 extraction
        S1["dic_f"][i] = DIC_f / 1000  # convert final DIC to mol/L

        H_bi = (
            findH_CO2(
                seawater_config,
                DIC_f / 1000,
                CO2_bi / 1000,
                -np.log10(H_af / 1000),
                pH_i,
                0.01,
            )
        ) * 1000  # (mol/m3) acidity after CO2 extraction (note min of search is the acidified seawater pH)

        # Find TA Before Base Addition
        TA_bi = (findTA(seawater_config, DIC_f / 1000, H_bi / 1000)) * 1000  # (mol/m3)

        # Find TA After Base Addition
        TA_bf = (TA_bi * p.pumpASW.Q + C_b * p.pumpB.Q) / (
            p.pumpASW.Q + p.pumpB.Q
        )  # (mol/m3)

        # Find pH After Base Addition
        H_bf = (
            findH_TA(
                seawater_config,
                DIC_f / 1000,
                TA_bf / 1000,
                -np.log10(H_bi / 1000),
                -np.log10(kw),
                0.01,
            )
        ) * 1000  # (mol/m3) acidity after base addition
        S1["pH_f"][i] = -np.log10(H_bf / 1000)

        # Outtake
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q  # (m3/s) Outtake flow rate
        S1["Qout"][i] = p.pumpF.Q  # (m3/s) Outtake

        ############################### S3: Chem Ranges: ED not active, tanks not zeros ##################
        P_EDi = 0  # ED Unit is off
        p.pumpED.Q = 0  # ED Unit is off
        p.pumpO.Q = (
            100 * (i + N_edMin) * Q_ed1
        )  # Flow rates for intake based on equivalent ED units that would be active
        S3["Qin"][i] = p.pumpO.Q
        p.pumpI.Q = p.pumpO.Q  # since no flow is going to the ED unit
        p.pumpA.Q = (
            (i + N_edMin) * Q_ed1 / 2
        )  # Flow rate for acid pump based on equivalent ED units that would be active
        p.pumpB.Q = p.pumpA.Q

        # Change in volume due to acid and base use
        S3["volAcid"][i] = -p.pumpA.Q * 3600  # (m3) volume of acid lost by the tank
        S3["volBase"][i] = -p.pumpB.Q * 3600  # (m3) volume of base lost by the tank

        # The concentration of acid and base produced does not vary with flow rate since Q_a = Q_b = Q_ed/2
        # Also does not vary with power since the power for the ED units scale directly with the flow rate
        C_a = (1 / p.pumpA.Q_min) * (
            P_ed1 * N_edMin / (3600 * (E_HCl * 1000)) - (p.pumpED.Q_min * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        S3["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration used in S3
        C_b = (1 / p.pumpB.Q_min) * (
            P_ed1 * N_edMin / (3600 * (E_NaOH * 1000))
            - (p.pumpED.Q_min * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        S3["c_b"][i] = C_b / 1000  # (mol/L) Base concentration used in S3

        # Acid addition
        n_a = p.pumpA.Q * C_a  # mole rate of acid (mol HCl/s)
        n_tai = ta_i * 1000 * p.pumpI.Q  # mole rate of total alkalinity (mol TA/s)
        if n_a >= n_tai:
            Q_a1 = n_tai / C_a  # flow rate needed to reach equivalence point (m3/s)
            Q_a2 = p.pumpA.Q - Q_a1  # remaining flow rate (m3/s)
            H_af = (h_eq2 * 1000 * (p.pumpI.Q + Q_a1) + C_a * Q_a2) / (
                p.pumpA.Q + p.pumpI.Q
            )  # (mol/m3) concentration after acid addition
        elif n_a < n_tai:
            n_TAaf = n_tai - n_a  # (mol/s) remaining mole rate of total alkalinity
            TA_af = n_TAaf / (
                p.pumpI.Q + p.pumpA.Q
            )  # (mol/m3) remaining concentration of total alkalinity
            H_af = (
                findH_TA(dic_i, TA_af / 1000, pH_eq2, pH_i, 0.01)
            ) * 1000  # (mol/m3) Function result is mol/L need mol/m3

        # Find CO2 Extracted
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q  # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q  # post extraction
        CO2_af = (
            findCO2(seawater_config, dic_i, H_af / 1000)
        ) * 1000  # (mol/m3) Concentration of aqueous CO2 in the acidified seawater
        n_co2_h2o = CO2_af * p.pumpASW.Q  # mole rate of CO2 in the water
        n_co2_ext = n_co2_h2o * y_ext  # mole rate of CO2 extracted
        S3["mCC"][i] = n_co2_ext * co2_mm * 3600 / 10**6  # tCO2/step

        # Find pH After CO2 Extraction & Before Base Addition
        CO2_bi = (
            1 - y_ext
        ) * CO2_af  # (mol/m3) CO2 conc before base add and after CO2 extraction
        DIC_f = dic_i * 1000 - (
            y_ext * CO2_af
        )  # (mol/m3) dic conc before base add and after CO2 extraction
        S3["dic_f"][i] = DIC_f / 1000  # (mol/L)
        H_bi = (
            findH_CO2(
                seawater_config,
                DIC_f / 1000,
                CO2_bi / 1000,
                -np.log10(H_af / 1000),
                pH_i,
                0.01,
            )
        ) * 1000  # (mol/m3) acidity after CO2 extraction (note min of search is the acidified seawater pH)
        # Find TA Before Base Addition
        TA_bi = (findTA(seawater_config, DIC_f / 1000, H_bi / 1000)) * 1000  # (mol/m3)
        # Find TA After Base Addition
        TA_bf = (TA_bi * p.pumpASW.Q + C_b * p.pumpB.Q) / (
            p.pumpASW.Q + p.pumpB.Q
        )  # (mol/m3)
        # Find pH After Base Addition
        H_bf = (
            findH_TA(
                seawater_config,
                DIC_f / 1000,
                TA_bf / 1000,
                -np.log10(H_bi / 1000),
                -np.log10(kw),
                0.01,
            )
        ) * 1000  # (mol/m3) acidity after base addition
        S3["pH_f"][i] = -np.log10(H_bf / 1000)
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q  # Outtake flow rate
        S3["Qout"][i] = p.pumpF.Q

        ############################### S4: Chem Ranges: ED active, no capture ###########################
        P_EDi = (i + N_edMin) * P_ed1  # ED unit power requirements
        p.pumpED.Q = 0  # Regular ED pump is inactive here
        p.pumpED4.Q = (i + N_edMin) * Q_ed1  # ED pump with filtration pressure

        # Acid and base concentrations
        p.pumpA.Q = p.pumpED4.Q / 2  # Acid flow rate
        p.pumpB.Q = p.pumpED4.Q - p.pumpA.Q  # Base flow rate
        C_a = (1 / p.pumpA.Q) * (
            P_EDi / (3600 * (E_HCl * 1000)) - (p.pumpED4.Q * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        S4["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED4.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        S4["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units

        # Acid added to the tank
        n_aT = C_a * p.pumpA.Q  # (mol/s) rate of acid moles added to tank
        S4["volAcid"][i] = p.pumpA.Q * 3600  # volume of acid in tank after time step

        # Base added to the tank
        n_bT = C_b * p.pumpB.Q  # (mol/s) rate of base moles added to tank
        S4["volBase"][i] = p.pumpB.Q * 3600  # volume of base in tank after time step

        # Intake (ED4 pump not O pump is used)
        p.pumpO.Q = 0  # Need intake for ED & min CC
        S4["Qin"][i] = p.pumpED4.Q  # (m3/s) Intake

        # Other pumps not used
        p.pumpI.Q = 0  # Intake remaining after diversion to ED
        p.pumpCO2ex.Q = 0  # Acid addition
        p.pumpASW.Q = 0  # post extraction

        # Outtake
        p.pumpF.Q = 0  # Outtake flow rate
        S4["Qout"][i] = p.pumpF.Q  # (m3/s) Outtake

        # Since no capture is conducted the final DIC and pH is the same as the initial
        S4["pH_f"][i] = pH_i
        S4["dic_f"][i] = dic_i  # (mol/L)

        # Define vacuum pump based on mCC ranges from S1 and S3 (S3 has a slightly higher mCC)
        vacCO2 = Vacuum(
            min(np.concatenate([S1["mCC"], S3["mCC"]])),
            max(np.concatenate([S1["mCC"], S3["mCC"]])),
            pump_config.p_co2_min_bar,
            pump_config.p_co2_max_bar,
            ed_config.y_vac,
        )

        ############################### S1: Power Ranges: Tank filled ####################################
        P_EDi = (i + N_edMin) * P_ed1  # ED unit power requirements
        p.pumpED.Q = (i + N_edMin) * Q_ed1  # Flow rates for ED Units
        p.pumpO.Q = 100 * p.pumpED.Q  # Intake is 100x larger than ED unit
        p.pumpA.Q = p.pumpED.Q / 2  # Acid flow rate
        p.pumpI.Q = p.pumpO.Q - p.pumpED.Q  # Intake remaining after diversion to ED
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q  # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q  # post extraction
        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q  # Base flow rate
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q  # Outtake flow rate
        vacCO2.mCC = S1["mCC"][i]  # mCC rate from the previous calculation
        S1["pwrRanges"][i] = (
            P_EDi
            + p.pumpED.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpCO2ex.power()
            + p.pumpASW.power()
            + p.pumpB.power()
            + p.pumpF.power()
            + vacCO2.power()
        )
        # Minimum Power Input for Scenario 1 (ED active Tanks inactive)
        P_minS1_tot = (
            ed_config.P_minED
            + p.pumpO.P_min
            + p.pumpED.P_min
            + p.pumpA.P_min
            + p.pumpI.P_min
            + p.pumpCO2ex.P_min
            + p.pumpASW.P_min
            + p.pumpB.P_min
            + p.pumpF.P_min
        )

        ######################## S3: Power Ranges: ED not active, tanks not zeros ########################
        P_EDi = 0  # ED Unit is off
        p.pumpED.Q = 0  # ED Unit is off
        p.pumpO.Q = (
            100 * (i + N_edMin) * Q_ed1
        )  # Flow rates for intake based on equivalent ED units that would be active
        p.pumpI.Q = p.pumpO.Q  # since no flow is going to the ED unit
        p.pumpA.Q = (
            (i + N_edMin) * Q_ed1 / 2
        )  # Flow rate for acid pump based on equivalent ED units that would be active
        p.pumpB.Q = p.pumpA.Q
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q  # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q  # post extraction
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q  # Outtake flow rate
        vacCO2.mCC = S3["mCC"][i]  # mCC rate from the previous calculation
        S3["pwrRanges"][i] = (
            P_EDi
            + p.pumpED.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpCO2ex.power()
            + p.pumpASW.power()
            + p.pumpB.power()
            + p.pumpF.power()
            + vacCO2.power()
        )
        P_minS3_tot = min(S3["pwrRanges"])

        ######################## S4: Power Ranges: ED active, no capture #################################
        P_EDi = (i + N_edMin) * P_ed1  # ED unit power requirements
        p.pumpED.Q = 0  # Regular ED pump is inactive here
        p.pumpED4.Q = (i + N_edMin) * Q_ed1  # ED pump with filtration pressure
        p.pumpO.Q = 0  # Need intake for ED & min CC
        p.pumpA.Q = p.pumpED4.Q / 2  # Acid flow rate
        p.pumpI.Q = 0  # Intake remaining after diversion to ED
        p.pumpCO2ex.Q = 0  # Acid addition
        p.pumpASW.Q = 0  # post extraction
        p.pumpB.Q = p.pumpED4.Q - p.pumpA.Q  # Base flow rate
        p.pumpF.Q = 0  # Outtake flow rate
        vacCO2.mCC = S4["mCC"][i]  # mCC rate from the previous calculation
        S4["pwrRanges"][i] = (
            P_EDi
            + p.pumpED4.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpCO2ex.power()
            + p.pumpASW.power()
            + p.pumpB.power()
            + p.pumpF.power()
            + vacCO2.power()
        )
        P_minS4_tot = min(S4["pwrRanges"])

    ################################ Chemical & Power Ranges: S2 #####################################
    for i in range(S2_tot_range):
        ##################### S2: Chem Ranges Avoiding Overflow: Capture CO2 & Fill Tank #################
        # ED Unit Characteristics
        N_edi = S2_ranges[i, 0] + S2_ranges[i, 1]
        P_EDi = (N_edi) * P_ed1  # ED unit power requirements
        p.pumpED.Q = (N_edi) * Q_ed1  # Flow rates for ED Units

        # Acid and Base Creation
        p.pumpA.Q = p.pumpED.Q / 2  # Acid flow rate
        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q  # Base flow rate
        C_a = (1 / p.pumpA.Q) * (
            P_EDi / (3600 * (E_HCl * 1000)) - (p.pumpED.Q * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        S2["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        S2["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units

        # Amount of acid added for mCC
        Q_aMCC = S2_ranges[i, 0] * Q_ed1 / 2  # flow rate used for mCC

        # Acid addition to tank (base volume will be the same)
        Q_aT = p.pumpA.Q - Q_aMCC  # (m3/s) flow rate of acid to tank
        n_aT = C_a * Q_aT  # (mol/s) rate of acid moles added to tank
        S2["volAcid"][i] = Q_aT * 3600  # (m3) acid added to tank

        # Seawater Intake
        p.pumpO.Q = Q_aMCC * 2 * 100 + (
            p.pumpED.Q - (Q_aMCC * 2)
        )  # total seawater intake
        S2["Qin"][i] = p.pumpO.Q  # (m3/s) intake

        # Acid addition to seawater
        p.pumpI.Q = p.pumpO.Q - p.pumpED.Q  # seawater that will recieve acid

        # Acid Flow Rate for mCC Chemistry Calcs
        p.pumpA.Q = Q_aMCC  # flow rate remaining after adding acid to tank
        n_a = p.pumpA.Q * C_a  # mole rate of acid (mol HCl/s)
        n_tai = ta_i * 1000 * p.pumpI.Q  # mole rate of total alkalinity (mol TA/s)
        if n_a >= n_tai:
            Q_a1 = n_tai / C_a  # flow rate needed to reach equivalence point (m3/s)
            Q_a2 = p.pumpA.Q - Q_a1  # remaining flow rate (m3/s)
            H_af = (h_eq2 * 1000 * (p.pumpI.Q + Q_a1) + C_a * Q_a2) / (
                p.pumpA.Q + p.pumpI.Q
            )  # (mol/m3) concentration after acid addition

        elif n_a < n_tai:
            n_TAaf = n_tai - n_a  # (mol/s) remaining mole rate of total alkalinity
            TA_af = n_TAaf / (
                p.pumpI.Q + p.pumpA.Q
            )  # (mol/m3) remaining concentration of total alkalinity

            H_af = (
                findH_TA(dic_i, TA_af / 1000, pH_eq2, pH_i, 0.01)
            ) * 1000  # (mol/m3) Function result is mol/L need mol/m3

        # Find CO2 Extracted
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q  # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q  # post extraction
        CO2_af = (
            findCO2(seawater_config, dic_i, H_af / 1000)
        ) * 1000  # (mol/m3) Concentration of aqueous CO2 in the acidified seawater
        n_co2_h2o = CO2_af * p.pumpASW.Q  # mole rate of CO2 in the water
        n_co2_ext = n_co2_h2o * y_ext  # mole rate of CO2 extracted
        S2["mCC"][i] = n_co2_ext * co2_mm * 3600 / 10**6  # tCO2/step

        # Find pH After CO2 Extraction & Before Base Addition
        CO2_bi = (
            1 - y_ext
        ) * CO2_af  # (mol/m3) CO2 conc before base add and after CO2 extraction
        DIC_f = dic_i * 1000 - (
            y_ext * CO2_af
        )  # (mol/m3) dic conc before base add and after CO2 extraction
        S2["dic_f"][i] = DIC_f / 1000  # convert final DIC to mol/L
        H_bi = (
            findH_CO2(
                seawater_config,
                DIC_f / 1000,
                CO2_bi / 1000,
                -np.log10(H_af / 1000),
                pH_i,
                0.01,
            )
        ) * 1000  # (mol/m3) acidity after CO2 extraction (note min of search is the acidified seawater pH)

        # Find TA Before Base Addition
        TA_bi = (findTA(seawater_config, DIC_f / 1000, H_bi / 1000)) * 1000  # (mol/m3)

        # Add Additional Base to Tank
        # Amount of base added for mCC
        Q_bMCC = Q_aMCC  # flow rate used for minimal mCC

        # Base addition to tank
        Q_bT = p.pumpB.Q - Q_bMCC  # (m3/s) flow rate of base to tank
        n_bT = C_b * Q_bT  # (mol/s) rate of base moles added to tank
        S2["volBase"][i] = Q_bT * 3600  # (m3) base added to tank

        # Base Flow Rate Adjusted to Minimum for Chemistry Calcs
        p.pumpB.Q = Q_bMCC  # flow rate remaining after adding base to tank

        # Find TA After Base Addition
        TA_bf = (TA_bi * p.pumpASW.Q + C_b * p.pumpB.Q) / (
            p.pumpASW.Q + p.pumpB.Q
        )  # (mol/m3)

        # Find pH After Base Addition
        H_bf = (
            findH_TA(
                seawater_config,
                DIC_f / 1000,
                TA_bf / 1000,
                -np.log10(H_bi / 1000),
                -np.log10(kw),
                0.01,
            )
        ) * 1000  # (mol/m3) acidity after base addition
        S2["pH_f"][i] = -np.log10(H_bf / 1000)

        # Seawater Outtake
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q  # Outtake flow rate
        S2["Qout"][i] = p.pumpF.Q  # (m3/s) Outtake

        ##################### S2: Power Ranges: Capture CO2 & Fill Tank ##################################
        N_edi = S2_ranges[i, 0] + S2_ranges[i, 1]
        P_EDi = (N_edi) * P_ed1  # ED unit power requirements
        p.pumpED.Q = (N_edi) * Q_ed1  # Flow rates for ED Units
        # Amount of acid added for mCC
        Q_aMCC = S2_ranges[i, 0] * Q_ed1 / 2  # flow rate used for mCC
        p.pumpO.Q = Q_aMCC * 2 * 100 + (
            p.pumpED.Q - (Q_aMCC * 2)
        )  # total seawater intake
        p.pumpA.Q = p.pumpED.Q / 2  # Acid flow rate
        p.pumpI.Q = p.pumpO.Q - p.pumpED.Q  # Intake remaining after diversion to ED
        p.pumpCO2ex.Q = p.pumpI.Q + p.pumpA.Q  # Acid addition
        p.pumpASW.Q = p.pumpCO2ex.Q  # post extraction
        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q  # Base flow rate
        p.pumpF.Q = p.pumpASW.Q + p.pumpB.Q  # Outtake flow rate
        vacCO2.mCC = S2["mCC"][i]  # mCC rate from the previous calculation
        S2["pwrRanges"][i] = (
            P_EDi
            + p.pumpED.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpCO2ex.power()
            + p.pumpASW.power()
            + p.pumpB.power()
            + p.pumpF.power()
            + vacCO2.power()
        )
        P_minS2_tot = min(S2["pwrRanges"])

    ##################### S5: Chem Ranges: When all input power is excess ############################
    S5["volAcid"] = 0  # No acid generated
    S5["volBase"] = 0  # No base generated
    S5["mCC"] = 0  # No CO2 capture
    S5["pH_f"] = pH_i  # No changes in sea pH
    S5["dic_f"] = dic_i  # (mol/L) No changes in sea DIC
    S5["c_a"] = (
        h_i  # (mol/L) No acid generated so acid concentration is the same as that of seawater
    )
    S5["c_b"] = (
        kw / h_i
    )  # (mol/L) No base generated so base concentration is the same as that of seawater
    S5["Qin"] = 0  # (m3/s) No intake
    S5["Qout"] = 0  # (m3/s) No outtake

    # Define Tank Max Volumes (note there are two but they have the same volume)
    V_aT_max = (
        p.pumpED.Q_min / 2 * ed_config.store_hours * 3600
    )  # enables enough storage for 1 day or the hours from storeTime
    V_bT_max = V_aT_max  # tanks have the same volume

    # Volume needed for S3
    V_a3_min = p.pumpED.Q_min / 2 * 3600  # enables minimum mCC for 1 timestep
    V_b3_min = V_a3_min  # same volume needed for base

    # Pump Power Ranges
    pumpPwr_min = (
        p.pumpO.P_min
        + p.pumpI.P_min
        + p.pumpED.P_min
        + p.pumpA.P_min
        + p.pumpB.P_min
        + p.pumpCO2ex.P_min
        + p.pumpASW.P_min
        + p.pumpF.P_min
    ) / 1e6
    pumpPwr_max = (
        p.pumpO.P_max
        + p.pumpI.P_max
        + p.pumpED.P_max
        + p.pumpA.P_max
        + p.pumpB.P_max
        + p.pumpCO2ex.P_max
        + p.pumpASW.P_max
        + p.pumpF.P_max
    ) / 1e6

    return ElectrodialysisRangeOutputs(
        S1=S1,
        S2=S2,
        S3=S3,
        S4=S4,
        S5=S5,
        P_minS1_tot=P_minS1_tot,
        P_minS2_tot=P_minS2_tot,
        P_minS3_tot=P_minS3_tot,
        P_minS4_tot=P_minS4_tot,
        V_aT_max=V_aT_max,
        V_bT_max=V_bT_max,
        V_a3_min=V_a3_min,
        V_b3_min=V_b3_min,
        N_range=N_range,
        S2_tot_range=S2_tot_range,
        S2_ranges=S2_ranges,
        pump_power_min=pumpPwr_min,
        pump_power_max=pumpPwr_max,
        vacuum_power_min=vacCO2.P_min / 10e6,
        vacuum_power_max=vacCO2.P_max / 10e6,
    )


@define
class ElectrodialysisOutputs:
    """Outputs from the electrodialysis process.

    Attributes:
        ED_outputs (dict): Dictionary containing various output arrays from the electrodialysis process.
            Keys include:
                - N_ed (array): Number of electrodialysis units in operation at each time step.
                - P_xs (array): Excess power available at each time step (W).
                - volAcid (array): Volume of acid added or removed from tanks at each time step (m³).
                - volBase (array): Volume of base added or removed from tanks at each time step (m³).
                - mCC (array): Amount of CO2 captured at each time step (tCO2/hr).
                - pH_f (array): Final pH at each time step.
                - dic_f (array): Final dissolved inorganic carbon concentration at each time step.
                - c_a (array): Acid concentration at each time step (mol/L).
                - c_b (array): Base concentration at each time step (mol/L).
                - Qin (array): Intake flow rate at each time step (m³/s).
                - Qout (array): Outtake flow rate at each time step (m³/s).
        mCC_total (float): Total amount of CO2 captured over the entire process.
        capacity_factor (float): Overall capacity factor of the electrodialysis system.
        mCC_yr (float): Average yearly CO2 capture.
        mCC_yr_MaxPwr (float): Yearly CO2 capture under constant maximum power conditions.
    """

    ED_outputs: dict
    mCC_total: float
    capacity_factor: float
    mCC_yr: float
    mCC_yr_MaxPwr: float


def simulate_electrodialysis(
    ranges: ElectrodialysisRangeOutputs,
    ed_config: ElectrodialysisInputs,
    power_profile,
    initial_tank_volume_m3,
):
    """
    Simulates the operation of an electrodialysis (ED) system over time, given power availability and initial tank volumes.
    The simulation considers various scenarios based on the power profile and tank volumes, updating the state of the system
    at each time step.

    Parameters:
        ranges (ElectrodialysisRangeOutputs): The power and chemical ranges for different scenarios of ED operation.
        ed_config (ElectrodialysisInputs): Configuration inputs for the electrodialysis system.
        power_profile (np.ndarray): Array representing the available power at each time step (W).
        initial_tank_volume_m3 (float): The initial volume of acid and base in the tanks (m³).

    Returns:
        ElectrodialysisOutputs: A data class containing the simulation results, including the total CO2 captured,
        capacity factor, and yearly CO2 capture under actual and maximum power conditions.

    Notes:
        - The function evaluates five scenarios based on the available power and tank volumes, prioritizing CO2 capture and
          tank filling in the most effective way possible.
        - Scenario 5 is considered when all input power is excess, meaning no ED units are used.
    """
    N_edMin = ed_config.N_edMin

    tank_vol_a = np.zeros(len(power_profile) + 1)
    tank_vol_b = tank_vol_a
    tank_vol_a[0] = initial_tank_volume_m3
    tank_vol_b[0] = tank_vol_a[0]

    # Define the array names
    keys = [
        "N_ed",  # Number of ED units active
        "P_xs",  # (W) Excess power at each time
        "volAcid",  # (m³) Volume of acid added/removed to/from tanks at each time
        "volBase",  # (m³) Volume of base added/removed to/from tanks at each time
        "mCC",  # (tCO2/hr) Amount CO2 captured at each time
        "pH_f",  # Final pH at each time
        "dic_f",  # (mol/L) Final DIC at each time
        "c_a",  # (mol/L) Acid concentration at each time step
        "c_b",  # (mol/L) Base concentration at each time step
        "Qin",  # (m³/s) Intake flow rate at each time step
        "Qout",  # (m³/s) Outtake flow rate at each time step
    ]

    # Initialize the dictionaries
    ED_outputs = {key: np.zeros(len(power_profile)) for key in keys}

    nON = 0  # Timesteps when capture occurs (S1-3) used to determine capacity factor
    S_t = []  # Scenario for each time step

    for i in range(len(power_profile)):
        if power_profile[i] >= ranges.P_minS1_tot and tank_vol_a[i] == ranges.V_aT_max:
            # Note Scenario 1 is active
            S_t.append("S1")

            # Find number of active units based on power
            for j in range(ranges.N_range):
                if power_profile[i] >= ranges.S1["pwrRanges"][j]:
                    i_ed = j  # determine how many ED units can be used
            ED_outputs["N_ed"][i] = N_edMin + i_ed  # number of ED units active

            # Update recorded values based on number of ED units active
            ED_outputs["volAcid"][i] = ranges.S1["volAcid"][i_ed]
            ED_outputs["volBase"][i] = ranges.S1["volBase"][i_ed]
            ED_outputs["mCC"][i] = ranges.S1["mCC"][i_ed]
            ED_outputs["pH_f"][i] = ranges.S1["pH_f"][i_ed]
            ED_outputs["dic_f"][i] = ranges.S1["dic_f"][i_ed]
            ED_outputs["c_a"][i] = ranges.S1["c_a"][i_ed]
            ED_outputs["c_b"][i] = ranges.S1["c_b"][i_ed]
            ED_outputs["Qin"][i] = ranges.S1["Qin"][i_ed]
            ED_outputs["Qout"][i] = ranges.S1["Qout"][i_ed]

            # Update Tank Volumes
            tank_vol_a[i + 1] = tank_vol_a[i] + ED_outputs["volAcid"][i]
            tank_vol_b[i + 1] = tank_vol_b[i] + ED_outputs["volBase"][i]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max

            # Excess Power
            P_mCC = ranges.S1["pwrRanges"][
                i_ed
            ]  # power needed for mCC given the available power
            ED_outputs["P_xs"][i] = (
                power_profile[i] - P_mCC
            )  # Remaining power available for batteries

            # Number of times system is on
            nON = nON + 1  # Used to determine Capacity Factor

        # Scenario 2: Capture CO2 and Fill Tanks
        elif power_profile[i] >= ranges.P_minS2_tot and tank_vol_a[i] < ranges.V_aT_max:
            # Note Scenario 2 is active
            S_t.append("S2")

            # Find number of units that can be active based on power and volume
            # Determine number of scenarios that meet the qualifications
            v = 0
            for j in range(ranges.S2_tot_range):
                if (
                    power_profile[i] >= ranges.S2["pwrRanges"][j]
                    and ranges.V_aT_max >= tank_vol_a[i] + ranges.S2["volAcid"][j]
                ):
                    v = v + 1  # determine size of matrix for qualifying scenarios
            S2_viableRanges = np.zeros((v, 2))
            i_v = 0
            for j in range(ranges.S2_tot_range):
                if (
                    power_profile[i] >= ranges.S2["pwrRanges"][j]
                    and ranges.V_aT_max >= tank_vol_a[i] + ranges.S2["volAcid"][j]
                ):
                    S2_viableRanges[i_v, 0] = j  # index in the scenarios
                    S2_viableRanges[i_v, 1] = ranges.S2["volAcid"][
                        j
                    ]  # adding volume to the tanks is prioritized
                    i_v = i_v + 1
            # Select the viable scenario that fills the tank the most
            for j in range(len(S2_viableRanges[:, 1])):
                if S2_viableRanges[j, 1] == max(S2_viableRanges[:, 1]):
                    i_s2 = int(S2_viableRanges[j, 0])

            # Number of ED Units Active
            ED_outputs["N_ed"][i] = (
                ranges.S2_ranges[i_s2, 0] + ranges.S2_ranges[i_s2, 1]
            )  # number of ED units active

            # Update recorded values based on the case within S2
            ED_outputs["volAcid"][i] = ranges.S2["volAcid"][i_s2]
            ED_outputs["volBase"][i] = ranges.S2["volBase"][i_s2]
            ED_outputs["mCC"][i] = ranges.S2["mCC"][i_s2]
            ED_outputs["pH_f"][i] = ranges.S2["pH_f"][i_s2]
            ED_outputs["dic_f"][i] = ranges.S2["dic_f"][i_s2]
            ED_outputs["c_a"][i] = ranges.S2["c_a"][i_s2]
            ED_outputs["c_b"][i] = ranges.S2["c_b"][i_s2]
            ED_outputs["Qin"][i] = ranges.S2["Qin"][i_s2]
            ED_outputs["Qout"][i] = ranges.S2["Qout"][i_s2]

            # Update Tank Volumes
            tank_vol_a[i + 1] = tank_vol_a[i] + ED_outputs["volAcid"][i]
            tank_vol_b[i + 1] = tank_vol_b[i] + ED_outputs["volBase"][i]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max

            # Find excess power
            P_mCC = ranges.S2["pwrRanges"][
                i_s2
            ]  # power needed for mCC given the available power
            ED_outputs["P_xs"][i] = (
                power_profile[i] - P_mCC
            )  # Remaining power available for batteries

            # Number of times system is on
            nON = nON + 1  # Used to determine Capacity Factor

        # Scenario 3: Tanks Used for CO2 Capture
        elif (
            power_profile[i] >= ranges.P_minS3_tot and tank_vol_a[i] >= ranges.V_a3_min
        ):
            # Note Scenario 3 is active
            S_t.append("S3")

            # Find number of equivalent units active based on power
            for j in range(ranges.N_range):
                if (
                    power_profile[i] >= ranges.S3["pwrRanges"][j]
                    and tank_vol_a[i] >= ranges.S3["volAcid"][j]
                ):
                    i_ed = j  # determine how many ED units can be used
            ED_outputs["N_ed"][i] = N_edMin + i_ed  # number of ED units active

            # Update recorded values based on number of ED units active
            ED_outputs["volAcid"][i] = ranges.S3["volAcid"][i_ed]
            ED_outputs["volBase"][i] = ranges.S3["volBase"][i_ed]
            ED_outputs["mCC"][i] = ranges.S3["mCC"][i_ed]
            ED_outputs["pH_f"][i] = ranges.S3["pH_f"][i_ed]
            ED_outputs["dic_f"][i] = ranges.S3["dic_f"][i_ed]
            ED_outputs["c_a"][i] = ranges.S3["c_a"][i_ed]
            ED_outputs["c_b"][i] = ranges.S3["c_b"][i_ed]
            ED_outputs["Qin"][i] = ranges.S3["Qin"][i_ed]
            ED_outputs["Qout"][i] = ranges.S3["Qout"][i_ed]

            # Update Tank Volumes
            tank_vol_a[i + 1] = tank_vol_a[i] + ED_outputs["volAcid"][i]
            tank_vol_b[i + 1] = tank_vol_b[i] + ED_outputs["volBase"][i]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max

            # Find excess power
            P_mCC = ranges.S3["pwrRanges"][i_ed]  # Power needed for mCC
            ED_outputs["P_xs"][i] = (
                power_profile[i] - P_mCC
            )  # Excess Power to Batteries

            # Number of times system is on
            nON = nON + 1  # Used to determine Capacity Factor

        # Scenario 4: No Capture, Tanks Filled by ED Units
        elif power_profile[i] >= ranges.P_minS4_tot and tank_vol_a[i] < ranges.V_a3_min:
            # Note Scenario 4 is active
            S_t.append("S4")

            # Determine number of ED units active
            for j in range(ranges.N_range):
                if (
                    exPwr[i] >= ranges.S4["pwrRanges"][j]
                    and ranges.V_aT_max >= tank_vol_a[i] + ranges.S4["volAcid"][j]
                ):
                    i_ed = j  # determine how many ED units can be used
            ED_outputs["N_ed"][i] = N_edMin + i_ed  # number of ED units active

            # Update recorded values based on number of ED units active
            ED_outputs["volAcid"][i] = ranges.S4["volAcid"][i_ed]
            ED_outputs["volBase"][i] = ranges.S4["volBase"][i_ed]
            ED_outputs["mCC"][i] = ranges.S4["mCC"][i_ed]
            ED_outputs["pH_f"][i] = ranges.S4["pH_f"][i_ed]
            ED_outputs["dic_f"][i] = ranges.S4["dic_f"][i_ed]
            ED_outputs["c_a"][i] = ranges.S4["c_a"][i_ed]
            ED_outputs["c_b"][i] = ranges.S4["c_b"][i_ed]
            ED_outputs["Qin"][i] = ranges.S4["Qin"][i_ed]
            ED_outputs["Qout"][i] = ranges.S4["Qout"][i_ed]

            # Update Tank Volumes
            tank_vol_a[i + 1] = tank_vol_a[i] + ED_outputs["volAddAcid"][i]
            tank_vol_b[i + 1] = tank_vol_b[i] + ED_outputs["volAddBase"][i]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max

            # Find excess power
            P_mCC = ranges.S4["pwrRanges"][
                i_ed
            ]  # power needed for mCC system given the available power
            ED_outputs["P_xs"][i] = (
                power_profile[i] - P_mCC
            )  # Remaining power available for batteries

            # No change to nON since no capture is done

        # Scenario 5: When all Input Power is Excess
        else:
            # Note Scenario 5 is active
            S_t.append("S5")

            # Determine number of ED units active
            ED_outputs["N_ed"][i] = 0  # None are used in this case

            # Update recorded values based on number of ED units active
            ED_outputs["volAcid"][i] = ranges.S5["volAcid"]
            ED_outputs["volBase"][i] = ranges.S5["volBase"]
            ED_outputs["mCC"][i] = ranges.S5["mCC"]
            ED_outputs["pH_f"][i] = ranges.S5["pH_f"]
            ED_outputs["dic_f"][i] = ranges.S5["dic_f"]
            ED_outputs["c_a"][i] = ranges.S5["c_a"]
            ED_outputs["c_b"][i] = ranges.S5["c_b"]
            ED_outputs["Qin"][i] = ranges.S5["Qin"]
            ED_outputs["Qout"][i] = ranges.S5["Qout"]

            # Update Tank Volumes
            tank_vol_a[i + 1] = tank_vol_a[i] + ED_outputs["volAcid"][i]
            tank_vol_b[i + 1] = tank_vol_b[i] + ED_outputs["volBase"][i]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max

            # Find excess power
            ED_outputs["P_xs"][i] = power_profile[
                i
            ]  # Otherwise the input power goes directly to the batteries

            # No change to nON since no capture is done

    mCC_total = sum(ED_outputs["mCC"])  # total amount of CO2 captured
    capacity_factor = nON / len(ED_outputs["N_ed"])  # overall capcity factor

    # Average yearly CO2 capture
    mCC_yr = sum(ED_outputs["mCC"][0:8760])

    # Yearly CO2 capture under constant max power
    mCC_yr_MaxPwr = max(ranges.S1["mCC"]) * 8760

    return ElectrodialysisOutputs(
        ED_outputs=ED_outputs,
        mCC_total=mCC_total,
        capacity_factor=capacity_factor,
        mCC_yr=mCC_yr,
        mCC_yr_MaxPwr=mCC_yr_MaxPwr,
    )


@define
class ElectrodialysisCostInputs:
    """Inputs for the electrodialysis cost model.

    Attributes:
        electrodialysis_inputs (ElectrodialysisInputs): Inputs related to the electrodialysis process.
        mCC_yr (float): Average yearly CO2 capture.
        max_theoretical_mCC (float): Maximum theoretical CO2 capture TODO: add units.
        infrastructure_type (str): Infrastructure type, with options "desal", "swCool", or "new". Defaults to "new".
        user_costs (bool): If True, user-defined cost inputs are used, and the costs are initialized to zero.
                           If False, default costs are applied. Defaults to False.
        initial_capital_cost (float, optional): Initial capital cost provided by the user if `user_costs` is True.
                                                Initializes to zero if `user_costs` is True. Defaults to None.
        yearly_capital_cost (float, optional): Yearly capital cost provided by the user if `user_costs` is True.
                                               Initializes to zero if `user_costs` is True. Defaults to None.
        yearly_operational_cost (float, optional): Yearly operational cost provided by the user if `user_costs` is True.
                                                   Initializes to zero if `user_costs` is True. Defaults to None.
        initial_bop_capital_cost (float, optional): Initial Balance of Plant (BOP) capital cost provided by the user if `user_costs` is True.
                                                    Initializes to zero if `user_costs` is True. Defaults to None.
        yearly_bop_capital_cost (float, optional): Yearly BOP capital cost provided by the user if `user_costs` is True.
                                                   Initializes to zero if `user_costs` is True. Defaults to None.
        yearly_bop_operational_cost (float, optional): Yearly BOP operational cost provided by the user if `user_costs` is True.
                                                       Initializes to zero if `user_costs` is True. Defaults to None.
        initial_ed_capital_cost (float, optional): Initial electrodialysis (ED) capital cost provided by the user if `user_costs` is True.
                                                   Initializes to zero if `user_costs` is True. Defaults to None.
        yearly_ed_capital_cost (float, optional): Yearly ED capital cost provided by the user if `user_costs` is True.
                                                  Initializes to zero if `user_costs` is True. Defaults to None.
        yearly_ed_operational_cost (float, optional): Yearly ED operational cost provided by the user if `user_costs` is True.
                                                      Initializes to zero if `user_costs` is True. Defaults to None.
    """

    electrodialysis_inputs: ElectrodialysisInputs
    mCC_yr: float
    max_theoretical_mCC: float
    infrastructure_type: str = "new"
    user_costs: bool = False

    initial_capital_cost: float = field(default=None, init=False)
    yearly_capital_cost: float = field(default=None, init=False)
    yearly_operational_cost: float = field(default=None, init=False)
    initial_bop_capital_cost: float = field(default=None, init=False)
    yearly_bop_capital_cost: float = field(default=None, init=False)
    yearly_bop_operational_cost: float = field(default=None, init=False)
    initial_ed_capital_cost: float = field(default=None, init=False)
    yearly_ed_capital_cost: float = field(default=None, init=False)
    yearly_ed_operational_cost: float = field(default=None, init=False)

    def __post_init__(self):
        if self.user_costs:
            self.initial_capital_cost = 0.0
            self.yearly_capital_cost = 0.0
            self.yearly_operational_cost = 0.0
            self.initial_bop_capital_cost = 0.0
            self.yearly_bop_capital_cost = 0.0
            self.yearly_bop_operational_cost = 0.0
            self.initial_ed_capital_cost = 0.0
            self.yearly_ed_capital_cost = 0.0
            self.yearly_ed_operational_cost = 0.0


@define
class ElectrodialysisCostOutputs:
    """
    Outputs from the electrodialysis cost model. If default cost model is used all costs are in 2023 USD.

    Attributes:
        initial_capital_cost (float): Total initial capital cost of the electrodialysis system.
        yearly_capital_cost (float): Yearly capital cost for the electrodialysis system.
        yearly_operational_cost (float): Yearly operational cost excluding energy costs for the electrodialysis system.
        initial_bop_capital_cost (float): Initial capital cost for the Balance of Plant (BOP).
        yearly_bop_capital_cost (float): Yearly capital cost for the BOP.
        yearly_bop_operational_cost (float): Yearly operational cost for the BOP, excluding energy costs.
        initial_ed_capital_cost (float): Initial capital cost for the electrodialysis unit.
        yearly_ed_capital_cost (float): Yearly capital cost for the electrodialysis unit.
        yearly_ed_operational_cost (float): Yearly operational cost for the electrodialysis unit, excluding energy costs.
    """

    initial_capital_cost: float
    yearly_capital_cost: float
    yearly_operational_cost: float
    initial_bop_capital_cost: float
    yearly_bop_capital_cost: float
    yearly_bop_operational_cost: float
    initial_ed_capital_cost: float
    yearly_ed_capital_cost: float
    yearly_ed_operational_cost: float


def electrodialysis_cost_model(
    cost_config: ElectrodialysisCostInputs,
) -> ElectrodialysisCostOutputs:
    """
    Calculates the costs associated with electrodialysis based on user inputs or default literature values.

    Args:
        cost_config (ElectrodialysisCostInputs): Configuration object containing user-defined or default inputs
                                                 for the electrodialysis cost model. Includes infrastructure type,
                                                 yearly CO2 capture, and cost settings.

    Returns:
        ElectrodialysisCostOutputs: Object containing calculated capital and operational costs,
                                    both initial and yearly, for the electrodialysis system.

    Calculation Logic:
        - If `user_costs` is True, the model directly uses the costs provided by the user.
        - If `user_costs` is False, the model calculates costs using default values from literature,
          applying learning rates and amortization as necessary.
        - The infrastructure type ('desal', 'swCool', or 'new') determines the baseline costs, which
          are then adjusted based on the modeled CO2 capture capacity.

    Raises:
        ValueError: If the `infrastructure_type` is not one of 'desal', 'swCool', or 'new'.
    """
    ed = cost_config.electrodialysis_inputs

    if cost_config.user_costs:
        # Use user-provided costs
        CEyr = cost_config.yearly_capital_cost
        BOPcapYr = cost_config.yearly_bop_capital_cost
        EDcapYr = cost_config.yearly_ed_capital_cost
        OEnoEyr = cost_config.yearly_operational_cost
        BOPopNoEyr = cost_config.yearly_bop_operational_cost
        EDopNoEyr = cost_config.yearly_ed_operational_cost
        CEi = cost_config.initial_capital_cost
        BOPcapI = cost_config.initial_bop_capital_cost
        EDcapI = cost_config.initial_ed_capital_cost
    else:
        infra_costs = {
            "desal": (253, 197, 204, 159, 48, 39),
            "swCool": (512, 263, 466, 228, 46, 35),
            "new": (1484, 530, 1434, 509, 49, 21),
        }
        if cost_config.infrastructure_type not in infra_costs:
            raise ValueError(
                "`infrastructure_type` must be 'desal', 'swCool', or 'new'"
            )

        CEco2, OEnoEco2, BOPcapCo2, BOPopNoEco2, EDcapCo2, EDopNoEco2 = infra_costs[
            cost_config.infrastructure_type
        ]

        # Apply learning rate
        lr = 0.15
        capFactLit = (
            0.934  # Capacity factor from literature needed for capital cost (93.4%)
        )
        mCC_yr_lit = 20 * ed.co2_mm / 1000 * capFactLit * 24 * 365
        mCC_yr_mod = cost_config.max_theoretical_mCC * capFactLit * 24 * 365
        print("theoretical max", cost_config.max_theoretical_mCC)
        b = -math.log2(1 - lr)
        CEco2Mod = CEco2 * (mCC_yr_mod / mCC_yr_lit) ** (-b)
        BOPcapCo2Mod = BOPcapCo2 * (mCC_yr_mod / mCC_yr_lit) ** (-b)
        EDcapCo2Mod = EDcapCo2 * (mCC_yr_mod / mCC_yr_lit) ** (-b)

        # Calculate CapEx and OpEx Costs
        CEyr = CEco2Mod * mCC_yr_mod
        BOPcapYr = BOPcapCo2Mod * mCC_yr_mod
        EDcapYr = EDcapCo2Mod * mCC_yr_mod
        OEnoEyr = OEnoEco2 * cost_config.mCC_yr
        BOPopNoEyr = BOPopNoEco2 * cost_config.mCC_yr
        EDopNoEyr = EDopNoEco2 * cost_config.mCC_yr

        # Calculate Initial CapEx
        r_int = 0.05  # Rate of interest or return (5%)
        n_pay = 12  # Number of monthly payments in a year (12)
        t_amor = 20  # (years) Amortization time (20 years)
        amort_factor = (1 - (1 + r_int / n_pay) ** (-n_pay * t_amor)) / r_int
        CEi = CEyr * amort_factor
        BOPcapI = BOPcapYr * amort_factor
        EDcapI = EDcapYr * amort_factor

    return ElectrodialysisCostOutputs(
        initial_capital_cost=CEi,
        yearly_capital_cost=CEyr,
        yearly_operational_cost=OEnoEyr,
        initial_bop_capital_cost=BOPcapI,
        yearly_bop_capital_cost=BOPcapYr,
        yearly_bop_operational_cost=BOPopNoEyr,
        initial_ed_capital_cost=EDcapI,
        yearly_ed_capital_cost=EDcapYr,
        yearly_ed_operational_cost=EDopNoEyr,
    )


if __name__ == "__main__":
    pumps = initialize_pumps(
        ed_config=ElectrodialysisInputs(), pump_config=PumpInputs()
    )

    res1 = initialize_power_chemical_ranges(
        ed_config=ElectrodialysisInputs(),
        pump_config=PumpInputs(),
        seawater_config=SeaWaterInputs(),
    )

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

    res = simulate_electrodialysis(
        ranges=res1,
        ed_config=ElectrodialysisInputs(),
        power_profile=exPwr,
        initial_tank_volume_m3=0,
    )

    costs = electrodialysis_cost_model(
        ElectrodialysisCostInputs(
            electrodialysis_inputs=ElectrodialysisInputs(),
            mCC_yr=res.mCC_yr,
            max_theoretical_mCC=max(res1.S1["mCC"]),
        )
    )

    print(costs)
