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
    """ "
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
    volAddAcid: list  # (m^3) Volume of acid added to tanks at each time
    volAddBase: list  # (m^3) Volume of base added to tanks at each time
    Q_in: list  # (m^3/s) Intake flow rate at each time step
    Q_out: list  # (m^3/s) Outtake flow rate at each time step
    ca_t: list  # (mol/L) Acid concentration at each time step
    cb_t: list  # (mol/L) Base concentration at each time step
    S_t: list  # Scenario for each time step
    nON: int = (
        0  # Timesteps when capture occurs (S1-3) used to determine capacity factor
    )


def power_chemical_ranges(config: ElectroDialysisInputs) -> ElectrodialysisOutputs:
    # Define the range sizes
    N_range = config.N_edMax - config.N_edMin + 1
    S2_tot_range = (N_range * (N_range + 1)) // 2
    range2_size = S2_tot_range

    S2_ranges = np.zeros((S2_tot_range, 2))

    # Fill the ranges array
    k = 0
    for i in range(N_range):
        for j in range(N_range - i):
            S2_ranges[k, 0] = i + config.N_edMin
            S2_ranges[k, 1] = j
            k += 1

    # Define the array names
    keys = [
        "volAddAcid",
        "volAddBase",
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
    S2 = {key: np.zeros(range2_size) for key in keys}
    S3 = {key: np.zeros(N_range) for key in keys}
    S4 = {key: np.zeros(N_range) for key in keys}


# def initialize_scenarios():

if __name__ == "__main__":
    pumps = initialize_pumps(
        ed_config=ElectroDialysisInputs(), pump_config=PumpInputs()
    )

    print(pumps.pumpA.Q_max)
