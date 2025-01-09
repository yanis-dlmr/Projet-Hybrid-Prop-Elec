import numpy as np
from dataclasses import dataclass

from ..db import DataHandler

__all__ = [ 'Car' ]

@dataclass
class Car:
    THERMAL_ENGINE_PRESENCE: bool = True
    
    WHEEL_RADIUS: float = 0.35 # m
    DEMULTIPLICATION: int = 11
    F0: float = 150 # N
    F1: float = 0.0 * 3.6 # N/mps
    F2: float = 0.0477 * 3.6**2 # N/mps^2
    WEIGHT: float = 1800 # kg
    AUXILIARY_POWER: float = 300 # W
    GRAVITY: float = 9.81 # m/s^2
    
    N_CELLS: int = 84
    # BATTERY_INITIAL_SOC: float = 0.95 #0.22 # 0.95
    TH_ENGINE_MINIMAL_TIME_ON: float = 10 # s
    BATTERY_SOC_RANGE: float = 0.005
    BATTERY_TARGET_SOC: float = 0.22
    BATTERY_CELL_CAPACITY: float = 40 # Ah
    
    EMOTOR2_OPTIMAL_RPM: float = 3000 # RPM
    EMOTOR2_OPTIMAL_CME: float = 130 # Nm
    EL2_efficiency: float = 0.9 # [-]
    
    
    def __init__(self, dataHandler: DataHandler, type: str) -> None:
        self.__dataHandler = dataHandler
        self.type = type
        
        if type == "CD":
            self.BATTERY_INITIAL_SOC = 0.95
        elif type == "CS":
            self.BATTERY_INITIAL_SOC = 0.22
    
    @property
    def WHEEL_PERIMETER(self) -> float:
        return 2 * np.pi * self.WHEEL_RADIUS
    
    @property
    def EMOTOR2_OPTIMAL_SPECIFIC_FUEL_CONSUMPTION(self) -> float:
        return self.ICE_specific_fuel_consumption(RPM=self.EMOTOR2_OPTIMAL_RPM, CME=self.EMOTOR2_OPTIMAL_CME)
    
    def ICE_fuel_consumption(self, RPM: float, CME: float) -> float:
        """Return the fuel consumption of the ICE in g/s"""
        return self.__dataHandler.interpolate_ICE_fuel_consumption(RPM=RPM, CME=CME)
    
    def ICE_specific_fuel_consumption(self, RPM: float, CME: float) -> float:
        """Return the specific fuel consumption of the ICE in g/kWh"""
        return self.__dataHandler.interpolate_ICE_specific_fuel_consumption(RPM=RPM, CME=CME)
    
    def eMotor_efficiency(self, RPM: float, MTE: float) -> float:
        """Return the efficiency of the electric motor"""
        return self.__dataHandler.interpolate_eMotor_efficiency(RPM=RPM, MTE=MTE)
    
    def losses(self, RPM: float, MTE: float) -> float:
        """Return the losses of the vehicle in W"""
        return self.__dataHandler.interpolate_losses(RPM=RPM, MTE=MTE)
    
    def internal_battery_resistance_during_charge(self, battery_SOC: float, pulse_duration: float) -> float:
        """Return the internal battery resistance during charge at 25°C in Ohm"""
        return self.__dataHandler.interpolate_internal_battery_resistance_during_charge(battery_SOC=battery_SOC, pulse_duration=pulse_duration) / 1000 * self.N_CELLS
    
    def internal_battery_resistance_during_discharge(self, battery_SOC: float, pulse_duration: float) -> float:
        """Return the internal battery resistance during discharge at 25°C in Ohm"""
        return self.__dataHandler.interpolate_internal_battery_resistance_during_discharge(battery_SOC=battery_SOC, pulse_duration=pulse_duration) / 1000 * self.N_CELLS
    
    def battery_OCV(self, battery_SOC: float) -> float:
        """Return the battery open circuit voltage in V"""
        return self.__dataHandler.interpolate_OCV(battery_SOC=battery_SOC)