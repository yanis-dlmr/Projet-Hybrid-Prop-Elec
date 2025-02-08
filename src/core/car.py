import numpy as np
from dataclasses import dataclass

from ..db import DataHandler

__all__ = [ 'Car' ]

@dataclass
class Car:
    WHEEL_RADIUS: float = None # m
    DEMULTIPLICATION: int = None
    WEIGHT: float = None # kg
    GRAVITY: float = None # m/s^2
    AUXILIARY_POWER: float = None # W
    THERMAL_ENGINE_PRESENCE: bool = None
    
    N_CELLS: int = None
    CD_INITIAL_SOC: float = None
    CS_INITIAL_SOC: float = None
    RANGE_SOC: float = None
    TARGET_SOC: float = None
    CELL_CAPACITY: float = None # Ah
    
    F0: float = None # N
    F1: float = None # N/mps
    F2: float = None # N/mps^2
    
    TH_ENGINE_MINIMAL_TIME_ON: float = None # s
    EMOTOR2_OPTIMAL_RPM: float = None # RPM
    EMOTOR2_OPTIMAL_CME: float = None # Nm
    EL2_efficiency: float = None # [-]
    
    
    def __init__(self, config: dict, dataHandler: DataHandler, type: str) -> None:
        for key, value in config.items():
            for subKey, subValue in value.items():
                if hasattr(self, subKey):
                    setattr(self, subKey, subValue)
        
        self.__dataHandler = dataHandler
        self.type = type
        
        if type == "CD":
            self.BATTERY_INITIAL_SOC = self.CD_INITIAL_SOC
        elif type == "CS":
            self.BATTERY_INITIAL_SOC = self.CS_INITIAL_SOC
    
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