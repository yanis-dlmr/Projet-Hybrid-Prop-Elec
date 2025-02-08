from dataclasses import dataclass, field
import numpy as np

from ..db import DataHandler
from ..utils import logger, ResultsSaver, ResultsPlotter
from .car import Car

__all__ = [ 'SimulationModel' ]


@dataclass
class SimulationModel:
    config: dict
    result: dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self.__dataHandler: DataHandler = DataHandler( road_name=self.config['cycle']['road_name'] )
    
    def run(self) -> None:
        types: list[str] = ["CD", "CS"]
        for type in types:
            self.car = Car(config=self.config, dataHandler=self.__dataHandler, type=type)
            self.current_type = type
            logger.info(f"Simulation {type}")
            self.process_data()

            if self.config['output']['SAVE_RESULTS']:
                ResultsSaver(self).save_results()
            if self.config['output']['PLOT_RESULTS']:
                ResultsPlotter(self).plot_results()

    def process_data(self) -> None:
        self.time: np.ndarray = self.__dataHandler.get_time(number_of_cycles=self.config['cycle'][f'number_of_{self.current_type}_cycles'])
        self.speed: np.ndarray = self.__dataHandler.get_speed(number_of_cycles=self.config['cycle'][f'number_of_{self.current_type}_cycles'])
        self.WLTP_Ufi: tuple = self.__dataHandler.get_WLTP_Ufi()
        
        computations: list[callable] = [ 
            self.compute_distance,
            self.compute_acceleration,
            self.compute_motor_rpm,
            self.compute_vehicle_forces,
            self.compute_torque,
            
            self.compute_power_consumption,
            self.compute_battery_thermal_motor_usage,
            self.compute_CO2,
            self.compute_with_utility_factor
        ]
        for computation in computations:
            logger.debug(f"Computing {computation.__name__}")
            try:
                computation()
            except Exception as e:
                logger.error(f"Error while computing {computation.__name__}: {e}")

    def compute_distance(self) -> None:
        """Compute the distance covered by the vehicle in meters"""
        dt: np.ndarray = np.diff(self.time, prepend=self.time[0])
        self.distance: np.ndarray = np.cumsum(self.speed * dt)
    
    def compute_acceleration(self) -> None:
        """Compute the acceleration of the vehicle in m/s^2"""
        self.acceleration: np.ndarray = np.zeros(len(self.time))
        self.acceleration[1:] = np.diff(self.speed) / np.diff(self.time)
        # replace the nan or inf values by 0
        self.acceleration = np.nan_to_num(self.acceleration, nan=0, posinf=0, neginf=0)
    
    def compute_motor_rpm(self) -> None:
        """Compute the motor rpm in revolutions per minute"""
        self.motor_rpm: np.ndarray = self.speed * self.car.DEMULTIPLICATION / self.car.WHEEL_PERIMETER * 60
        
    def compute_vehicle_forces(self) -> None:
        """Compute the friction, gravity and drag of the vehicle in N"""
        self.vehicle_friction: np.ndarray = self.car.F0 + self.car.F1 * self.speed + self.car.F2 * self.speed**2
        self.vehicle_acceleration: np.ndarray = self.car.WEIGHT * self.acceleration
        self.vehicle_drag: np.ndarray = 0.0
        self.vehicle_total_force: np.ndarray = self.vehicle_friction + self.vehicle_acceleration + self.vehicle_drag
    
    def compute_torque(self) -> None:
        """Compute:
        - the ideal torque of the vehicle in Nm
        - the reduction loss of the vehicle in W \\
            Hint: W = N.m/s
        - the real torque of the vehicle in Nm"""
        self.ideal_torque: np.ndarray = self.car.WHEEL_RADIUS * self.vehicle_total_force / self.car.DEMULTIPLICATION
        self.reduction_loss: np.ndarray = np.array([
            self.car.losses(RPM=rpm, MTE=torque)
            for rpm, torque in zip(self.motor_rpm, self.ideal_torque)
        ])
        self.real_torque: np.ndarray = np.where(
            self.motor_rpm == 0,
            0,
            self.ideal_torque + self.reduction_loss / self.motor_rpm * 2 * np.pi / 60,
        )
    
    def compute_power_consumption(self) -> None:
        """Compute the power consumption of the vehicle in W"""
        self.mecanical_power: np.ndarray = (
            self.real_torque * self.motor_rpm * 2 * np.pi / 60
        )
        self.eMotor_efficiency: np.ndarray = np.array([
            self.car.eMotor_efficiency(RPM=rpm, MTE=torque)
            for rpm, torque in zip(self.motor_rpm, self.real_torque)
        ])
        self.power_consumption: np.ndarray = np.where(
            self.eMotor_efficiency == 0,
            self.car.AUXILIARY_POWER,
            np.where(
                self.real_torque >= 0,
                self.mecanical_power / self.eMotor_efficiency + self.car.AUXILIARY_POWER,
                self.mecanical_power * self.eMotor_efficiency + self.car.AUXILIARY_POWER,
            ),
        )

    def compute_battery_thermal_motor_usage(self) -> None:
        """TODO change this description and put in functions
        
        Compute the battery usage of the vehicle including:
        - The internal resistance of the battery in Ohm
        - The intensity of the battery in A
        - The U0CV of the battery in V
        - The tension of the battery in V
        - The SOC of the battery
        - The right SOC of the battery
        - The usage duration of the battery in s
        - The pulse duration of the battery in s"""
        
        
        ########### THERMAL SIDE ###########
        self.engine_state: np.ndarray = np.zeros(len(self.time))
        self.engine_cumulative_time: np.ndarray = np.zeros(len(self.time))
        self.engine_time_on: np.ndarray = np.zeros(len(self.time))
        self.remaining_power: np.ndarray = np.zeros(len(self.time))
        self.remaining_power_for_battery: np.ndarray = np.zeros(len(self.time))
        self.instantaneous_fuel_consumption: np.ndarray = np.zeros(len(self.time))
        self.cumulative_fuel_consumption: np.ndarray = np.zeros(len(self.time))
        
        ########### ELECTRICAL SIDE ###########
        self.battery_internal_resistance, self.battery_intensity, self.battery_OCV, self.battery_tension, self.battery_SOC, self.battery_usage_duration, self.battery_pulse_duration = [np.zeros(len(self.time)) for _ in range(7)]
        self.battery_state: np.ndarray = np.zeros(len(self.time))
        self.battery_cumulative_time: np.ndarray = np.zeros(len(self.time))
        
        self.battery_SOC[0] = self.car.BATTERY_INITIAL_SOC
        self.battery_usage_duration[0] = 0
        self.battery_pulse_duration[0] = 2
        
        self.requested_power_consumption_on_battery: np.ndarray = np.zeros(len(self.time))
        self.power_provided_by_thermal_engine: np.ndarray = np.zeros(len(self.time))
        
        ########### COMPUTATION ###########
        for i in range(len(self.time)):
            if i == 0:
                prev_i = 0
            else:
                prev_i = i - 1
                ########### THERMAL SIDE ###########
                # Check if the engine should stay on due to minimal runtime
                if (self.engine_state[i-1] == 1) and (self.engine_cumulative_time[i-1] < self.car.TH_ENGINE_MINIMAL_TIME_ON):
                    self.engine_state[i] = 1
                # Turn off the engine if the battery SOC is above the target threshold
                elif self.battery_SOC[prev_i] > self.car.TARGET_SOC + self.car.RANGE_SOC:
                    self.engine_state[i] = 0
                # Turn on the engine if the battery SOC is below the target threshold
                elif self.battery_SOC[prev_i] <= self.car.TARGET_SOC - self.car.RANGE_SOC:
                    self.engine_state[i] = 1
                # Default behavior
                else:
                    self.engine_state[i] = self.engine_state[i-1] 

            if self.car.THERMAL_ENGINE_PRESENCE == False:
                self.engine_state[i] = 0
            
            if self.engine_state[i] == 1:
                self.engine_time_on[i] = self.time[i] - self.time[i-1]
            
            if self.engine_state[i] == self.engine_state[i-1]:
                self.engine_cumulative_time[i] = self.engine_cumulative_time[i-1] + self.time[i] - self.time[i-1]
            else:
                self.engine_cumulative_time[i] = self.time[i] - self.time[i-1]
            
            if self.engine_state[i] == 1:
                # Get optimal power from thermal engine
                remaining_power = self.car.EMOTOR2_OPTIMAL_CME * self.car.EMOTOR2_OPTIMAL_RPM * 2 * np.pi / 60
                # Get electrical power from thermal engine
                self.power_provided_by_thermal_engine[i] = remaining_power * self.car.EL2_efficiency
                # Get fuel consumption
                self.instantaneous_fuel_consumption[i] = self.car.ICE_fuel_consumption(RPM=self.car.EMOTOR2_OPTIMAL_RPM, CME=self.car.EMOTOR2_OPTIMAL_CME) * (self.time[i] - self.time[i-1])
                self.cumulative_fuel_consumption[i] = self.cumulative_fuel_consumption[i-1] + self.instantaneous_fuel_consumption[i]
            else:
                self.power_provided_by_thermal_engine[i] = 0
                self.instantaneous_fuel_consumption[i] = 0
                self.cumulative_fuel_consumption[i] = self.cumulative_fuel_consumption[i-1]
            
            ########### LINKING THERMAL AND ELECTRICAL SIDES ###########
            #print("Power consumption", self.power_consumption[i], "Power provided by thermal engine", self.power_provided_by_thermal_engine[i], self.engine_state[i])
            self.requested_power_consumption_on_battery[i] = self.power_consumption[i] - self.power_provided_by_thermal_engine[i]
            if self.requested_power_consumption_on_battery[i] < 0:
                self.remaining_power_for_battery[i] = 0
            else:
                self.remaining_power_for_battery[i] = self.requested_power_consumption_on_battery[i]
            
            ########### ELECTRICAL SIDE ###########
            
            # Battery internal resistance
            if self.requested_power_consumption_on_battery[i] > 0:
                self.battery_internal_resistance[i] = self.car.internal_battery_resistance_during_discharge(battery_SOC=self.battery_SOC[prev_i], pulse_duration=self.battery_cumulative_time[prev_i])
            else:
                self.battery_internal_resistance[i] = self.car.internal_battery_resistance_during_charge(battery_SOC=self.battery_SOC[prev_i], pulse_duration=self.battery_cumulative_time[prev_i])
            # Battery OCV
            self.battery_OCV[i] = self.car.battery_OCV(battery_SOC=self.battery_SOC[prev_i]) * self.car.N_CELLS
            
            # Battery intensity 
            # formula: I = OCV - sqrt(OCV^2 - 4 * R * P) / (2 * R)
            discriminant = self.battery_OCV[i]**2 - 4 * self.battery_internal_resistance[i] * self.requested_power_consumption_on_battery[i]
            if (discriminant < 0):
                self.battery_intensity[i] = 0
                # print("Discriminant < 0")
            else:
                self.battery_intensity[i] = (self.battery_OCV[i] - np.sqrt(discriminant)) / (2 * self.battery_internal_resistance[i])
                # print("(+) Intensity", self.battery_intensity[i])

            # Battery tension
            # formula: U = OCV - R * I
            self.battery_tension[i] = self.battery_OCV[i] - self.battery_intensity[prev_i] * self.battery_internal_resistance[i]
            
            # Battery SOC
            # formula: SOC = SOC - 1/C * integral(I * dt)
            if i != 0:
                self.battery_SOC[i] = self.battery_SOC[prev_i] - self.battery_intensity[prev_i] / (3600 * self.car.CELL_CAPACITY) * (self.time[i] - self.time[prev_i])
            
            # Battery state
            battery_on: bool = self.remaining_power_for_battery[i] > 0
            self.battery_state[i] = 1 if battery_on else 0
            if self.battery_state[prev_i] == self.battery_state[i]:
                # Never exceed 30 seconds in order to interpolate the internal resistance
                if self.battery_cumulative_time[prev_i] >= 30:
                    self.battery_cumulative_time[i] = 30
                else:
                    self.battery_cumulative_time[i] = self.battery_cumulative_time[prev_i] + self.time[i] - self.time[prev_i]
            else:
                self.battery_cumulative_time[i] = self.time[i] - self.time[prev_i]

    def compute_CO2(self) -> None:
        """
        Compute CO2 emissions
        """
        DYNAMISQUE_ICE: float = 0.03 # 3%
        CO2_MASS_OVER_FUEL_MASS: float = 3.04266666666667
        COLD_START: float = 52 # gCO2
        RESTART: float = 0.5 # gCO2/start
        CAR_HEATING: float = 75 # gCO2
        
        self.instantaneous_CO2_emissions: np.ndarray = np.zeros(len(self.time))
        self.cumulative_CO2_emissions: np.ndarray = np.zeros(len(self.time))
        
        first_restart: bool = False
        number_of_restart: int = 0
        
        for i in range(len(self.time)):
            if i == 0:
                self.instantaneous_CO2_emissions[i] = COLD_START + CAR_HEATING + self.instantaneous_fuel_consumption[i] * CO2_MASS_OVER_FUEL_MASS * (1 + DYNAMISQUE_ICE) * (self.time[i] - self.time[i-1])
                self.cumulative_CO2_emissions[i] = self.instantaneous_CO2_emissions[i]
            else:
                self.instantaneous_CO2_emissions[i] = self.instantaneous_fuel_consumption[i] * CO2_MASS_OVER_FUEL_MASS * (1 + DYNAMISQUE_ICE) * (self.time[i] - self.time[i-1])
                if (self.engine_state[i] == 1) and (self.engine_state[i-1] == 0):
                    self.instantaneous_CO2_emissions[i] += RESTART
                    number_of_restart += 1
                    if first_restart == False:
                        first_restart = True
                        autonomy_distance = self.distance[i] 
                        autonomy_time = self.time[i]
                        logger.info(f"First restart {self.car.type} at {autonomy_time} s and {autonomy_distance} m")
                self.cumulative_CO2_emissions[i] = self.cumulative_CO2_emissions[i-1] + self.instantaneous_CO2_emissions[i]
                
        self.result[self.current_type] = {
            "autonomy_distance": autonomy_distance,
            "autonomy_time": autonomy_time,
            "total_fuel_consumption": self.cumulative_fuel_consumption[-1],
            "average_fuel_consumption": self.cumulative_fuel_consumption[-1] / self.distance[-1] * 1000,
            "total_CO2_emissions": self.cumulative_CO2_emissions[-1],
            "average_CO2_emissions": self.cumulative_CO2_emissions[-1] / self.distance[-1] * 1000,
            "number_of_restart": number_of_restart,
            "average_th_engine_runtime": np.sum(self.engine_time_on) / number_of_restart
        }
        logger.info(f"Total fuel consumption {self.car.type}: {self.cumulative_fuel_consumption[-1]} g")
        logger.info(f"Average fuel consumption {self.car.type}: {self.cumulative_fuel_consumption[-1] / self.distance[-1] * 1000} g/km")
        logger.info(f"Total CO2 emissions {self.car.type}: {self.cumulative_CO2_emissions[-1]} g")
        logger.info(f"Average CO2 emissions {self.car.type}: {self.cumulative_CO2_emissions[-1] / self.distance[-1] * 1000} g/km")
    
    def compute_with_utility_factor(self) -> None:
        """
        Weighted charge-depleting $CO_2$ emissions:
        \begin{equation}
            M_{\text{CO}_2,\text{CD,weighted}} = \frac{\sum_{j=1}^{k} \left( M_{\text{CO}_2,\text{CD},j} \times UF_j \right)}{\sum_{j=1}^{k} UF_j}
        \end{equation}

        The weighted, combined $CO_2$ emissions are calculated using the following formula:

        \begin{equation}
            M_{\text{CO}_2,\text{weighted,combined}} =
            M_{\text{CO}_2,\text{CD,weighted}} \cdot \sum_{j=1}^{n} UF_j 
            + M_{\text{CO}_2,\text{CS}} \cdot \left( 1 - \sum_{j=1}^{n} UF_j \right)
        \end{equation}
        """
        logger.info("Computing with utility factor")
        phase_distance: np.ndarray = self.WLTP_Ufi[0] * 1000
        phase_Ufi: np.ndarray = self.WLTP_Ufi[1][:]
        phase_CO2_emissions: np.ndarray = np.zeros(len(phase_distance))
        phase_CO2_emissions_per_km: np.ndarray = np.zeros(len(phase_distance))
        
        total_distance: float = 0
        previous_distance_index: int = 0
        
        for i in range(len(phase_distance)):
            total_distance += phase_distance[i]
            
            if total_distance > self.distance[-1]:
                # crop the list
                phase_distance = phase_distance[:i]
                phase_Ufi = phase_Ufi[:i]
                phase_CO2_emissions = phase_CO2_emissions[:i]
                phase_CO2_emissions_per_km = phase_CO2_emissions_per_km[:i]
                break
            
            distance_index: int = np.argmin(np.abs(self.distance - total_distance))
            
            if i == 0:
                phase_CO2_emissions[i] = self.cumulative_CO2_emissions[distance_index]
            else:
                phase_CO2_emissions[i] = self.cumulative_CO2_emissions[distance_index] - self.cumulative_CO2_emissions[previous_distance_index]
            phase_CO2_emissions_per_km[i] = phase_CO2_emissions[i] / (phase_distance[i] / 1000)
            logger.debug(f"During phase {i}: {phase_CO2_emissions[i]:.0f} g CO2 emitted on {phase_distance[i]:.0f} m ie {phase_CO2_emissions_per_km[i]:.0f} g/km")
            previous_distance_index = distance_index
        
        if self.current_type == "CD":
            # Weighted charge-depleting CO2 emissions
            self.weighted_CO2_per_km_CD: float = np.sum(phase_CO2_emissions_per_km * phase_Ufi) / np.sum(phase_Ufi)
        elif self.current_type == "CS":
            # Weighted, combined CO2 emissions
            self.CO2_CD_mix: float = self.weighted_CO2_per_km_CD * np.sum(phase_Ufi)
            self.CO2_CS_mix: float = self.result["CS"]["average_CO2_emissions"] * (1 - np.sum(phase_Ufi))
            self.weighted_CO2_combined: float = self.CO2_CD_mix + self.CO2_CS_mix
            
            logger.info(f"CO2_CD_mix: Charge-depleting CO2 emissions: {self.CO2_CD_mix:.0f} g/km")
            logger.info(f"CO2_CS_mix: Charge-sustaining CO2 emissions: {self.CO2_CS_mix:.0f} g/km")
            logger.info(f"CO2_mix: Combined CO2 emissions: {self.weighted_CO2_combined:.0f} g/km")
            
            
            self.result['CS']['CO2_mix'] = self.CO2_CS_mix
            self.result['CD']['CO2_mix'] = self.CO2_CD_mix
            self.result['weighted_CO2_combined'] = self.weighted_CO2_combined
        
        return None

    def get_results(self) -> dict:
        return self.result
    
