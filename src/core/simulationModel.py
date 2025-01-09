from dataclasses import dataclass
import numpy as np
from dlmr_tools.graph_tool import Graph_1D

from ..db import DataHandler
from ..utils import logger, Graph
from .car import Car

__all__ = [ 'SimulationModel' ]

    
configuration = {
    'road_name': 'WLTP',
    'number_of_CD_cycles': 4,
    'number_of_CS_cycles': 1
}

@dataclass
class SimulationModel:
    __dataHandler: DataHandler = DataHandler( road_name=configuration['road_name'] )
    
    def run(self) -> None:
        types: list[str] = [ "CD", "CS" ]
        for i, type in enumerate(types):
            self.__car = Car(dataHandler=self.__dataHandler, type=type)
            self.current_type = type
            logger.info(f"Simulation {type}")
            self.process_data()
            self.save_results()
            self.plot_results()

    def process_data(self) -> None:
        self.time: np.ndarray = self.__dataHandler.get_time(number_of_cycles=configuration[f'number_of_{self.current_type}_cycles'])
        self.speed: np.ndarray = self.__dataHandler.get_speed(number_of_cycles=configuration[f'number_of_{self.current_type}_cycles'])
        self.WLTP_Ufi: tuple = self.__dataHandler.get_WLTP_Ufi()
        
        computations: list[callable] = [ 
            self.compute_distance,
            self.compute_acceleration,
            self.compute_motor_rpm,
            self.compute_vehicle_forces,
            self.compute_torque,
            
            self.compute_power_consumption,
            self.compute_battery_thermal_motor_usage,
            self.compute_CO2
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
        self.acceleration[0] = 0
        for i in range(1, len(self.time)):
            self.acceleration[i] = (self.speed[i] - self.speed[i - 1]) / (self.time[i] - self.time[i - 1])
    
    def compute_motor_rpm(self) -> None:
        """Compute the motor rpm in revolutions per minute"""
        self.motor_rpm: np.ndarray = self.speed * self.__car.DEMULTIPLICATION / self.__car.WHEEL_PERIMETER * 60
        
    def compute_vehicle_forces(self) -> None:
        """Compute the friction, gravity and drag of the vehicle in N"""
        self.vehicle_friction: np.ndarray = self.__car.F0 + self.__car.F1 * self.speed + self.__car.F2 * self.speed**2
        self.vehicle_acceleration: np.ndarray = self.__car.WEIGHT * self.acceleration
        self.vehicle_drag: np.ndarray = 0.0
        self.vehicle_total_force: np.ndarray = self.vehicle_friction + self.vehicle_acceleration + self.vehicle_drag
    
    def compute_torque(self) -> None:
        """Compute:
        - the ideal torque of the vehicle in Nm
        - the reduction loss of the vehicle in W \\
            Hint: W = N.m/s
        - the real torque of the vehicle in Nm"""
        self.ideal_torque: np.ndarray = self.__car.WHEEL_RADIUS * self.vehicle_total_force / self.__car.DEMULTIPLICATION
        self.reduction_loss: np.ndarray = np.zeros(len(self.time))
        for i in range(len(self.time)):
            self.reduction_loss[i] = self.__car.losses(RPM=self.motor_rpm[i], MTE=self.ideal_torque[i])
        self.real_torque: np.ndarray = np.zeros(len(self.time))
        for i in range(len(self.time)):
            if self.motor_rpm[i] == 0:
                self.real_torque[i] = 0
            else:
                if self.reduction_loss[i] == 0:
                    self.real_torque[i] = self.ideal_torque[i]
                else:
                    self.real_torque[i] = self.ideal_torque[i] + self.reduction_loss[i] / self.motor_rpm[i] * 2 * np.pi / 60
    
    def compute_power_consumption(self) -> None:
        """Compute the power consumption of the vehicle in W"""
        self.mecanical_power: np.ndarray = np.zeros(len(self.time))
        self.eMotor_efficiency: np.ndarray = np.zeros(len(self.time))
        self.power_consumption: np.ndarray = np.zeros(len(self.time))
        for i in range(len(self.time)):
            self.mecanical_power[i] = self.real_torque[i] * self.motor_rpm[i] * 2 * np.pi / 60
            self.eMotor_efficiency[i] = self.__car.eMotor_efficiency(RPM=self.motor_rpm[i], MTE=self.real_torque[i])
            
            if self.eMotor_efficiency[i] == 0:
                self.power_consumption[i] = self.__car.AUXILIARY_POWER
            else:
                if self.real_torque[i] >= 0: #traction
                    self.power_consumption[i] = self.mecanical_power[i] / self.eMotor_efficiency[i] + self.__car.AUXILIARY_POWER
                else: #regen
                    self.power_consumption[i] = self.mecanical_power[i] * self.eMotor_efficiency[i] + self.__car.AUXILIARY_POWER


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
        self.remaining_power: np.ndarray = np.zeros(len(self.time))
        self.remaining_power_for_battery: np.ndarray = np.zeros(len(self.time))
        self.instantaneous_fuel_consumption: np.ndarray = np.zeros(len(self.time))
        self.cumulative_fuel_consumption: np.ndarray = np.zeros(len(self.time))
        
        ########### ELECTRICAL SIDE ###########
        self.battery_internal_resistance, self.battery_intensity, self.battery_OCV, self.battery_tension, self.battery_SOC, self.battery_usage_duration, self.battery_pulse_duration = [np.zeros(len(self.time)) for _ in range(7)]
        self.battery_state: np.ndarray = np.zeros(len(self.time))
        self.battery_cumulative_time: np.ndarray = np.zeros(len(self.time))
        
        self.battery_SOC[0] = self.__car.BATTERY_INITIAL_SOC
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
                if (self.engine_state[i-1] == 1) and (self.engine_cumulative_time[i-1] < self.__car.TH_ENGINE_MINIMAL_TIME_ON):
                    self.engine_state[i] = 1
                # Turn off the engine if the battery SOC is above the target threshold
                elif self.battery_SOC[prev_i] > self.__car.BATTERY_TARGET_SOC + self.__car.BATTERY_SOC_RANGE:
                    self.engine_state[i] = 0
                # Turn on the engine if the battery SOC is below the target threshold
                elif self.battery_SOC[prev_i] <= self.__car.BATTERY_TARGET_SOC - self.__car.BATTERY_SOC_RANGE:
                    self.engine_state[i] = 1
                # Default behavior
                else:
                    self.engine_state[i] = self.engine_state[i-1] 

            if self.__car.THERMAL_ENGINE_PRESENCE == False:
                self.engine_state[i] = 0
            
            if self.engine_state[i] == self.engine_state[i-1]:
                self.engine_cumulative_time[i] = self.engine_cumulative_time[i-1] + self.time[i] - self.time[i-1]
            else:
                self.engine_cumulative_time[i] = self.time[i] - self.time[i-1]
            
            if self.engine_state[i] == 1:
                # Get optimal power from thermal engine
                remaining_power = self.__car.EMOTOR2_OPTIMAL_CME * self.__car.EMOTOR2_OPTIMAL_RPM * 2 * np.pi / 60
                # Get electrical power from thermal engine
                self.power_provided_by_thermal_engine[i] = remaining_power * self.__car.EL2_efficiency
                # Get fuel consumption
                self.instantaneous_fuel_consumption[i] = self.__car.ICE_fuel_consumption(RPM=self.__car.EMOTOR2_OPTIMAL_RPM, CME=self.__car.EMOTOR2_OPTIMAL_CME) * (self.time[i] - self.time[i-1])
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
                self.battery_internal_resistance[i] = self.__car.internal_battery_resistance_during_discharge(battery_SOC=self.battery_SOC[prev_i], pulse_duration=self.battery_cumulative_time[prev_i])
            else:
                self.battery_internal_resistance[i] = self.__car.internal_battery_resistance_during_charge(battery_SOC=self.battery_SOC[prev_i], pulse_duration=self.battery_cumulative_time[prev_i])
            # Battery OCV
            self.battery_OCV[i] = self.__car.battery_OCV(battery_SOC=self.battery_SOC[prev_i]) * self.__car.N_CELLS
            
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
                self.battery_SOC[i] = self.battery_SOC[prev_i] - self.battery_intensity[prev_i] / (3600 * self.__car.BATTERY_CELL_CAPACITY) * (self.time[i] - self.time[prev_i])
            
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
        
        for i in range(len(self.time)):
            if i == 0:
                self.instantaneous_CO2_emissions[i] = COLD_START + CAR_HEATING + self.instantaneous_fuel_consumption[i] * CO2_MASS_OVER_FUEL_MASS * (1 + DYNAMISQUE_ICE) * (self.time[i] - self.time[i-1])
                self.cumulative_CO2_emissions[i] = self.instantaneous_CO2_emissions[i]
            else:
                self.instantaneous_CO2_emissions[i] = self.instantaneous_fuel_consumption[i] * CO2_MASS_OVER_FUEL_MASS * (1 + DYNAMISQUE_ICE) * (self.time[i] - self.time[i-1])
                if (self.engine_state[i] == 1) and (self.engine_state[i-1] == 0):
                    self.instantaneous_CO2_emissions[i] += RESTART
                    if first_restart == False:
                        first_restart = True
                        logger.info(f"First restart {self.__car.type} at {self.time[i]} s and {self.distance[i]} m")
                self.cumulative_CO2_emissions[i] = self.cumulative_CO2_emissions[i-1] + self.instantaneous_CO2_emissions[i]
        logger.info(f"Total fuel consumption {self.__car.type}: {self.cumulative_fuel_consumption[-1]} g")
        logger.info(f"Average fuel consumption {self.__car.type}: {self.cumulative_fuel_consumption[-1] / self.distance[-1] * 1000} g/km")
        logger.info(f"Total CO2 emissions {self.__car.type}: {self.cumulative_CO2_emissions[-1]} g")
        logger.info(f"Average CO2 emissions {self.__car.type}: {self.cumulative_CO2_emissions[-1] / self.distance[-1] * 1000} g/km")

    def plot_results(self) -> None:
        # SOC and engine state
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='SOC [-]', sci=False)
        Graph.plot(x=self.time, y=self.battery_SOC, color='chartjs_blue', marker='', label='Battery SOC')
        Graph.add_axis()
        Graph.setup_secondary_axis(ylabel='Engine state [-]', sci=False)
        Graph.plot(x=self.time, y=self.engine_state, color='chartjs_red', marker='', label='Engine state', axis_number=1)
        # Graph.show(dx=0.2, dy=1.15, ncol=2)
        Graph.save(filename=f'output/{self.current_type}/SOC_state', ncol=2, dy=1.19, dx=0.3)
        Graph.delete() #test
        
        # CO2 emissions
        Graph = Graph_1D(figsize=(5, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='CO2 emissions [g]', sci=False)
        Graph.plot(x=self.time, y=self.cumulative_CO2_emissions, color='chartjs_purple', marker='', label='Cumulative CO2 emissions')
        # Graph.show(dx=0.2, dy=1.15, ncol=2)
        Graph.save(filename=f'output/{self.current_type}/CO2', ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # eMotor 2
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Consumption [W]', sci=False)
        # Provided by thermal engine
        power_thermal_engine = self.power_provided_by_thermal_engine / self.__car.EL2_efficiency
        Graph.plot(x=self.time, y=power_thermal_engine, color='chartjs_red', marker='', label='Power provided by thermal engine', linestyle='-')
        Graph.plot(x=self.time, y=self.power_provided_by_thermal_engine, color='chartjs_orange', marker='', label='Power provided by eMotor2', linestyle='-')
        Graph.add_axis()
        Graph.setup_secondary_axis(ylabel='Fuel consumption [g]', sci=False)
        Graph.plot(x=self.time, y=self.cumulative_fuel_consumption, color='chartjs_green', marker='', label='Cumulative fuel consumption', axis_number=1)
        # Graph.show(dx=0.2, dy=1.15, ncol=2)
        Graph.save(filename=f'output/{self.current_type}/eMoror2_power', ncol=2, dy=1.19, dx=0.15)
        Graph.delete()
        
        # Battery
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Consumption [W]', sci=False)
        smoothed_curve = np.convolve(self.requested_power_consumption_on_battery, np.ones(20)/20, mode='same')
        Graph.plot(x=self.time, y= -smoothed_curve, color='chartjs_green', marker='', label='Requested power consumption on battery', linestyle='-')
        Graph.save(filename=f'output/{self.current_type}/Battery_power', ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # eMotor 1
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Power consumption [W]', sci=False)
        smoothed_curve = np.convolve(self.power_consumption, np.ones(10)/10, mode='same')
        Graph.plot(x=self.time[:1800], y=smoothed_curve[:1800], color='chartjs_blue', marker='', label='Power consumption by eMotor1', linestyle='-')
        Graph.save(filename=f'output/{self.current_type}/eMoror1_power', ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # eMotor_efficiency
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Efficiency [-]', sci=False)
        Graph.plot(x=self.time[:1800], y=self.eMotor_efficiency[:1800], color='chartjs_blue', marker='', label='eMotor efficiency', linestyle='-')
        Graph.save(filename=f'output/{self.current_type}/eMotor_efficiency', ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # torque
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Torque [Nm]', sci=False, ymin=-100, ymax=100)
        Graph.plot(x=self.time[:1800], y=self.real_torque[:1800], color='chartjs_pink', marker='', label='Real Torque', linestyle='-')
        # reduction loss
        Graph.add_axis()
        Graph.setup_secondary_axis(ylabel='Reduction Loss [W]', sci=False, ymin=-1500, ymax=1500)
        Graph.plot(x=self.time[:1800], y=self.reduction_loss[:1800], color='chartjs_purple', marker='', label='Reduction Loss', axis_number=1)
        Graph.save(filename=f'output/{self.current_type}/Torque', ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # remaining_power_for_battery
        # self.simple_plot(y_values=[self.remaining_power_for_battery], labels=["Remaining power for battery [W]"], title="Remaining_power_for_battery")
        # power consumption emotor1
        # self.simple_plot(y_values=[self.power_consumption], labels=["Power consumption eMotor1 [W]"], title="Power_consumption_eMotor1")
        # self.simple_plot(y_values=[self.cumulative_fuel_consumption, self.cumulative_CO2_emissions], labels=["Cumulative fuel consumption [g]", "Cumulative CO2 emissions [g]"], title="Cumulative_fuel_consumption_and_CO2_emissions")
        
        return None
        self.subplot_plot(
            y_values=[self.instantaneous_fuel_consumption, self.cumulative_fuel_consumption, self.instantaneous_CO2_emissions, self.cumulative_CO2_emissions],
            labels=["Instantaneous fuel consumption [g]", "cumulative_fuel_consumption [g]", "Instantaneous CO2 emissions [g]", "Cumulative CO2 emissions [g]"]
        )
        self.subplot_plot(
            y_values=[self.speed, self.cumulative_fuel_consumption, self.battery_SOC, self.power_consumption, self.real_torque, self.battery_intensity],
            labels=["Speed [m/s]", "cumulative_fuel_consumption [g]", "Battery SOC [-]", "Power consumption eMotor1 [W]", "Real Torque [Nm]", "Battery Intensity [A]"]
        )
    
    def simple_plot(self, y_values: np.ndarray, labels: list[str], title: str) -> None:
        #graph = Graph()
        #graph.simple_plot(
        #    x_values=self.time,
        #    list_y_values=[y_values],
        #    labels=[label],
        #    title="Vehicle Simulation",
        #    x_label="Time [s]",
        #    y_label="Values"
        #)
        colors = ['chartjs_blue', 'chartjs_orange', 'chartjs_green', 'chartjs_red', 'chartjs_purple', 'chartjs_pink']
        graph = Graph_1D(figsize=(5, 4), fontsize=11)
        graph.setup_axis(xlabel='Time [s]', ylabel='Values [SI]', sci=False)
        for i, y in enumerate(y_values):
            graph.plot(x=self.time, y=y, color=colors[i], marker='', label=labels[i])
        #graph.show(dx=0.2, dy=1.15, ncol=2)
        graph.save(filename=f'output/{self.current_type}/{title}', ncol=2, dy=1.15, dx=0.2)
        graph.delete()
    
    def subplot_plot(self, y_values: list[np.ndarray], labels: list[str]) -> None:
        graph = Graph()
        graph.subplot_plot(
            x_values=self.time,
            list_y_values=y_values,
            x_label="Time [s]",
            y_labels=labels
        )
    
    def save_results(self) -> None:
        """Save the results of the simulation in a CSV file"""
        path: str = "results.csv"
        with open(path, "w") as file:
            file.write("Time;Speed;Distance;Acceleration;Motor RPM;Ideal Torque;Reduction Loss;Real Torque;Electric Power;eMotor Efficiency;Power Consumption;Battery Internal Resistance;Battery Intensity;Battery OCV;Battery Tension;Battery SOC;Battery Usage Duration;Battery Pulse Duration\n")
            for i in range(len(self.time)):
                file.write(f"{self.time[i]:.2f};{self.speed[i]:.2f};{self.distance[i]:.2f};{self.acceleration[i]:.2f};{self.motor_rpm[i]:.2f};{self.ideal_torque[i]:.2f};{self.reduction_loss[i]:.2f};{self.real_torque[i]:.2f};{self.mecanical_power[i]:.2f};{self.eMotor_efficiency[i]:.2f};{self.power_consumption[i]:.2f};{self.battery_internal_resistance[i]:.2f};{self.battery_intensity[i]:.2f};{self.battery_OCV[i]:.2f};{self.battery_tension[i]:.2f};{self.battery_SOC[i]:.2f};{self.battery_usage_duration[i]:.2f};{self.battery_pulse_duration[i]:.2f}\n".replace('.', ','))

            file.close()