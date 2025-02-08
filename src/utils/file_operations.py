import yaml
import os
import json
import xlsxwriter

__all__ = ['load_config', 'init_output_dir', 'load_parametric_study', 'save_results', 'ResultsSaver']

def load_config(file_path: str) -> dict:
    """
    Load a YAML file and return a dictionary with the values.
    
    ## Parameters
    ```py
    >>> file_path : str
    ```
    Path to the YAML configuration file.
    
    ## Returns
    ```py
    >>> config : dict
    ```
    Dictionary containing the configuration values.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def init_output_dir() -> None:
    """
    Initialize the output directory structure.
    
    Creates the following directories if they do not exist:
    - `output/`
    - `output/CD/`
    - `output/CS/`
    - `output/parametric_study/`
    
    ## Returns
    ```py
    >>> None
    ```
    """
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('output/CD'):
        os.makedirs('output/CD')
    if not os.path.exists('output/CS'):
        os.makedirs('output/CS')
    if not os.path.exists('output/parametric_study'):
        os.makedirs('output/parametric_study')
    for road_name in ['MotorWay CCN5', 'HFET', 'WLTP']:
        if not os.path.exists(f'output/CD/{road_name}'):
            os.makedirs(f'output/CD/{road_name}')
        if not os.path.exists(f'output/CS/{road_name}'):
            os.makedirs(f'output/CS/{road_name}')

def load_parametric_study(name: str) -> dict:
    """
    Load parametric study data from a JSON file.
    
    ## Parameters
    None
    
    ## Returns
    ```py
    >>> data : dict
    ```
    Dictionary containing the parametric study data.
    """
    with open(f'output/parametric_study_{name}.json', 'r') as file:
        data = json.load(file)
    return data

def save_results(results: dict, file_path: str) -> None:
    """
    Save results to a JSON file.
    
    ## Parameters
    ```py
    >>> results : dict
    ```
    Dictionary containing the results to save.
    ```py
    >>> file_path : str
    ```
    Path to the output JSON file.
    
    ## Returns
    ```py
    >>> None
    ```
    """
    with open(file_path, 'w') as file:
        json.dump(results, file)
    return None


class ResultsSaver:
    def __init__(self, simulation_model):
        self.__dict__.update(simulation_model.__dict__)

    def save_results(self) -> None:
        """Save the results of the simulation in a CSV file"""
        path: str = f"output/{self.current_type}/{self.config['cycle']['road_name']}/results.csv"
        with open(path, "w") as file:
            file.write("Time;Speed;Distance;Acceleration;Motor RPM;Ideal Torque;Reduction Loss;Real Torque;Electric Power;eMotor Efficiency;Power Consumption;Battery Internal Resistance;Battery Intensity;Battery OCV;Battery Tension;Battery SOC;Battery Usage Duration;Battery Pulse Duration\n")
            for i in range(len(self.time)):
                file.write(f"{self.time[i]:.2f};{self.speed[i]:.2f};{self.distance[i]:.2f};{self.acceleration[i]:.2f};{self.motor_rpm[i]:.2f};{self.ideal_torque[i]:.2f};{self.reduction_loss[i]:.2f};{self.real_torque[i]:.2f};{self.mecanical_power[i]:.2f};{self.eMotor_efficiency[i]:.2f};{self.power_consumption[i]:.2f};{self.battery_internal_resistance[i]:.2f};{self.battery_intensity[i]:.2f};{self.battery_OCV[i]:.2f};{self.battery_tension[i]:.2f};{self.battery_SOC[i]:.2f};{self.battery_usage_duration[i]:.2f};{self.battery_pulse_duration[i]:.2f}\n".replace('.', ','))

            file.close()
        
        """Save the results of the simulation in a .xlsx file with cell formatting"""

        # Ensure the output directory exists
        output_path = f"output/{self.current_type}/{self.config['cycle']['road_name']}/"
        os.makedirs(output_path, exist_ok=True)

        # Create workbook and worksheet
        workbook = xlsxwriter.Workbook(f"{output_path}results.xlsx")
        worksheet = workbook.add_worksheet()

        # Create formatting objects
        header_format = workbook.add_format({'bold': True, 'align': 'center'})
        num_format = workbook.add_format({'num_format': '0.0000'})

        # Column headers
        headers = [
            'Time', 'Speed', 'Distance', 'Acceleration', 'Motor RPM', 'Ideal Torque', 
            'Reduction Loss', 'Real Torque', 'Electric Power', 'eMotor Efficiency', 
            'Power Consumption', 'Battery Internal Resistance', 'Battery Intensity', 
            'Battery OCV', 'Battery Tension', 'Battery SOC', 'Battery Usage Duration', 
            'Battery Pulse Duration', 'CO2 emissions', 'Consumption'
        ]

        # Write headers in bold
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)

        # Set column width and format
        worksheet.set_column('A:T', 15, num_format)

        # Write data
        for i in range(len(self.time)):
            worksheet.write(i + 1, 0, self.time[i], num_format)
            worksheet.write(i + 1, 1, self.speed[i], num_format)
            worksheet.write(i + 1, 2, self.distance[i], num_format)
            worksheet.write(i + 1, 3, self.acceleration[i], num_format)
            worksheet.write(i + 1, 4, self.motor_rpm[i], num_format)
            worksheet.write(i + 1, 5, self.ideal_torque[i], num_format)
            worksheet.write(i + 1, 6, self.reduction_loss[i], num_format)
            worksheet.write(i + 1, 7, self.real_torque[i], num_format)
            worksheet.write(i + 1, 8, self.mecanical_power[i], num_format)
            worksheet.write(i + 1, 9, self.eMotor_efficiency[i], num_format)
            worksheet.write(i + 1, 10, self.power_consumption[i], num_format)
            worksheet.write(i + 1, 11, self.battery_internal_resistance[i], num_format)
            worksheet.write(i + 1, 12, self.battery_intensity[i], num_format)
            worksheet.write(i + 1, 13, self.battery_OCV[i], num_format)
            worksheet.write(i + 1, 14, self.battery_tension[i], num_format)
            worksheet.write(i + 1, 15, self.battery_SOC[i], num_format)
            worksheet.write(i + 1, 16, self.battery_usage_duration[i], num_format)
            worksheet.write(i + 1, 17, self.battery_pulse_duration[i], num_format)
            worksheet.write(i + 1, 18, self.cumulative_CO2_emissions[i], num_format)
            worksheet.write(i + 1, 19, self.cumulative_fuel_consumption[i], num_format)
        
        # Add a graph of the SOC evolution
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({
            'name': 'Battery SOC',
            'categories': f'=Sheet1!$A$2:$A${len(self.time) + 1}',
            'values': f'=Sheet1!$P$2:$P${len(self.time) + 1}',
        })
        chart.set_title({'name': 'Battery SOC'})
        chart.set_x_axis({'name': 'Time [s]'})
        chart.set_y_axis({'name': 'SOC [-]'})
        worksheet.insert_chart('A21', chart)

        # Close the workbook
        workbook.close()