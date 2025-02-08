import json
import numpy as np
import os

from src import SimulationModel
from src.utils import load_config, init_output_dir
from src.utils.file_operations import save_results
from src.utils.graph import plot_parametric_study_range_soc_runtime, plot_parametric_study_CD_cycles_weighted_combined_CO2, plot_parametric_study_CD_CS_cycles_weighted_combined_CO2


def run() -> None:
    """
    Run the simulation with the given configuration.
    """
    simulationModel = SimulationModel(config)
    simulationModel.run()

def run_parametric_study_range_soc_runtime(config):
    """
    Run the parametric study, by varying the parameters:
    - `RANGE_SOC` from 0.001 to 0.03 with a step of 0.001
    - `TH_ENGINE_MINIMAL_TIME_ON` from 10 to 60 s with a step of 10 s
    """
    # range_soc_list = np.array( [ range_soc / 1000 for range_soc in range(1, 31, 2) ] )
    small_values = np.arange(0.0001, 0.0011, 0.0001)  # 0.0001 à 0.001 (incrément de 0.0001)
    madium_values = np.arange(0.001, 0.01, 0.001)  # 0.001 à 0.01 (incrément de 0.001)
    large_values = np.arange(0.01, 0.031, 0.001)  # 0.01 à 0.03 (incrément de 0.001)
    range_soc_list = np.unique(np.concatenate((small_values, madium_values, large_values)))
    
    #th_engine_minimal_time_on_list = np.array( [ th_engine_minimal_time_on for th_engine_minimal_time_on in range(0, 61, 10) ] )
    small_step = np.arange(0, 11, 1)  # De 0 à 10 inclus, pas de 1
    large_step = np.arange(10, 61, 10)  # De 10 à 60 inclus, pas de 10
    th_engine_minimal_time_on_list = np.unique(np.concatenate((small_step, large_step)))
    
    results_2D_array = np.zeros( ( len(range_soc_list), len(th_engine_minimal_time_on_list) ) )
    data = {
        'RANGE_SOC': range_soc_list,
        'TH_ENGINE_MINIMAL_TIME_ON': th_engine_minimal_time_on_list,
        'CS': {},
        'CD': {}
    }
    for i, range_soc in enumerate(data['RANGE_SOC']):
        for j, th_engine_minimal_time_on in enumerate(data['TH_ENGINE_MINIMAL_TIME_ON']):
            config['battery']['RANGE_SOC'] = range_soc
            config['engine']['TH_ENGINE_MINIMAL_TIME_ON'] = th_engine_minimal_time_on
            simulationModel = SimulationModel(config)
            simulationModel.run()
            results = simulationModel.get_results()
            for key in results['CS']:
                if key not in data['CS']:
                    data['CS'][key] = results_2D_array.copy()
                data['CS'][key][i, j] = results['CS'][key]
            for key in results['CD']:
                if key not in data['CD']:
                    data['CD'][key] = results_2D_array.copy()
                data['CD'][key][i, j] = results['CD'][key]

    # TypeError: Object of type ndarray is not JSON serializable
    data = { key: data[key].tolist() if isinstance(data[key], np.ndarray) else data[key] for key in data }
    # same for sub-dictionaries
    data['CS'] = { key: data['CS'][key].tolist() for key in data['CS'] }
    data['CD'] = { key: data['CD'][key].tolist() for key in data['CD'] }
    save_results(data, 'output/parametric_study_range_soc_runtime.json')
    
    plot_parametric_study_range_soc_runtime()
    
    return None

def run_parametric_study_CD_cycles_weighted_combined_CO2(road_name: str = '') -> None:
    """
    Run the parametric study, by varying the parameters:
    - cycle number_of_CD_cycles from 3 to 5 with a step of 1
    Then plot the results.
    One histogram with:
    - CD average_CO2_emissions
    - CS average_CO2_emissions
    - CD CO2_mix
    - CS CO2_mix
    - weighted_CO2_combined
    """
    number_of_CD_cycles_list = np.array( [ number_of_CD_cycles for number_of_CD_cycles in range(3, 6, 1) ] )
    data: dict = {
        f'Number of CD cycles: {number_of_CD_cycles}': {} for number_of_CD_cycles in number_of_CD_cycles_list
    }
    
    for number_of_CD_cycles in number_of_CD_cycles_list:
        config['cycle']['number_of_CD_cycles'] = number_of_CD_cycles
        config['cycle']['number_of_CS_cycles'] = 1
        simulationModel = SimulationModel(config)
        simulationModel.run()
        results = simulationModel.get_results()
        data[f'Number of CD cycles: {number_of_CD_cycles}'] = results
        
    save_results(data, f'output/parametric_study_{road_name}_CD_cycles_weighted_combined_CO2.json')
    
    plot_parametric_study_CD_cycles_weighted_combined_CO2(road_name)

def run_parametric_study_CD_CS_cycles_weighted_combined_CO2(road_name: str = '') -> None:
    """
    Run the parametric study, by varying the parameters:
    - cycle number_of_CD_cycles and number_of_CS_cycles from 3 to 5 with a step of 1
    Then plot the results.
    One histogram with:
    - CD average_CO2_emissions
    - CS average_CO2_emissions
    - CD CO2_mix
    - CS CO2_mix
    - weighted_CO2_combined
    """
    number_of_CD_cycles_list = np.array( [ number_of_CD_cycles for number_of_CD_cycles in range(3, 6, 1) ] )
    data: dict = {
        f'Number of CD/CS cycles: {number_of_CD_cycles}': {} for number_of_CD_cycles in number_of_CD_cycles_list
    }
    
    for number_of_CD_cycles in number_of_CD_cycles_list:
        config['cycle']['number_of_CD_cycles'] = number_of_CD_cycles
        config['cycle']['number_of_CS_cycles'] = number_of_CD_cycles
        simulationModel = SimulationModel(config)
        simulationModel.run()
        results = simulationModel.get_results()
        data[f'Number of CD/CS cycles: {number_of_CD_cycles}'] = results
        
    save_results(data, f'output/parametric_study_{road_name}_CD_CS_cycles_weighted_combined_CO2.json')
    
    plot_parametric_study_CD_CS_cycles_weighted_combined_CO2(road_name)
    
def run_parametric_study_CCN5_HFET_fuel_consumption():
    """
    Run the parametric study, by varying the parameters:
    - cycle road_name from 'MotorWay CCN5' to 'HFET'
    For each, copy and save the graph in the output directory:
    - output/CD/eMoror2_power.png
    - output/CS/eMotor2_power.png
    """
    road_names = ['MotorWay CCN5', 'HFET', 'WLTP']
    for road_name in road_names:
        config['cycle']['road_name'] = road_name
        simulationModel = SimulationModel(config)
        simulationModel.run()
    return None
    
    
if __name__ == '__main__':
    init_output_dir()
    config = load_config('config.yaml')
    #run()
    #run_parametric_study_CD_cycles_weighted_combined_CO2()
    #run_parametric_study_CD_CS_cycles_weighted_combined_CO2()
    # run_parametric_study_range_soc_runtime(config)
    run_parametric_study_CCN5_HFET_fuel_consumption()
    #plot_parametric_study_CD_cycles_weighted_combined_CO2()
    #plot_parametric_study_CD_CS_cycles_weighted_combined_CO2()