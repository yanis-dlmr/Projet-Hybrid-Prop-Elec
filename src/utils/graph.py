import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from dlmr_tools.graph_tool import Graph_1D

from src.utils.file_operations import load_parametric_study

from . import logger

__all__ = ['Graph', 'ResultsPlotter', 'plot_parametric_study_range_soc_runtime', 'plot_parametric_study_CD_cycles_weighted_combined_CO2', 'plot_parametric_study_CD_CS_cycles_weighted_combined_CO2']

@dataclass
class Graph:
        
    @staticmethod
    def simple_plot(x_values: np.ndarray, list_y_values: list[np.ndarray], labels: list[str], title: str, x_label: str, y_label: str) -> None:
        plt.figure()
        for y_values, label in zip(list_y_values, labels):
            plt.plot(x_values, y_values, label=label)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        
        return None
    
    @staticmethod
    def subplot_plot(x_values: np.ndarray, list_y_values: list[np.ndarray], x_label: str, y_labels: list[str]) -> None:
        nrows: int = 2
        ncolumns: int = len(list_y_values) // nrows + len(list_y_values) % nrows
        fig, ax = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=(10, 5 * nrows))
        for i, y_values, y_label in zip(range(len(list_y_values)), list_y_values, y_labels):
            axi = ax[i // ncolumns, i % ncolumns]
            axi.plot(x_values, y_values)
            axi.set_xlabel(x_label)
            axi.set_ylabel(y_label)
        plt.show()
        
        
        return None


def plot_parametric_study_range_soc_runtime():
    data = load_parametric_study('range_soc_runtime')

    # Plot the average CO2 emissions
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.imshow(data['CS']['average_CO2_emissions'], cmap='gist_rainbow', aspect='auto')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(data['TH_ENGINE_MINIMAL_TIME_ON'])))
    ax.set_yticks(np.arange(len(data['RANGE_SOC'])))
    ax.set_xticklabels(data['TH_ENGINE_MINIMAL_TIME_ON'])
    range_soc = data['RANGE_SOC']
    range_soc = [f'{range_soc[i]:.4f}' for i in range(len(range_soc))]
    ax.set_yticklabels(range_soc)
    ax.set_xlabel('Thermal engine minimal runtime [s]')
    ax.set_ylabel('Range $\epsilon_{SOC}$ [-]')
    ax.set_title('Average CO$_2$ emissions CS (g/km)')
    # annotate the values
    for i in range(len(data['RANGE_SOC'])):
        for j in range(len(data['TH_ENGINE_MINIMAL_TIME_ON'])):
            ax.text(j, i, f'{data["CS"]["average_CO2_emissions"][i][j]:.2f}', ha='center', va='center', color='black')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.tight_layout()
    plt.savefig('output/parametric_study/average_CO2_emissions_CS.png')
    plt.close()
    
    # Plot the number of restart
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.imshow(data['CS']['number_of_restart'], cmap='gist_rainbow', aspect='auto')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(data['TH_ENGINE_MINIMAL_TIME_ON'])))
    ax.set_yticks(np.arange(len(data['RANGE_SOC'])))
    ax.set_xticklabels(data['TH_ENGINE_MINIMAL_TIME_ON'])
    range_soc = data['RANGE_SOC']
    range_soc = [f'{range_soc[i]:.4f}' for i in range(len(range_soc))]
    ax.set_yticklabels(range_soc)
    ax.set_xlabel('Thermal engine minimal time on [s]')
    ax.set_ylabel('Range $\epsilon_{SOC}$ [-]')
    ax.set_title('Number of restart')
    # annotate the values
    for i in range(len(data['RANGE_SOC'])):
        for j in range(len(data['TH_ENGINE_MINIMAL_TIME_ON'])):
            ax.text(j, i, f'{data["CS"]["number_of_restart"][i][j]:.0f}', ha='center', va='center', color='black')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.tight_layout()
    plt.savefig('output/parametric_study/number_of_restart.png')
    plt.close()
    
    # plot average_th_engine_runtime
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.imshow(data['CS']['average_th_engine_runtime'], cmap='gist_rainbow', aspect='auto')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(data['TH_ENGINE_MINIMAL_TIME_ON'])))
    ax.set_yticks(np.arange(len(data['RANGE_SOC'])))
    ax.set_xticklabels(data['TH_ENGINE_MINIMAL_TIME_ON'])
    range_soc = data['RANGE_SOC']
    range_soc = [f'{range_soc[i]:.4f}' for i in range(len(range_soc))]
    ax.set_yticklabels(range_soc)
    ax.set_xlabel('Thermal engine minimal time on [s]')
    ax.set_ylabel('Range $\epsilon_{SOC}$ [-]')
    ax.set_title('Average thermal engine runtime [s]')
    # annotate the values
    for i in range(len(data['RANGE_SOC'])):
        for j in range(len(data['TH_ENGINE_MINIMAL_TIME_ON'])):
            ax.text(j, i, f'{data["CS"]["average_th_engine_runtime"][i][j]:.2f}', ha='center', va='center', color='black')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.tight_layout()
    plt.savefig('output/parametric_study/average_th_engine_runtime.png')
    plt.close()
    
    return None


def plot_parametric_study_CD_cycles_weighted_combined_CO2(road_name):
    """
    One histogram with the groups:
    - CD average_CO2_emissions
    - CS average_CO2_emissions
    - CD CO2_mix
    - CS CO2_mix
    - weighted_CO2_combined
    """
    data = load_parametric_study(f'{road_name}_CD_cycles_weighted_combined_CO2')
    important_data = {}
    chartjs_colors = [ "#36a2eb", "#ff6384", "#9966ff", "#ffce56", "#4bc0c0" ]
    for key, value in data.items():
        important_data[key] = [
            value['CD']['average_CO2_emissions'],
            value['CS']['average_CO2_emissions'],
            value['CD']['CO2_mix'],
            value['CS']['CO2_mix'],
            value['weighted_CO2_combined']
        ]
    
    groups: list[str] = [ "Charge Depleting", "Charge Sustaining", "Charge Depleting mix", "Charge Sustaining mix", "Weighted combined" ]
    
    x = np.arange(len(groups))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for key, values in important_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=key, color=chartjs_colors[multiplier])
        ax.bar_label(rects, labels=[f"{v:.1f}" for v in values], padding=3)
        multiplier += 1
    
    ax.set_ylabel('$CO_2$ emissions [g/km]')
    ax.set_title('$CO_2$ emissions')
    ax.set_xticks(x + width, groups)
    ax.legend()
    ax.set_ylim(0, 170)
        
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.tight_layout()
    plt.savefig(f'output/parametric_study/{road_name}_CD_cycles_weighted_combined_CO2.png', dpi=300)
    
def plot_parametric_study_CD_CS_cycles_weighted_combined_CO2(road_name):
    """
    One histogram with the groups:
    - CD average_CO2_emissions
    - CS average_CO2_emissions
    - CD CO2_mix
    - CS CO2_mix
    - weighted_CO2_combined
    """
    data = load_parametric_study(f'{road_name}_CD_CS_cycles_weighted_combined_CO2')
    important_data = {}
    chartjs_colors = [ "#36a2eb", "#ff6384", "#9966ff", "#ffce56", "#4bc0c0" ]
    for key, value in data.items():
        important_data[key] = [
            value['CD']['average_CO2_emissions'],
            value['CS']['average_CO2_emissions'],
            value['CD']['CO2_mix'],
            value['CS']['CO2_mix'],
            value['weighted_CO2_combined']
        ]
    
    groups: list[str] = [ "Charge Depleting", "Charge Sustaining", "Charge Depleting mix", "Charge Sustaining mix", "Weighted combined" ]
    
    x = np.arange(len(groups))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for key, values in important_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=key, color=chartjs_colors[multiplier])
        ax.bar_label(rects, labels=[f"{v:.1f}" for v in values], padding=3)
        multiplier += 1
    
    ax.set_ylabel('$CO_2$ emissions [g/km]')
    ax.set_title('$CO_2$ emissions')
    ax.set_xticks(x + width, groups)
    ax.legend()
    ax.set_ylim(0, 170)
        
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.tight_layout()
    plt.savefig(f'output/parametric_study/{road_name}_CD_CS_cycles_weighted_combined_CO2.png', dpi=300)


class ResultsPlotter:
    def __init__(self, simulation_model):
        self.__dict__.update(simulation_model.__dict__)
        
    def plot_results(self) -> None:
        # SOC and engine state
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='SOC [-]', sci=False)
        Graph.plot(x=self.time, y=self.battery_SOC, color='chartjs_blue', marker='', label='Battery SOC')
        Graph.add_axis()
        Graph.setup_secondary_axis(ylabel='Engine state [-]', sci=False)
        Graph.plot(x=self.time, y=self.engine_state, color='chartjs_red', marker='', label='Engine state', axis_number=1)
        # Graph.show(dx=0.2, dy=1.15, ncol=2)
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/SOC_state", ncol=2, dy=1.19, dx=0.3)
        Graph.delete() #test
        
        # CO2 emissions
        Graph = Graph_1D(figsize=(5, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='CO2 emissions [g]', sci=False)
        Graph.plot(x=self.time, y=self.cumulative_CO2_emissions, color='chartjs_purple', marker='', label='Cumulative CO2 emissions')
        # Graph.show(dx=0.2, dy=1.15, ncol=2)
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/CO2", ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # eMotor 2
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Consumption [W]', sci=False)
        # Provided by thermal engine
        power_thermal_engine = self.power_provided_by_thermal_engine / self.car.EL2_efficiency
        Graph.plot(x=self.time, y=power_thermal_engine, color='chartjs_red', marker='', label='Power provided by thermal engine', linestyle='-')
        Graph.plot(x=self.time, y=self.power_provided_by_thermal_engine, color='chartjs_orange', marker='', label='Power provided by eMotor2', linestyle='-')
        Graph.add_axis()
        Graph.setup_secondary_axis(ylabel='Fuel consumption [g]', sci=False)
        Graph.plot(x=self.time, y=self.cumulative_fuel_consumption, color='chartjs_green', marker='', label='Cumulative fuel consumption', axis_number=1)
        # Graph.show(dx=0.2, dy=1.15, ncol=2)
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/eMoror2_power", ncol=2, dy=1.19, dx=0.15)
        Graph.delete()
        
        # Battery
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Consumption [W]', sci=False)
        smoothed_curve = np.convolve(self.requested_power_consumption_on_battery, np.ones(20)/20, mode='same')
        Graph.plot(x=self.time, y= -smoothed_curve, color='chartjs_green', marker='', label='Requested power consumption on battery', linestyle='-')
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/Battery_power", ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # eMotor 1
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Power consumption [W]', sci=False)
        smoothed_curve = np.convolve(self.power_consumption, np.ones(10)/10, mode='same')
        Graph.plot(x=self.time[:1800], y=smoothed_curve[:1800], color='chartjs_blue', marker='', label='Power consumption by eMotor1', linestyle='-')
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/eMoror1_power", ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # eMotor_efficiency
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Efficiency [-]', sci=False)
        Graph.plot(x=self.time[:1800], y=self.eMotor_efficiency[:1800], color='chartjs_blue', marker='', label='eMotor efficiency', linestyle='-')
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/eMotor_efficiency", ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # torque
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Torque [Nm]', sci=False, ymin=-100, ymax=100)
        Graph.plot(x=self.time[:1800], y=self.real_torque[:1800], color='chartjs_pink', marker='', label='Real Torque', linestyle='-')
        # reduction loss
        Graph.add_axis()
        Graph.setup_secondary_axis(ylabel='Reduction Loss [W]', sci=False, ymin=-1500, ymax=1500)
        Graph.plot(x=self.time[:1800], y=self.reduction_loss[:1800], color='chartjs_purple', marker='', label='Reduction Loss', axis_number=1)
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/Torque", ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        # speed
        Graph = Graph_1D(figsize=(10, 4), fontsize=11)
        Graph.setup_axis(xlabel='Time [s]', ylabel='Speed [km/h]', sci=False)
        Graph.plot(x=self.time, y=self.speed * 3.6, color='chartjs_orange', marker='', label='Speed', linestyle='-')
        Graph.save(filename=f"output/{self.current_type}/{self.config['cycle']['road_name']}/Speed", ncol=2, dy=1.15, dx=0.2)
        Graph.delete()
        
        return None