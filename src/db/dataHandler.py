from dlmr_tools.data_tool import Data
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp2d, interp1d

from ..utils import logger

__all__ = ['DataHandler']

@dataclass
class DataHandler:
    __datasheet_name: str = "./data/Projet_1_Hybride_série_v3_Stu.xlsx"
    
    def __init__(self, road_name: str) -> None:
        self.__road_name = road_name
        self.__post_init__()
        return None
    
    def __post_init__(self) -> None:
        sheet_name: str = self.__road_name
        self.__spreadsheet = Data(
            data_path=self.__datasheet_name,
            sheet_name=[sheet_name]
        )
        
        self.__data: dict = {}
        
        print(f"Importing data from the sheet {sheet_name}")
        self.__data["time"] = self.__spreadsheet.get_list(sheet_name=sheet_name, list_name="Time")
        self.__data["speed"] = self.__spreadsheet.get_list(sheet_name=sheet_name, list_name="Speed")
            
        self.import_interpolation_tables()
        self.import_datasets()
        return None
            
    def get_data(self) -> dict:
        return self.__data

    def get_time(self, number_of_cycles: int) -> np.ndarray:
        """Return the time in s"""
        single_time: np.ndarray = np.asarray(self.__data["time"])
        total_time: np.ndarray = np.zeros(0)
        for i in range(number_of_cycles):
            total_time = np.append(total_time, single_time + single_time[-1] * i)
        print(f"Total time: {total_time}")
        return total_time
    
    def get_speed(self, number_of_cycles: int) -> np.ndarray:
        """Return the speed in m/s"""
        return self.get_array(number_of_cycles, "speed") / 3.6

    def get_array(self, number_of_cycles: int, array_name: str) -> np.ndarray:
        array: np.ndarray = np.array(self.__data[array_name])
        return np.tile(array, number_of_cycles)
    
    def get_number_of_simulations(self) -> int:
        return len(self.__data)
    
    def import_interpolation_tables(self) -> None:
        """Import the interpolation tables"""
        self.interpolation_tables: dict[str, np.ndarray] = {}
        self.import_interpolation_1D_tables()
        self.import_interpolation_2D_tables()
    
    def import_datasets(self) -> None:
        """Import the datasets"""
        self.datasets: dict[str, np.ndarray] = {}
        datasets_to_import: list[str] = [ _.split(".")[0] for _ in os.listdir("./data/datasets") if _.endswith(".tsv") ]
        for dataset in datasets_to_import:
            self.datasets[dataset] = {}
            path: str = f"./data/datasets/{dataset}.tsv"
            data = np.loadtxt(path, delimiter="\t")
            for i, column in enumerate(data.T):
                self.datasets[dataset][f"column_{i}"] = column
        return None

    def get_dataset(self, dataset_name: str, column_id: int = 0) -> np.ndarray:
        return self.datasets[dataset_name][f"column_{column_id}"]

    def get_WLTP_Ufi(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_dataset(dataset_name="WLTP_Ufi", column_id=0), self.get_dataset(dataset_name="WLTP_Ufi", column_id=1)
    
    def import_interpolation_1D_tables(self) -> None:
        """
        Import the 1D interpolation tables from the ./data/interpolation_1D_tables/*.tsv files
        Hint only 2 columns: x and y
        """
        tables_to_import: list[str] = [ _.split(".")[0] for _ in os.listdir("./data/interpolation_1D_tables") if _.endswith(".tsv") ]
        for table in tables_to_import:
            self.interpolation_tables[table] = {}
            path: str = f"./data/interpolation_1D_tables/{table}.tsv"
            data = np.loadtxt(path, delimiter="\t")
            self.interpolation_tables[table]["x"] = np.array(data[:,0])
            self.interpolation_tables[table]["y"] = np.array(data[:,1])
        return None
    
    def import_interpolation_2D_tables(self) -> None:
        """
        Import the 2D interpolation tables from the ./data/interpolation_2D_tables/*.tsv files
        """
        tables_to_import: list[str] = [ _.split(".")[0] for _ in os.listdir("./data/interpolation_2D_tables") if _.endswith(".tsv") ]
        for table in tables_to_import:
            self.interpolation_tables[table] = {}
            path: str = f"./data/interpolation_2D_tables/{table}.tsv"
            df: pd.DataFrame = pd.read_csv(path, sep="\t", index_col=0)
            self.interpolation_tables[table]["x"] = np.array(df.columns.astype(float).values)
            self.interpolation_tables[table]["y"] = np.array(df.index.astype(float).values)
            self.interpolation_tables[table]["z"] = np.array(df.values)
        return None
    
    def interpolate_2D_generic(self, table_name: str, x: float, y: float) -> float:
        """
        Interpolate a 2D table
        - `table_name`: Name of the table
        - `x`: x value
        - `y`: y value
        """
        x_table: np.ndarray = self.interpolation_tables[table_name]["x"] # 1D array
        y_table: np.ndarray = self.interpolation_tables[table_name]["y"] # 1D array
        z_table: np.ndarray = self.interpolation_tables[table_name]["z"] # 2D array
        f: interp2d = interp2d(x=x_table, y=y_table, z=z_table, kind="linear")
        return f(x, y)[0]
    
    def interpolate_1D_generic(self, table_name: str, x: float) -> float:
        """
        Interpolate a 1D table
        - `table_name`: Name of the table
        - `x`: x value
        """
        x_table: np.ndarray = self.interpolation_tables[table_name]["x"]
        y_table: np.ndarray = self.interpolation_tables[table_name]["y"]
        f = interp1d(x=x_table, y=y_table, kind="linear")
        return f(x)
    
    def interpolate_ICE_fuel_consumption(self, RPM: float, CME: float) -> float:
        """
        Interpolate the ICE fuel consumption from the 2D interpolation table
        - ICE fuel consumption [g/s]
        - `RPM`: Engine speed [rpm]
        - `CME`: Engine torque [N.m]
        """
        return self.interpolate_2D_generic(table_name="ICE_fuel_consumption", x=RPM, y=CME)
    
    def interpolate_ICE_specific_fuel_consumption(self, RPM: float, CME: float) -> float:
        """
        Interpolate the ICE specific fuel consumption from the 2D interpolation table
        - ICE specific fuel consumption [g/kWh]
        - `RPM`: Engine speed [rpm]
        - `CME`: Engine torque [N.m]
        """
        return self.interpolate_2D_generic(table_name="ICE_specific_fuel_consumption", x=RPM, y=CME)
    
    def interpolate_eMotor_efficiency(self, RPM: float, MTE: float) -> float:
        """
        Interpolate the eMotor efficiency from the 2D interpolation table
        - eMotor efficiency [-]
        - `RPM`: Engine speed [rpm]
        - `MTE`: Engine torque [N.m]
        """
        return self.interpolate_2D_generic(table_name="eMotor_efficiency", x=RPM, y=MTE)

    def interpolate_losses(self, RPM: float, MTE: float) -> float:
        """
        Interpolate the losses from the 2D interpolation table
        - Losses [W]
        - `RPM`: Engine speed [rpm]
        - `MTE`: Engine torque [N.m]
        """
        return self.interpolate_2D_generic(table_name="losses", x=RPM, y=MTE)
    
    def interpolate_internal_battery_resistance_during_charge(self, battery_SOC: float, pulse_duration: float) -> float:
        """
        Interpolate the internal battery resistance during charge at 25°C from the 2D interpolation table
        - Internal battery resistance during charge at 25°C [mOhm]\\
            Hint: To divide by 1000 and multiply by the number of cells
        - `battery_SOC`: State of charge of the battery [-]
        - `pulse_duration`: Pulse duration [s]
        """
        return self.interpolate_2D_generic(table_name="internal_battery_resistance_during_charge", x=battery_SOC, y=pulse_duration)
    
    def interpolate_internal_battery_resistance_during_discharge(self, battery_SOC: float, pulse_duration: float) -> float:
        """
        Interpolate the internal battery resistance during discharge at 25°C from the 2D interpolation table
        - Internal battery resistance during discharge at 25°C [mOhm]\\
            Hint: To divide by 1000 and multiply by the number of cells
        - `battery_SOC`: State of charge of the battery [-]
        - `pulse_duration`: Pulse duration [s]
        """
        return self.interpolate_2D_generic(table_name="internal_battery_resistance_during_discharge", x=battery_SOC, y=pulse_duration)
    
    def interpolate_OCV(self, battery_SOC: float) -> float:
        """
        Interpolate the open circuit voltage from the 1D interpolation table
        - Open circuit voltage [V]
        - `battery_SOC`: State of charge of the battery [-]
        """
        return self.interpolate_1D_generic(table_name="OCV", x=battery_SOC)