import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

from . import logger

__all__ = ['Graph']

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