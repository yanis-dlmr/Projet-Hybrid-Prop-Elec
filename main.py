import configparser
import os

from src import SimulationModel


def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def init_output_dir():
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('output/CD'):
        os.makedirs('output/CD')
    if not os.path.exists('output/CS'):
        os.makedirs('output/CS')

def run():
    init_output_dir()
    config = load_config()
    print(config.sections())
    print(config['drag']['F2'])
    return
    simulationModel = SimulationModel()
    simulationModel.run()

if __name__ == '__main__':
    run()