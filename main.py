import configparser

from src import SimulationModel


def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def run():
    config = load_config()
    print(config.sections())
    print(config['drag']['F2'])
    return
    simulationModel = SimulationModel()
    simulationModel.run()

if __name__ == '__main__':
    run()