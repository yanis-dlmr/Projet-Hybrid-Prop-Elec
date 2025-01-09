from src import SimulationModel


if __name__ == '__main__':
    simulationModel: SimulationModel = SimulationModel()
    simulationModel.run()


""" 
Couple à fournir:

ICE fuel consumption 
RPM donné doit choisir le couple

couple à fournir = couple moteur thermique optimal + couple moteur électrique déduit ?
"""