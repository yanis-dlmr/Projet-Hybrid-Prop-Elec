# Project powertrain hybrid

This project simulates the performance of a hybrid vehicle under various conditions and configurations. It includes parametric studies to analyze the impact of different parameters on the vehicle's performance.

## Table of Contents

- Installation
- Usage
- Project Structure
- Configuration
- Running the Simulation
- Results
- License

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yourproject.git
    cd yourproject
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Configure the simulation parameters in `config.yaml`.

2. Run the simulation:
    ```sh
    python main.py
    ```

## Project Structure

```
.gitignore
config.yaml
data/
    datasets/
    interpolation_1D_tables/
    interpolation_2D_tables/
main_parametric_study.log
main.log
main.py
output/
    assets/
    CD/
    CS/
    parametric_study/
    parametric_study_CCN5_CD_CS_cycles_weighted_combined_CO2.json
    parametric_study_CCN5_CD_cycles_weighted_combined_CO2.json
    parametric_study_CD_CS_cycles_weighted_combined_CO2.json
report.tex
requirements.txt
src/
    __init__.py
    core/
        simulationModel.py
        car.py
    utils/
        graph.py
        file_operations.py
        logger.py
```

## Configuration

The simulation parameters are defined in the `config.yaml` file. Here is an example configuration:

```yaml
car:
  WHEEL_RADIUS: 0.35  # m
  DEMULTIPLICATION: 11  # -
  WEIGHT: 1800  # kg
  GRAVITY: 9.81  # m/s^2
  AUXILIARY_POWER: 300  # W
  THERMAL_ENGINE_PRESENCE: true

battery:
  N_CELLS: 84  # -
  CD_INITIAL_SOC: 0.95  # -
  CS_INITIAL_SOC: 0.22  # -
  RANGE_SOC: 0.002  # -
  TARGET_SOC: 0.22  # -
  CELL_CAPACITY: 40  # Ah

drag:
  F0: 150  # N
  F1: 0.0  # = 0.0 * 3.6 N/mps
  F2: 0.618192  # = 0.0477 * 3.6**2 N/mps^2

engine:
  TH_ENGINE_MINIMAL_TIME_ON: 50  # s

cycle:
  road_name: "WLTP"  # -
  number_of_CD_cycles: 6  # -
  number_of_CS_cycles: 1  # -

output:
  SAVE_RESULTS: true  # -
  PLOT_RESULTS: true  # -

logger:
  LOG_LEVEL: "DEBUG"  # -
  LOG_FILE: "main.log"  # -
  LOG_FORMAT: "%(levelname)s - %(message)s"  # %(asctime)s - %(name)s -
```

## Running the Simulation

To run the simulation, execute the following command:

```sh
python main.py
```

