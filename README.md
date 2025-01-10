# Transport Air Pollution Scenario ABM

# README

This repository (or script) contains an **Agent-Based Model (ABM)** that simulates individual-level mobility and exposure to air pollution under different urban scenarios. The model uses the **[Mesa](https://github.com/projectmesa/mesa)** library for its ABM framework, **[GeoPandas](https://github.com/geopandas/geopandas)** and **[Shapely](https://github.com/shapely/shapely)** for geographic data handling, and additional libraries for parallelization, route calculation (via **[OSRM](https://project-osrm.org/)**), and data analysis. The primary goal is to estimate traffic behavior and air pollution concentrations on an hourly or sub-hourly basis, and to evaluate how these influence human exposure to NO2.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation & Dependencies](#installation--dependencies)
4. [Required Data & Folder Structure](#required-data--folder-structure)
5. [Usage](#usage)
6. [Script Structure & Main Components](#script-structure--main-components)
7. [Running the Simulation](#running-the-simulation)
8. [Outputs](#outputs)
9. [Customization & Scenarios](#customization--scenarios)
10. [Performance & Parallelization](#performance--parallelization)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)
13. [Contact](#contact)

---

## 1. Overview

This ABM simulates:

- **Human agents** with demographic attributes, daily schedules, and location-based activities (e.g., work, school, shopping).
- **Travel decisions** (mode choice, routes) based on environmental attributes and personal demographics.
- **Traffic density** on road networks derived from aggregated individual travel.
- **Air pollution levels (NO2)**, calculated from a hybrid dispersion model that uses baseline concentration, traffic volume, and built-environment features.
- **Exposure** of each agent to hourly NO2 concentrations, differentiating between indoor/outdoor exposure.

It leverages real-world data (e.g., geospatial building footprints, road networks, travel patterns, demographic data) to simulate within the Amsterdam region (or your chosen city).

---

## 2. Key Features

- **Simultaneous Activation** of agents (via [Mesa's `SimultaneousActivation`](https://mesa.readthedocs.io/en/master/apis/time.html)).
- **OSRM-based routing** for bike, drive, walk, or transit modes.  
- **Conditional route reuse** (caching of frequently traveled routes).
- **Hourly traffic assignment** to a grid, with optional regression-based or remainder-based approaches to match observed traffic.
- **In-line** or **hybrid dispersion** calculations for NO2, using customized neighborhood diffusion (via `CellAutDisp`).
- **Scenario-based** interventions (e.g., no-emission zones, 15-minute city, increased parking fees).
- **Parallelization** with Python’s `multiprocessing` to handle large agent sets efficiently.

---

## 3. Installation & Dependencies

### A) Python Version

We recommend **Python 3.8** or higher.  

### B) Required Libraries

Below is a list of main libraries used. You can install them via `pip install <library_name>` or create a conda environment if preferred.

1. **[Mesa](https://github.com/projectmesa/mesa)** – for agent-based modeling.
2. **GeoPandas** – for handling geospatial data.
3. **Shapely** – advanced geometry operations.
4. **scikit-learn** – for mode choice and regression tasks.
5. **pandas**, **numpy** – data manipulation and numerical computation.
6. **matplotlib**, **seaborn** (optional) – plotting.
7. **xarray** – reading or writing multi-dimensional arrays (used in pollution grid).
8. **requests**, **polyline** – for HTTP requests to OSRM and decoding polylines.
9. **pyproj** / **Transformer** – coordinate system transformations.
10. **joblib** – possible model serialization.
11. **PMMLForestClassifier** (from `sklearn_pmml_model`) – loading PMML models.

Additionally:
- **multiprocessing**, **concurrent.futures** – parallel processing.
- **cProfile**, **pstats** – optional for profiling code.

### C) Installing & Setting Up an Environment

```bash
# Example using conda:
conda create -n abm_env python=3.9
conda activate abm_env

# Install key packages (example, adapt as needed):
conda install -c conda-forge mesa geopandas shapely xarray scikit-learn pyproj requests
pip install sklearn_pmml_model polyline
# ... etc.

# Then clone or download this repository:
git clone https://github.com/<your-org>/<this-repo>.git
cd <this-repo>
```

---

## 4. Required Data & Folder Structure

The script expects a certain file/folder layout under the variable `path_data`:

```
D:/PhD EXPANSE/Data/Amsterdam/ABMRessources/ABMData/
│
├── Population/
│   ├── Agent_pop_cleanElectricCarOwnership.csv  # for no-emission zone scenarios
│   └── Amsterdam_population_subsetXXXX.csv       # pre-sampled population files
│
├── ActivitySchedules/
│   ├── HETUS2010_Synthpop_schedulesclean_Monday.csv
│   ├── HETUS2010_Synthpop_schedulesclean_Tuesday.csv
│   └── ...
│
├── SpatialData/
│   ├── SpatialExtent.feather
│   ├── Buildings.feather
│   ├── Streets.feather
│   ├── Residences.feather
│   ├── carroads.feather
│   ├── Universities.feather
│   ├── GreenspaceCentroids.feather
│   ├── EnvBehavDeterminants.feather
│   └── ... (other scenario-specific .feather or shapefiles)
│
├── AirPollutionModelData/
│   ├── optimalparams_50m_...json
│   ├── Pred_50mTrV_TrI_noTrA.csv
│   ├── AirPollDeterm_grid_50m.tif
│   ├── moderator_50m.csv
│   ├── StreetLength_50m.csv
│   └── ...
│
├── Weather/
│   ├── monthlyWeather2019TempDiff.csv
│   └── ...
│
├── TrafficRemainder/
│   └── AirPollGrid_HourlyTraffRemainder_XXXX.csv
│
└── ModelRuns/
    ├── StatusQuo/
    │   └── 43500Agents/
    │       ├── AgentExposure/
    │       ├── NO2/
    │       ├── Traffic/
    │       ├── ModalSplit/
    │       └── Tracks/
    ├── PrkPriceInterv/
    └── ... 
```

> **Note**: This structure is illustrated in the script. You may need to adapt `path_data` or directory names if you use different paths.

---

## 5. Usage

1. **Edit** the parameters at the bottom of the script (e.g., `nb_humans`, `modelname`, `TraffStage`, `starting_date`, etc.) to configure your simulation run.
2. **Ensure** OSRM servers are running. The script calls a batch file `start_OSRM_Servers.bat` to start local OSRM instances for `bike`, `car`, and `foot` profiles on ports `5001`, `5000`, and `5002`.  
   - You may need to adapt or create your own `.bat` scripts for your OSRM setup.
3. **Run** the script:
   ```bash
   python your_script_name.py
   ```
4. **Monitor** logs and outputs. Various CSV files, shapefiles, or figures are saved into the `ModelRuns/...` subdirectories.

---

## 6. Script Structure & Main Components

### A) **Agent Class: `Humans`**
- Inherits from `mesa.Agent`.
- Manages:
  - **Demographics** (education, age, car ownership, etc.)
  - **Activity schedules** (from the `HETUS2010_Synthpop_schedulesclean_*.csv`).
  - **Mode choice** via a pre-trained PMML Random Forest classifier.
  - **Travel routing** using OSRM-based polylines, or fallback to Euclidean if OSRM fails.
  - **Exposure** to NO2, separated into indoor/outdoor fractions.

### B) **Model Class: `TransportAirPollutionExposureModel`**
- Inherits from `mesa.Model`.
- Manages:
  - **Agent creation** (the synthetic population).
  - **Global schedule** (hourly steps).
  - **Traffic assignment** on a grid for each hour, merging agent travel to produce aggregated road usage.
  - **Hybrid dispersion model** that takes traffic volumes and calculates updated NO2 in each grid cell.
  - **Logging** of outputs: agent exposures, travel modes, or pollution maps.

### C) **Helper Functions**
- `worker_process` / `hourly_worker_process`: Parallel step functions to update agents.
- `TraffSpatialJoint()`: Spatial join logic that counts how many routes intersect each grid cell.
- `RetrieveRoutes()`, `RetrieveExposure()`, etc.: Helper data collection from sub-sets of agents.

### D) **Scenario Logic & Parameter Flags**
- `modelname`: determines scenario (e.g., *StatusQuo*, *PrkPriceInterv*, *15mCity*, *NoEmissionZone2025*, etc.).
- `TraffStage`: decides how traffic volumes are calculated (e.g., *"Regression"*, *"Remainder"*, *"PredictionNoR2"*, etc.).
- `nb_humans`: total agents simulated.
- `starting_date`: simulation start date.
- `NrDays`, `NrHours`, `NrMonths`: length of simulation.
- More advanced parameters in the bottom block.

---

## 7. Running the Simulation

1. **Check** that all data needed is in the correct folder structure (see [Required Data & Folder Structure](#required-data--folder-structure)).
2. **Adjust** the parameters at the bottom of the script to your preferred simulation configuration.
3. **Start** your OSRM servers (or adapt the script to your environment).
4. **Execute** the Python script. By default, it will:
    - Create subfolders in `ModelRuns/<modelname>/<nb_humans>Agents`.
    - Sample or load a population (depending on `newpop`).
    - Initialize all agents.
    - Step through simulation time (`month -> day -> hour -> every 10 minutes`).
    - Save output files and logs to disk.

---

## 8. Outputs

During (and after) the run, the script writes:

- **Exposure** logs (`AgentExposure/...`): Hourly or sub-hourly NO2 exposures, time spent indoors, etc.
- **Traffic** logs (`Traffic/...`): Grid-level traffic volumes, possibly intermediate regression stats if *Regression* or *Remainder* mode is used.
- **NO2** files (`NO2/...`): Predicted NO2 columns appended to each grid cell in CSV form; can be used for further GIS or analysis.
- **Tracks** logs (`Tracks/...`): Each agent’s travel geometry per hour (for debugging or visualization).
- **ModalSplit** logs (`ModalSplit/...`): Aggregated usage counts of each mode (bike, drive, transit, walk) for each hour.
- **Console Output**: Periodic status about time steps, traffic assignment times, or memory usage.

---

## 9. Customization & Scenarios

- **Scenarios**: The script shows example scenarios:  
  - **`"StatusQuo"`**: Baseline city conditions.  
  - **`"PrkPriceInterv"`**: Example scenario with changed parking price variables.  
  - **`"15mCity"`** or **`"15mCityWithDestination"`**: Where destinations (work, shops) might be forced within 15 minutes from home.  
  - **`"NoEmissionZone2025/2030"`**: Vehicle-based restrictions if the user does not have electric car access.  

- **Traffic Stage** (`TraffStage`):  
  - `"Regression"` – Uses a linear regression approach to match assigned vs. observed traffic.  
  - `"Remainder"` – Stores partial differences in traffic for subsequent hours.  
  - `"PredictionNoR2"` or `"PredictionR2"` – A final forecast step with or without R² logging.

- **Population Size**: Adjust `nb_humans` to control how many agents are simulated (from ~1% to 10% of the real population).

- **Time Horizon**: The simulation time can be changed via `NrDays`, `NrMonths`, and `NrHours`.

---

## 10. Performance & Parallelization

- Uses **`multiprocessing.Pool`** to distribute agent steps and traffic assignment across multiple cores.  
- The variable `n = os.cpu_count() - 4` (by default) reserves some cores for system tasks. Adapt as necessary.  
- Large runs can be **memory-intensive**. You may need ample RAM if you simulate many agents or large geographies.  
- For further optimization, you can **profile** the code with the `profile = True` option (using `cProfile`).

---

## 11. Troubleshooting

1. **OSRM Connection Errors**  
   - Ensure OSRM servers are running at the expected ports (5000, 5001, 5002).  
   - Adjust the server URL in the `Routing()` method if you are running OSRM on another machine or ports.

2. **Missing Data Files**  
   - Verify that all CSV, feather, or raster data are correctly placed.  
   - Adjust `path_data` if needed.

3. **Memory Errors**  
   - Reduce `nb_humans` or the number of steps (days, hours).  
   - Increase system swap or run on a machine with more RAM.

4. **Coordinate Reference System Issues**  
   - Double check that your data are indeed in the same `epsg:28992` projection (or adapt the code if not).

---

## 12. License
This project is licensed under the [GNU General Public License (GPL) version 3](https://www.gnu.org/licenses/gpl-3.0.en.html).

## 13. Contact

For questions, suggestions, or further collaboration, please reach out to:
- **Name**: Tabea Sonnenschein
- **Email**: t.s.sonnenschein@uu.nl
- **Affiliations**: Utrecht University, Institute for Risk Assessment Sciences & Human Geography and Spatial Planning; University Medical Center Utrecht, Julius Center, Exposome and Planetary Health Group.

---

*Thank you for using this ABM script! We hope it proves insightful for urban mobility and air-quality research.*
