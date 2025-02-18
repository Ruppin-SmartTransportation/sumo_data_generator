# Sumo Traffic Simulation Project

## Overview
This project uses SUMO (Simulation of Urban Mobility) to generate and analyze traffic simulation data for ETA prediction. The simulation outputs a graph where nodes represent junctions and vehicles, and edges represent roads connecting them.

## Directory Structure
```
your_project/
│── data/                  # Stores raw and processed simulation data
│   ├── raw/               # Original SUMO-generated data
│   ├── processed/         # Cleaned and formatted data
│── scripts/               # Python scripts for simulation and data extraction
│   ├── generate_network.py  # Create or convert road network
│   ├── generate_routes.py    # Generate vehicle routes
│   ├── run_simulation.py     # Execute SUMO simulation
│   ├── extract_data.py       # Extract traffic data (graph representation)
│   ├── visualize_graph.py    # Visualize traffic graph
│── models/                 # Machine learning models for ETA prediction
│── configs/                # Configuration files for SUMO
│   ├── network.net.xml
│   ├── routes.rou.xml
│   ├── simulation.sumocfg
│── notebooks/              # Jupyter notebooks for exploratory data analysis
│── README.md               # Project documentation
│── requirements.txt        # Required Python dependencies
│── .gitignore              # Ignore unnecessary files (e.g., SUMO logs)
```

## Installation
1. Install SUMO from [official website](https://www.eclipse.org/sumo/).
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Simulation
### Step 1: Generate the Road Network
Run the following command to create a SUMO network from an OSM file:
```sh
python scripts/generate_network.py
```

### Step 2: Generate Traffic Routes
```sh
python scripts/generate_routes.py
```

### Step 3: Run the SUMO Simulation
```sh
python scripts/run_simulation.py
```

### Step 4: Extract Traffic Data
```sh
python scripts/extract_data.py
```

### Step 5: Visualize the Traffic Graph
```sh
python scripts/visualize_graph.py
```

## Author
Guy Tordjman

## License
This project is licensed under the MIT License.
