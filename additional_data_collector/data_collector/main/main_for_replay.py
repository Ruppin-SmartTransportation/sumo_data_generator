import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.simulation_data_replayer import SimulationDataReplayer

if __name__ == "__main__":
    delay = 0 # Add delay to slow down the simulation speed for better visualization
    simulation_replayer = SimulationDataReplayer(delay)
    simulation_replayer.run_replay()
