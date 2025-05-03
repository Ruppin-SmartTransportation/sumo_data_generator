from core.simulation_data_replayer import SimulationDataReplayer

if __name__ == "__main__":
    delay = 0 # Add delay to slow down the simulation speed for better visualization
    simulation_replayer = SimulationDataReplayer(delay)
    simulation_replayer.run_replay()
