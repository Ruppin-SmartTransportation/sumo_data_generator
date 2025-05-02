from additional_data_collector.data_collector.core.simulation_runner import SimulationRunner

if __name__ == "__main__":
    delay = 0 # Add delay to slow down the simulation speed for better visualization
    num_of_steps = 100 # Number of simulation steps to run
    simulation = SimulationRunner(delay, num_of_steps)
    simulation.run_simulation()
