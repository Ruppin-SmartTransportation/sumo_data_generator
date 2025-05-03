from core.simulation_runner import SimulationRunner

if __name__ == "__main__":
    delay = 0 # Add delay to slow down the simulation speed for better visualization
    num_of_steps = 86400 # 24 hours in seconds for now, 10 days in seconds = 864000
    # num_of_steps = 10000
    simulation = SimulationRunner(delay, num_of_steps)
    simulation.run_simulation()
