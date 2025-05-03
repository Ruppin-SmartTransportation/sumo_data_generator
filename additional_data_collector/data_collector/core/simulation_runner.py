import traci
import time
import random
from .logger import Logger
# from .node_logger import NodesLogger
# from .traffic_controller import TrafficController
# from .vehicle_controller import VehicleController
# from .junction_controller import JunctionController
# from .data_generator import DataGenerator
from .simulation_generator import SimulationGenerator

class SimulationRunner:
    """ Main class to run the SUMO simulation with plugins and dynamic vehicle behavior. """

    def __init__(self, delay=0, num_of_steps=100):
        self.logger = Logger(log_file_path="main/simulation_log.log")
        self.delay = delay
        self.num_of_steps = num_of_steps
        self.simulation_generator = SimulationGenerator(self.logger)
    
        self.time_windows = {
            "Morning rush hour": (23400, 43200),
            "Noon": (43200, 57600),
            "Afternoon rush hour": (57600, 68400),
            "Evening": (68400, 82800),
            "Night": ((82800, 86400), (0, 23400))
        }

        self.max_vehicles_per_window = {
            "Morning rush hour": 400,
            "Noon": 200,
            "Afternoon rush hour": 400,
            "Evening": 200,
            "Night": 100
        }

        self.setup_sumo()

    def setup_sumo(self):
        """ Start SUMO with config. """
        if traci.isLoaded():
            self.logger.log("⚠️ Closing existing SUMO connection.", "WARNING")
            traci.close()

        # sumo_cmd = ["sumo-gui", "-c", "sumo_config/simulation.sumocfg", "--start"]
        sumo_cmd = ["sumo", "-c", "sumo_config/simulation.sumocfg", "--start"]
        traci.start(sumo_cmd)
        self.logger.log("✅ Simulation started successfully with SUMO!", "INFO", "green",
                        class_name="SimulationRunner", function_name="__init__", print_to_console=True)

    def run_simulation(self):
        """ Runs the simulation with dynamic vehicle generation. """
        try:
            for step in range(self.num_of_steps):
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                current_window = self.get_current_time_window(current_time)

                if self.should_generate_vehicles_now(current_time):
                    self.logger.log(f"🚀 Should spawning vehicles at time {current_time}", "INFO")
                    num_to_generate = self.get_num_vehicles_to_generate_for_step(
                        current_time, self.time_windows[current_window], self.max_vehicles_per_window[current_window]
                    )
                    self.simulation_generator.generate_vehicles(current_window, num_to_generate)
                
                self.get_num_vehicles_and_max(step)

        except Exception as e:
            self.logger.log(f"❌ Critical simulation error: {e}", "ERROR", "red",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)

        finally:
            traci.close()
            self.logger.log(f"🚗 Most vehicles observed: {self.max_vehicles_info['count']} at step {self.max_vehicles_info['step']}", "INFO", "blue",
                             class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)
            self.logger.log("🔚 Simulation finished and closed successfully!", "INFO", "green",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)
            self.logger.close()

    def get_num_vehicles_and_max(self, step):
        """ Get the number of vehicles and max vehicles at a given step. """
        num_vehicles = traci.vehicle.getIDCount()
        self.logger.log(f"🔹 Step {step}: {num_vehicles} vehicles on the road", "INFO",
                        class_name="SimulationRunner", function_name="run_simulation")
        # Track the step with the most vehicles
        if not hasattr(self, 'max_vehicles_info'):
            self.max_vehicles_info = {"step": step, "count": num_vehicles}
        elif num_vehicles > self.max_vehicles_info["count"]:
            self.max_vehicles_info = {"step": step, "count": num_vehicles}
                    
    def get_current_time_window(self, current_time):
        """ Return current time window name based on time in day. """
        time_in_day = current_time % 86400
        for window, ranges in self.time_windows.items():
            if isinstance(ranges, tuple) and isinstance(ranges[0], tuple):
                for r in ranges:
                    if r[0] <= time_in_day < r[1]:
                        return window
            else:
                if ranges[0] <= time_in_day < ranges[1]:
                    return window
        return "Night"

    def should_generate_vehicles_now(self, current_time):
        """ Decide if it's time to generate vehicles (every 300s for example). """
        return int(current_time) % 300 == 0

    def get_num_vehicles_to_generate_for_step(self, current_time, window_range, max_vehicles):
        """ Calculate vehicle count for current step based on distance from window mid-point. """
        time_in_day = current_time % 86400

        if isinstance(window_range, tuple) and isinstance(window_range[0], tuple):
            # If it's 'Night' with two ranges, pick the one we're in
            for r in window_range:
                if r[0] <= time_in_day < r[1]:
                    window_range = r
                    break

        mid_point = (window_range[0] + window_range[1]) / 2
        distance_from_mid = abs(time_in_day - mid_point)
        relative_factor = 1 - (distance_from_mid / ((window_range[1] - window_range[0]) / 2))
        relative_factor = max(0, relative_factor)

        # minimum relative factors for each time window
        min_relative_factor = {
            "Morning rush hour": 0.4,
            "Noon": 0.3,
            "Afternoon rush hour": 0.4,
            "Evening": 0.3,
            "Night": 0.2
        }
        current_window = self.get_current_time_window(current_time)
        relative_factor = max(min_relative_factor[current_window], relative_factor)

        num_vehicles = int(max_vehicles * relative_factor * 0.1)  # 10% of max vehicles per 300s window

        self.logger.log(f" Should generate {num_vehicles} vehicles for {current_window} at time {current_time}", "INFO",
                        class_name="SimulationRunner", function_name="get_num_vehicles_to_generate_for_step")
        
        return num_vehicles
