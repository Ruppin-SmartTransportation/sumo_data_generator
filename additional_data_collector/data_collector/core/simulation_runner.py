import traci
import time
import random
from .logger import Logger
from .node_logger import NodesLogger
from .traffic_controller import TrafficController
from .vehicle_controller import VehicleController
from .junction_controller import JunctionController
from .data_generator import DataGenerator
from .simulation_generator import SimulationGenerator

class SimulationRunner:
    """ Main class to run the SUMO simulation with plugins and dynamic vehicle behavior. """

    def __init__(self, delay=0.01, num_of_steps=100):
        self.logger = Logger(log_file_path="main/simulation_log.log")
        self.nodes_logger = NodesLogger(log_file_path="main/nodes_log.log") 
         # Simulation parameters
        self.delay = delay
        self.num_of_steps = num_of_steps
        self.simulation_generator = SimulationGenerator(self.logger)
        self.vehicle_controller = VehicleController(self.logger)
    
        self.time_windows = {
            "Morning rush hour": (23400, 43200),
            "Noon": (43200, 57600),
            "Afternoon rush hour": (57600, 68400),
            "Evening": (68400, 82800),
            "Night": ((82800, 86400), (0, 23400))
        }

        self.max_vehicles_per_window = {
            "Morning rush hour": 600,
            "Noon": 350,
            "Afternoon rush hour": 600,
            "Evening": 400,
            "Night": 200
        }

        self.setup_sumo()

        # Initialize controllers
        self.traffic_controller = TrafficController(self.logger)
        self.junction_controller = JunctionController(self.logger)
        # Any appeal to traci should be done from VehicleController 
       
        # Initialize DataGenerator
        self.data_generator = DataGenerator(self.logger, "export_data")

    def setup_sumo(self):
        """ Start SUMO with config. """
        if traci.isLoaded():
            self.logger.log("⚠️ Closing existing SUMO connection.", "WARNING")
            traci.close()

        sumo_cmd = ["sumo", "-c", "sumo_config/my_3x3_simulation.sumocfg", "--start"]
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
                    num_to_generate = self.get_num_vehicles_for_step(
                        current_time, self.time_windows[current_window], self.max_vehicles_per_window[current_window]
                    )
                    self.simulation_generator.generate_vehicles(current_window, num_to_generate)

                # Log all nodes (junctions and vehicles)
                self.log_nodes(step)
                # Log current step and vehicle count
                num_vehicles = traci.vehicle.getIDCount()
                self.logger.log(f"🔹 Step {step}: {num_vehicles} vehicles on the road", "INFO",
                                class_name="SimulationRunner", function_name="run_simulation")
                # Log all vehicle information
                self.vehicle_controller.log_vehicle_info()

                time.sleep(self.delay)

        except Exception as e:
            self.logger.log(f"❌ Critical simulation error: {e}", "ERROR", "red",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)

        finally:
            traci.close()
            self.logger.log("🔚 Simulation finished and closed successfully!", "INFO", "green",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)
            self.logger.close()
            self.nodes_logger.close()

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

    def get_num_vehicles_for_step(self, current_time, window_range, max_vehicles):
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

        num_vehicles = int(max_vehicles * relative_factor * 0.1)  # 10% of max vehicles per 300s window
        return num_vehicles
    
###################################################
    # def run_simulation(self):
    #     """ Runs the simulation loop while logging all events. """
    #     try:
    #         self.junction_controller.subscribe_to_junctions() # register all junctions for vehicle tracking around them

    #         for pattern in self.traffic_patterns:
    #             self.logger.log(f"🔄 Starting simulation with traffic pattern: {pattern}", "INFO", "blue",
    #                             class_name="SimulationRunner", function_name="run_simulation")
    #             self.simulator_generator.generate_vehicles(pattern)


###########################################################################################


        #     for step in range(self.num_of_steps):
        #         traci.simulationStep()
        #         time.sleep(self.delay)

        #         # Log all nodes (junctions and vehicles)
        #         self.log_nodes(step)

        #         # Log current step and vehicle count
        #         num_vehicles = traci.vehicle.getIDCount()
        #         self.logger.log(f"🔹 Step {step}: {num_vehicles} vehicles on the road", "INFO",
        #                         class_name="SimulationRunner", function_name="run_simulation")

        #         # Log all vehicle information
        #         self.vehicle_controller.log_vehicle_info()

        # except Exception as e:
        #     self.logger.log(f"❌ Critical simulation error: {e}", "ERROR", "red",
        #                     class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)

        # finally:
        #     traci.close()
        #     self.logger.log("🔚 Simulation finished and closed successfully!", "INFO", "green",
        #                     class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)
        #     self.logger.close()
        #     self.nodes_logger.close()

    def get_static_nodes(self):
        """ Retrieves all static nodes (junctions) through JunctionController. """
        return self.junction_controller.get_all_junctions()

    def get_dynamic_nodes(self):
        """ Retrieves all dynamic nodes (vehicles) through VehicleController. """
        return self.vehicle_controller.get_active_vehicles()

    def log_nodes(self, step_number):
        """ Logs both static and dynamic nodes ONLY to nodes_log.log. """
        static_nodes = self.get_static_nodes()
        dynamic_nodes = self.get_dynamic_nodes()

        # filter internal junctions
        self.filtered_static_nodes = [node for node in static_nodes if not node.startswith(":")]

        # Log the nodes to the nodes log file
        self.nodes_logger.log("-------------------------", "INFO", 
                            class_name="SimulationRunner", function_name="log_nodes")
        self.nodes_logger.log(f"🔹 Step #{step_number}", "INFO",
                            class_name="SimulationRunner", function_name="log_nodes")
        self.nodes_logger.log(f"📍 Static Nodes Count (Real Only): {len(self.filtered_static_nodes)}", "INFO",
                            class_name="SimulationRunner", function_name="log_nodes")
        self.nodes_logger.log(f"🚗 Dynamic Nodes Count: {len(dynamic_nodes)}", "INFO",
                            class_name="SimulationRunner", function_name="log_nodes")

        
        # Log detailed information about each junction
        for junction_id in self.filtered_static_nodes:
            junction_info = self.junction_controller.get_junction_info(junction_id)
            
            log_message = f"""🔹 Junction {junction_id} 
            📍 Position: {junction_info['Position']}
            🚗 Vehicles in Junction: {junction_info['Vehicles in Junction']}
            🚦 Traffic Light: {junction_info['Traffic Light State']}
            🛣️ Connected Edges: {junction_info['Connected Edges']}
            🔀 Internal Edges: {junction_info['Internal Edges']}
            ➡️ Connected Lanes: {junction_info['Connected Lanes']}
            ⚙️ Internal Lanes: {junction_info['Internal Lanes']}
            """
            self.nodes_logger.log(log_message, "INFO",
                            class_name="SimulationRunner", function_name="log_nodes")

        self.nodes_logger.log(f"Dynamic Nodes: {dynamic_nodes}", "INFO",
                            class_name="SimulationRunner", function_name="log_nodes")


        if step_number % 10 == 0:
            self.data_generator.export_data(step_number, self.filtered_static_nodes)