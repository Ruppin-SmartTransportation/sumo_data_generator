import traci
import time
import random
from .logger import Logger
from .node_logger import NodesLogger
from .traffic_controller import TrafficController
from .vehicle_controller import VehicleController
from .junction_controller import JunctionController

class SimulationRunner:
    """ Main class to run the SUMO simulation with plugins and dynamic vehicle behavior. """

    def __init__(self, delay=0.01, num_of_steps=100):
        self.logger = Logger(log_file_path="main/simulation_log.log")
        self.nodes_logger = NodesLogger(log_file_path="main/nodes_log.log") 
    
        # Close existing SUMO connection if it's already active
        if traci.isLoaded():
            self.logger.log("⚠️ Closing existing SUMO connection before starting a new one.", "WARNING", "yellow",
                            class_name="SimulationRunner", function_name="__init__")
            traci.close()

        # Start SUMO-GUI with the simulation configuration
        sumo_cmd = ["sumo", "-c", "sumo_config/my_3x3_simulation.sumocfg", "--start"]
        traci.start(sumo_cmd)
        self.logger.log("✅ Simulation started successfully with SUMO!", "INFO", "green",
                        class_name="SimulationRunner", function_name="__init__", print_to_console=True)
        
        # Initialize controllers
        self.traffic_controller = TrafficController(self.logger)
        self.vehicle_controller = VehicleController(self.logger)
        self.junction_controller = JunctionController(self.logger)
        # Any appeal to traci should be done from VehicleController 
       
        # Simulation parameters
        self.delay = delay
        self.num_of_steps = num_of_steps
        self.most_veh = 0
        self.most_veh_step = 0
        self.traffic_phase_duration = 10

    def run_simulation(self):
        """ Runs the simulation loop while logging all events. """
        try:
            self.junction_controller.subscribe_to_junctions() # register all junctions for vehicle tracking around them

            for step in range(self.num_of_steps):
                traci.simulationStep()
                time.sleep(self.delay)

                # Log all nodes (junctions and vehicles)
                self.log_nodes(step)

                # Log current step and vehicle count
                num_vehicles = traci.vehicle.getIDCount()
                self.logger.log(f"🔹 Step {step}: {num_vehicles} vehicles on the road", "INFO",
                                class_name="SimulationRunner", function_name="run_simulation")

                # Track the maximum vehicle count
                if num_vehicles > self.most_veh:
                    self.most_veh = num_vehicles
                    self.most_veh_step = step

                # Adjust the speed of one random vehicle every 10 steps
                if step % 10 == 0:
                    self.adjust_vehicle_speeds_randomly()

                # # Update traffic lights
                # self.traffic_controller.update_traffic_light(step, self.traffic_phase_duration)

                # Log all vehicle information
                self.vehicle_controller.log_vehicle_info()

                # Track the fastest vehicle each step
                self.vehicle_controller.track_fastest_vehicle(step)

            # Log the summary of the fastest vehicle
            fastest_vehicle, fastest_speed, fastest_step = self.vehicle_controller.get_fastest_vehicle_summary()
            self.logger.log(f"\n✅ Most vehicles on the road: {self.most_veh}, at step {self.most_veh_step}", "INFO", "green",
                            class_name="SimulationRunner", function_name="run_simulation")
            self.logger.log(f"🚀 Fastest vehicle: {fastest_vehicle} with speed {fastest_speed:.2f} m/s at step {fastest_step}", "INFO", "green",
                            class_name="SimulationRunner", function_name="run_simulation")

        except Exception as e:
            self.logger.log(f"❌ Critical simulation error: {e}", "ERROR", "red",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)

        finally:
            traci.close()
            self.logger.log("🔚 Simulation finished and closed successfully!", "INFO", "green",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)
            self.logger.close()
            self.nodes_logger.close()

    def adjust_vehicle_speeds_randomly(self):
        """ Randomly adjust the speed of one random active vehicle. """
        vehicles = self.vehicle_controller.get_active_vehicles()
        if vehicles:
            selected_vehicle = random.choice(vehicles)
            random_speed = random.uniform(5, 25)  # Speed between 5 and 25 m/s
            self.vehicle_controller.update_vehicle_speed(selected_vehicle, random_speed)
            self.logger.log(f"🔀 Randomly adjusted speed of vehicle {selected_vehicle} to {random_speed:.2f} m/s",
                            "INFO", "blue", class_name="SimulationRunner", function_name="adjust_vehicle_speeds_randomly")
    
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

        # סינון צמתים פנימיים
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

        

        # הצגת מידע מפורט על כל צומת
        for junction_id in self.filtered_static_nodes:
            junction_info = self.junction_controller.get_junction_info(junction_id)
            
            # לוג מסודר ומפורמט
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

        # export_graph_flag = True if step_number % 5 == 0 else False

        if step_number % 10 == 0:
            self.junction_controller.export_network_graph(step_number, self.filtered_static_nodes)