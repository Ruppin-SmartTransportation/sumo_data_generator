import traci
import time
import csv
import os
import json
from collections import defaultdict
from .logger import Logger
from .node_logger import NodesLogger
from .vehicle_controller import VehicleController
from .junction_controller import JunctionController
from .data_generator import DataGenerator

class SimulationDataReplayer:
    """ Class to replay simulation data from CSV files and inject it into SUMO simulation. """

    def __init__(self, delay=0.01):
        self.logger = Logger(log_file_path="main/simulation_replay_log.log")
        self.nodes_logger = NodesLogger(log_file_path="main/nodes_replay_log.log") 
        
        # Close existing SUMO connection if it's already active
        if traci.isLoaded():
            self.logger.log("⚠️ Closing existing SUMO connection before starting a new one.", "WARNING", "yellow",
                            class_name="SimulationDataReplayer", function_name="__init__")
            traci.close()

        # Start SUMO-GUI with the simulation configuration
        sumo_cmd = ["sumo", "-c", "sumo_config/3x3_replay.sumocfg", "--start"]
        traci.start(sumo_cmd)
        self.logger.log("✅ Simulation started successfully with SUMO!", "INFO", "green",
                        class_name="SimulationDataReplayer", function_name="__init__", print_to_console=True)
        
        # Initialize controllers
        self.vehicle_controller = VehicleController(self.logger)
        self.junction_controller = JunctionController(self.logger)
        
        # Initialize DataGenerator
        self.data_generator = DataGenerator(self.logger, "export_data_from_replay")
        
        # Simulation parameters
        self.delay = delay
        self.data = self.load_data_from_csv()
        
        # Export the data to a JSON file
        json_file_path = "export_data_from_replay/simulation_data_replayed.json"
        with open(json_file_path, 'w') as json_file:
            json.dump(self.data, json_file, indent=4)
        self.logger.log(f"✅ Data exported to JSON file at {json_file_path}", "INFO", "green",
                class_name="SimulationDataReplayer", function_name="__init__", print_to_console=True)
        
        self.num_of_steps = max(self.data['vehicles'].keys()) + 1 if self.data['vehicles'] else 0
        print(f"Number of steps from loaded data: {self.num_of_steps}")

    def load_data_from_csv(self):
        """ Loads vehicle and junction data from CSV files into a dictionary. """
        data = {
            'vehicles': defaultdict(list)
            # 'junctions': defaultdict(list)
        }

        vehicle_csv_path = "export_data/vehicle_data.csv"
        # junction_csv_path = "export_data/junction_data.csv"

        # Load vehicle data
        with open(vehicle_csv_path, mode='r') as vehicle_csv_file:
            reader = csv.DictReader(vehicle_csv_file)
            for row in reader:
                step_number = int(row['Step Number'])
                data['vehicles'][step_number].append({
                    'Vehicle ID': row['Vehicle ID'],
                    'Vehicle Type': row['Vehicle Type'],
                    'X Coordinate': float(row['X Coordinate']),
                    'Y Coordinate': float(row['Y Coordinate']),
                    'Speed': float(row['Speed'])
                })

        # # Load junction data
        # with open(junction_csv_path, mode='r') as junction_csv_file:
        #     reader = csv.DictReader(junction_csv_file)
        #     for row in reader:
        #         step_number = int(row['Step Number'])
        #         data['junctions'][step_number].append({
        #             'Junction ID': row['Junction ID'],
        #             'X Coordinate': float(row['X Coordinate']),
        #             'Y Coordinate': float(row['Y Coordinate']),
        #             'Junction Type': row['Junction Type'],
        #             'Vehicle Count': int(row['Vehicle Count'])
        #         })

        return data

    def run_replay(self):
        """ Runs the simulation replay loop while logging all events. """
        try:
            for step in range(self.num_of_steps):
                print(f"Starting step: {step}")
                
                # Inject data from loaded data
                self.inject_data(step)
                print(f"Data injected for step: {step}")

                # Advance the simulation step
                print(f"Advancing simulation to step: {step}...")
                traci.simulationStep()
                print(f"Simulation advanced to step: {step}")
                time.sleep(self.delay)

                # Log all nodes (junctions and vehicles)
                self.log_nodes(step)

                # Log current step and vehicle count
                num_vehicles = traci.vehicle.getIDCount()
                self.logger.log(f"🔹 Step {step}: {num_vehicles} vehicles on the road", "INFO",
                                class_name="SimulationDataReplayer", function_name="run_replay")

                # Log all vehicle information
                self.vehicle_controller.log_vehicle_info()

            self.logger.log("🔚 Simulation replay finished successfully!", "INFO", "green",
                            class_name="SimulationDataReplayer", function_name="run_replay", print_to_console=True)

        except Exception as e:
            self.logger.log(f"❌ Critical simulation replay error: {e}", "ERROR", "red",
                            class_name="SimulationDataReplayer", function_name="run_replay", print_to_console=True)

        finally:
            traci.close()
            self.logger.log("🔚 Simulation closed successfully!", "INFO", "green",
                            class_name="SimulationDataReplayer", function_name="run_replay", print_to_console=True)
            self.logger.close()
            self.nodes_logger.close()

    def inject_data(self, step_number):
        """ Injects vehicle and junction data from loaded data into the simulation. """
        # Inject vehicle data
        for row in self.data['vehicles'][step_number]:
            vehicle_id = row['Vehicle ID']
            vehicle_type = row['Vehicle Type']
            x_coordinate = row['X Coordinate']
            y_coordinate = row['Y Coordinate']
            speed = row['Speed']
            if vehicle_id not in traci.vehicle.getIDList():
                traci.vehicle.add(vehicle_id, routeID="", typeID=vehicle_type, depart=0)

            traci.vehicle.moveToXY(vehicle_id, "", 0, x_coordinate, y_coordinate, angle=0, keepRoute=2)
            traci.vehicle.setSpeed(vehicle_id, speed)

        # # Inject junction data
        # for row in self.data['junctions'][step_number]:
        #     junction_id = row['Junction ID']
        #     x_coordinate = row['X Coordinate']
        #     y_coordinate = row['Y Coordinate']
        #     junction_type = row['Junction Type']
        #     vehicle_count = row['Vehicle Count']
        #     self.junction_controller.update_junction(junction_id, (x_coordinate, y_coordinate), junction_type, vehicle_count)
        #     print("Debug 2")

    def log_nodes(self, step_number):
        """ Logs both static and dynamic nodes ONLY to nodes_log.log. """
        static_nodes = self.junction_controller.get_all_junctions()
        dynamic_nodes = self.vehicle_controller.get_active_vehicles()

        # filter internal junctions
        filtered_static_nodes = [node for node in static_nodes if not node.startswith(":")]

        # Log the nodes to the nodes log file
        self.nodes_logger.log("-------------------------", "INFO", 
                            class_name="SimulationDataReplayer", function_name="log_nodes")
        self.nodes_logger.log(f"🔹 Step #{step_number}", "INFO",
                            class_name="SimulationDataReplayer", function_name="log_nodes")
        self.nodes_logger.log(f"📍 Static Nodes Count (Real Only): {len(filtered_static_nodes)}", "INFO",
                            class_name="SimulationDataReplayer", function_name="log_nodes")
        self.nodes_logger.log(f"🚗 Dynamic Nodes Count: {len(dynamic_nodes)}", "INFO",
                            class_name="SimulationDataReplayer", function_name="log_nodes")

        # Log detailed information about each junction
        for junction_id in filtered_static_nodes:
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
                            class_name="SimulationDataReplayer", function_name="log_nodes")

        self.nodes_logger.log(f"Dynamic Nodes: {dynamic_nodes}", "INFO",
                            class_name="SimulationDataReplayer", function_name="log_nodes")

        # if step_number % 20 == 0:
        self.data_generator.export_data(step_number, filtered_static_nodes)