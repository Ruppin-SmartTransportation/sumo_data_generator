import traci
import time
import csv
import json
from pathlib import Path
from collections import defaultdict
from .logger import Logger
from .node_logger import NodesLogger
from .vehicle_controller import VehicleController
from .junction_controller import JunctionController
from .data_generator import DataGenerator

class SimulationDataReplayer:
    def __init__(self, delay=0.01):
        # Set paths based on current file location
        base_dir = Path(__file__).resolve().parent.parent
        config_file = base_dir / "sumo_config" / "3x3_replay.sumocfg"
        log_path = base_dir / "logs" / "simulation_replay_log.log"
        nodes_log_path = base_dir / "logs" / "nodes_replay_log.log"
        export_path = base_dir / "export_data_from_replay"
        vehicle_csv_path = base_dir / "export_data" / "vehicle_data.csv"
        json_output_path = export_path / "simulation_data_replayed.json"

        export_path.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = Logger(str(log_path))
        self.nodes_logger = NodesLogger(str(nodes_log_path))

        if not config_file.exists():
            raise FileNotFoundError(f"🚫 SUMO config file not found: {config_file}")

        if not vehicle_csv_path.exists():
            raise FileNotFoundError(f"🚫 Vehicle CSV file not found: {vehicle_csv_path}")

        # Start SUMO GUI
        sumo_cmd = ["sumo", "-c", str(config_file), "--start"]
        traci.start(sumo_cmd)
        self.logger.log(f"✅ Simulation started with config: {config_file}", "INFO", "green",
                        class_name="SimulationDataReplayer", function_name="__init__", print_to_console=True)

        self.vehicle_controller = VehicleController(self.logger)
        self.junction_controller = JunctionController(self.logger)
        self.data_generator = DataGenerator(self.logger, str(export_path))
        self.delay = delay
        self.data = self.load_data_from_csv(vehicle_csv_path)

        with open(json_output_path, 'w') as json_file:
            json.dump(self.data, json_file, indent=4)

        self.logger.log(f"✅ Data exported to JSON: {json_output_path}", "INFO", "green",
                        class_name="SimulationDataReplayer", function_name="__init__")

        self.num_of_steps = max(self.data['vehicles'].keys()) + 1 if self.data['vehicles'] else 0

    def load_data_from_csv(self, csv_path):
        data = {'vehicles': defaultdict(list)}
        with open(csv_path, mode='r') as vehicle_csv_file:
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
        return data

    def run_replay(self):
        try:
            for step in range(self.num_of_steps):
                self.inject_data(step)
                traci.simulationStep()
                time.sleep(self.delay)
                self.log_nodes(step)
                self.logger.log(f"🔹 Step {step}: {traci.vehicle.getIDCount()} vehicles", "INFO",
                                class_name="SimulationDataReplayer", function_name="run_replay")
                self.vehicle_controller.log_vehicle_info()
            self.logger.log("🔚 Replay finished", "INFO", "green",
                            class_name="SimulationDataReplayer", function_name="run_replay", print_to_console=True)
        except Exception as e:
            self.logger.log(f"❌ Error: {e}", "ERROR", "red",
                            class_name="SimulationDataReplayer", function_name="run_replay", print_to_console=True)
        finally:
            traci.close()
            self.logger.close()
            self.nodes_logger.close()

    def inject_data(self, step_number):
        for row in self.data['vehicles'][step_number]:
            vid = row['Vehicle ID']
            vtype = row['Vehicle Type']
            x = row['X Coordinate']
            y = row['Y Coordinate']
            speed = row['Speed']
            if vid not in traci.vehicle.getIDList():
                traci.vehicle.add(vid, routeID="", typeID=vtype, depart=0)
            traci.vehicle.moveToXY(vid, "", 0, x, y, angle=0, keepRoute=2)
            traci.vehicle.setSpeed(vid, speed)

    def log_nodes(self, step_number):
        static_nodes = [n for n in self.junction_controller.get_all_junctions() if not n.startswith(":")]
        dynamic_nodes = self.vehicle_controller.get_active_vehicles()

        self.nodes_logger.log("-------------------------", "INFO", class_name="SimulationDataReplayer", function_name="log_nodes")
        self.nodes_logger.log(f"🔹 Step #{step_number}", "INFO", class_name="SimulationDataReplayer", function_name="log_nodes")
        self.nodes_logger.log(f"📍 Static Nodes Count: {len(static_nodes)}", "INFO", class_name="SimulationDataReplayer", function_name="log_nodes")
        self.nodes_logger.log(f"🚗 Dynamic Nodes Count: {len(dynamic_nodes)}", "INFO", class_name="SimulationDataReplayer", function_name="log_nodes")

        for junction_id in static_nodes:
            info = self.junction_controller.get_junction_info(junction_id)
            msg = f"""🔹 Junction {junction_id} 
📍 Position: {info['Position']}
🚗 Vehicles in Junction: {info['Vehicles in Junction']}
🚦 Traffic Light: {info['Traffic Light State']}
🛣️ Connected Edges: {info['Connected Edges']}
🔀 Internal Edges: {info['Internal Edges']}
➡️ Connected Lanes: {info['Connected Lanes']}
⚙️ Internal Lanes: {info['Internal Lanes']}"""
            self.nodes_logger.log(msg, "INFO", class_name="SimulationDataReplayer", function_name="log_nodes")

        self.nodes_logger.log(f"Dynamic Nodes: {dynamic_nodes}", "INFO", class_name="SimulationDataReplayer", function_name="log_nodes")
        self.data_generator.export_data(step_number, static_nodes)
