import traci
import time
import random
from pathlib import Path
from .logger import Logger
from .node_logger import NodesLogger
from .traffic_controller import TrafficController
from .vehicle_controller import VehicleController
from .junction_controller import JunctionController
from .data_generator import DataGenerator

class SimulationRunner:
    """ Main class to run the SUMO simulation with plugins and dynamic vehicle behavior. """

    def __init__(self, delay=0.01, num_of_steps=100, config_file_override=None, tag=""):
        base_dir = Path(__file__).resolve().parent.parent

        # Use override or default config
        config_filename = config_file_override if config_file_override else "holon.sumocfg"
        self.config_file = base_dir / "sumo_config" / config_filename

        # Logs per run
        suffix = f"_{tag}" if tag else ""
        self.log_path = base_dir / "logs" / f"simulation_log{suffix}.log"
        self.nodes_log_path = base_dir / "logs" / f"nodes_log{suffix}.log"
        export_path = base_dir / f"export_data{suffix}"

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize loggers
        self.logger = Logger(log_file_path=str(self.log_path))
        self.nodes_logger = NodesLogger(log_file_path=str(self.nodes_log_path))

        # Close existing SUMO session
        if traci.isLoaded():
            self.logger.log("⚠️ Closing existing SUMO connection before starting a new one.",
                            "WARNING", "yellow", class_name="SimulationRunner", function_name="__init__")
            traci.close()

        # Start SUMO
        try:
            if not self.config_file.exists():
                raise FileNotFoundError(f"SUMO config file not found: {self.config_file}")

            sumo_cmd = ["sumo-gui", "-c", str(self.config_file), "--start"]
            traci.start(sumo_cmd)
            self.logger.log(f"✅ Simulation started successfully with SUMO config: {self.config_file}",
                            "INFO", "green", class_name="SimulationRunner", function_name="__init__", print_to_console=True)
        except Exception as e:
            self.logger.log(f"❌ Failed to start SUMO: {e}", "ERROR", "red",
                            class_name="SimulationRunner", function_name="__init__", print_to_console=True)
            raise e

        # Controllers
        self.traffic_controller = TrafficController(self.logger)
        self.vehicle_controller = VehicleController(self.logger)
        self.junction_controller = JunctionController(self.logger)
        self.data_generator = DataGenerator(self.logger, str(export_path), time_tag=tag)

        # Parameters
        self.delay = delay
        self.num_of_steps = num_of_steps
        self.most_veh = 0
        self.most_veh_step = 0
        self.traffic_phase_duration = 10

    def run_simulation(self):
        """ Main simulation loop """
        try:
            self.junction_controller.subscribe_to_junctions()

            for step in range(self.num_of_steps):
                traci.simulationStep()
                time.sleep(self.delay)

                self.log_nodes(step)
                self.log_vehicle_count(step)
                self.adjust_vehicle_speeds_randomly(step)
                self.vehicle_controller.log_vehicle_info()
                self.vehicle_controller.track_fastest_vehicle(step)

            self.log_summary()

        except Exception as e:
            self.logger.log(f"❌ Critical simulation error: {e}", "ERROR", "red",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)
        finally:
            traci.close()
            self.logger.log("🔚 Simulation finished and closed successfully.", "INFO", "green",
                            class_name="SimulationRunner", function_name="run_simulation", print_to_console=True)
            self.logger.close()
            self.nodes_logger.close()

    def log_vehicle_count(self, step):
        num_vehicles = traci.vehicle.getIDCount()
        self.logger.log(f"🔹 Step {step}: {num_vehicles} vehicles on the road", "INFO",
                        class_name="SimulationRunner", function_name="log_vehicle_count")

        if num_vehicles > self.most_veh:
            self.most_veh = num_vehicles
            self.most_veh_step = step

    def adjust_vehicle_speeds_randomly(self, step):
        if step % 10 != 0:
            return

        vehicles = self.vehicle_controller.get_active_vehicles()
        if vehicles:
            selected = random.choice(vehicles)
            new_speed = random.uniform(5, 25)
            self.vehicle_controller.update_vehicle_speed(selected, new_speed)
            self.logger.log(f"🔀 Randomly adjusted speed of vehicle {selected} to {new_speed:.2f} m/s",
                            "INFO", "blue", class_name="SimulationRunner", function_name="adjust_vehicle_speeds_randomly")

    def log_summary(self):
        fastest_vehicle, speed, step = self.vehicle_controller.get_fastest_vehicle_summary()
        self.logger.log(f"✅ Most vehicles on road: {self.most_veh} at step {self.most_veh_step}", "INFO", "green",
                        class_name="SimulationRunner", function_name="log_summary")
        self.logger.log(f"🚀 Fastest vehicle: {fastest_vehicle} with {speed:.2f} m/s at step {step}", "INFO", "green",
                        class_name="SimulationRunner", function_name="log_summary")

    def get_static_nodes(self):
        return [n for n in self.junction_controller.get_all_junctions() if not n.startswith(":")]

    def get_dynamic_nodes(self):
        return self.vehicle_controller.get_active_vehicles()

    def log_nodes(self, step):
        static_nodes = self.get_static_nodes()
        dynamic_nodes = self.get_dynamic_nodes()

        self.nodes_logger.log(f"\n-------------------------", "INFO",
                              class_name="SimulationRunner", function_name="log_nodes")
        self.nodes_logger.log(f"🔹 Step #{step}", "INFO",
                              class_name="SimulationRunner", function_name="log_nodes")
        self.nodes_logger.log(f"📍 Static Nodes Count (Real Only): {len(static_nodes)}", "INFO",
                              class_name="SimulationRunner", function_name="log_nodes")
        self.nodes_logger.log(f"🚗 Dynamic Nodes Count: {len(dynamic_nodes)}", "INFO",
                              class_name="SimulationRunner", function_name="log_nodes")

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
            self.nodes_logger.log(msg, "INFO",
                                  class_name="SimulationRunner", function_name="log_nodes")

        self.nodes_logger.log(f"Dynamic Nodes: {dynamic_nodes}", "INFO",
                              class_name="SimulationRunner", function_name="log_nodes")

        self.data_generator.export_data(step, static_nodes)