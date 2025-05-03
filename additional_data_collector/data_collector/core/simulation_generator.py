import random
import traci

class SimulationGenerator:
    """ Generates vehicles dynamically for the SUMO traffic simulation. """

    def __init__(self, logger, edges_path="sumo_config"):
        self.logger = logger
        self.zone_edges = {}
        self.load_zone_edges(edges_path)
        self.traffic_patterns = self.define_traffic_patterns()

    def load_zone_edges(self, edges_path):
        """ Load edge lists for each zone from text files. """
        zones = ['A', 'B', 'C']
        for zone in zones:
            with open(f"{edges_path}/zone{zone}_edges.txt", 'r') as f:
                edges = [line.strip() for line in f.readlines()]
                self.zone_edges[zone] = edges
        self.logger.log(f"✅ Zone edges loaded: { {k: len(v) for k,v in self.zone_edges.items()} }", "INFO")

    def define_traffic_patterns(self):
        """ Define traffic patterns for each time window. """
        return {
            "Morning rush hour":    {"from": ["A", "A", "A", "C", "C", "B"], "to": ["B", "B", "B", "A", "C"]},
            "Noon":                 {"from": ["B", "B", "A", "C"],           "to": ["A", "A", "C", "C", "B"]},
            "Afternoon rush hour":  {"from": ["B", "B", "B", "A", "C"],      "to": ["C", "C", "C", "A", "B"]},
            "Evening":              {"from": ["A", "A", "C", "C", "B"],      "to": ["A", "C", "C", "B"]},
            "Night":                {"from": ["A", "B", "C"],                "to": ["A", "B", "C"]}
        }

    def generate_vehicles(self, traffic_pattern, num_vehicles):
        """ Generate vehicles dynamically according to the pattern and number requested. """
        try:            
            pattern = self.traffic_patterns[traffic_pattern]
            actual_generated = 0


            for i in range(num_vehicles):
                origin_zone = random.choice(pattern["from"])
                destination_zone = random.choice(pattern["to"])

                origin_edge = random.choice(self.zone_edges[origin_zone])
                origin_lane = origin_edge + "_0"
                destination_edge = random.choice(self.zone_edges[destination_zone])
                route = traci.simulation.findRoute(origin_edge, destination_edge).edges

                veh_id = f"veh_{traffic_pattern.replace(' ', '_')}_{traci.simulation.getTime()}_{i}"
                traci.vehicle.add(vehID=veh_id, routeID="", depart=None)
                traci.vehicle.setColor(veh_id, (255, 0, 0, 255))  # Set vehicle color to red (RGBA)
                if len(route) > 1:
                    traci.vehicle.setRoute(veh_id, route)
                    traci.vehicle.moveTo(veh_id, origin_lane, 0.0)
                    actual_generated += 1
                else:
                    self.logger.log(f"⚠️ No valid route between {origin_edge} and {destination_edge}", "WARNING",
                                    class_name="SimulationGenerator", function_name="generate_vehicles")
        except Exception as e:
            self.logger.log(f"❌ Error generating vehicles: {str(e)}", "ERROR", "red",
                            class_name="SimulationGenerator", function_name="generate_vehicles")

        self.logger.log(f"🚗 {actual_generated} vehicles generated for {traffic_pattern}", "INFO",
                        class_name="SimulationGenerator", function_name="generate_vehicles")