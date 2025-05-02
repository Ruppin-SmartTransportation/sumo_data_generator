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
        pattern = self.traffic_patterns[traffic_pattern]

        for i in range(num_vehicles):
            origin_zone = random.choice(pattern["from"])
            destination_zone = random.choice(pattern["to"])

            origin_edge = random.choice(self.zone_edges[origin_zone])
            destination_edge = random.choice(self.zone_edges[destination_zone])

            veh_id = f"veh_{traci.simulation.getTime()}_{i}"
            traci.vehicle.add(vehID=veh_id, routeID="", typeID="car", depart=None)
            traci.vehicle.moveTo(veh_id, origin_edge, 0.0)
            traci.vehicle.setRoute(veh_id, [origin_edge, destination_edge])

        self.logger.log(f"🚗 {num_vehicles} vehicles generated for {traffic_pattern}", "INFO")