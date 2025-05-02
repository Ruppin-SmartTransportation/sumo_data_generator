import traci

class SimulationGenerator:
    """ Generates a simulation for the SUMO traffic simulation. """
    def __init__(self, logger):
        self.logger = logger

    def generate_vehicles(self, traffic_pattern):
        """ Generates vehicles based on the specified traffic pattern. """
        self.logger.log(f"🚗 Generating vehicles for traffic pattern: {traffic_pattern}", "INFO", "blue",
                        class_name="SimulationGenerator", function_name="generate_vehicles")
        
        if traffic_pattern == "Morning rush hour":
            majority_travel_from = ["A", "A", "A", "C", "C", "B"]
            majority_travel_to   = ["B", "B", "B", "A", "C"]
        elif traffic_pattern == "Noon":
            majority_travel_from = ["B", "B", "A", "C"]
            majority_travel_to   = ["A", "A", "C", "C", "B"]
        elif traffic_pattern == "Afternoon rush hour":
            majority_travel_from = ["B", "B", "B", "A", "C"]
            majority_travel_to   = ["C", "C", "C", "A", "B"]
        elif traffic_pattern == "Evening":
            majority_travel_from = ["A", "A", "C", "C", "B"]
            majority_travel_to   = ["A", "C", "C", "B"]
        elif traffic_pattern == "Night":
            majority_travel_from = ["A", "B", "C"]
            majority_travel_to   = ["A", "B", "C"]

