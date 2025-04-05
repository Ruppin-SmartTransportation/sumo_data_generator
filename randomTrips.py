import random
import traci
import sumolib
import os

# Convert to absolute paths
net_path = os.path.abspath("additional_data_collector/data_collector/sumo_config/holon.net.xml")

cfg_path = os.path.abspath("additional_data_collector/data_collector/sumo_config/holon.sumocfg")


# Convert Windows path to URL format
net_url = f"file:///{net_path.replace(os.sep, '/')}"

# Load network
net = sumolib.net.readNet(net_url)

# Start SUMO
traci.start(["sumo", "-c", cfg_path])



# Number of vehicles
num_vehicles = 10

# Keep track of targets for each vehicle
vehicle_targets = {}

# Get only edges that are not internal or walking
valid_edges = [e for e in net.getEdges() if not e.getID().startswith(":") and e.allows("passenger")]

# Create vehicles
for i in range(num_vehicles):
    vehicle_id = f"vehicle_{i}"
    start_edge = random.choice(valid_edges)
    end_edge = random.choice(valid_edges)
    traci.route.add(f"route_{i}", [start_edge.getID(), end_edge.getID()])
    traci.vehicle.add(vehicle_id, f"route_{i}")
    traci.vehicle.setSpeed(vehicle_id, random.uniform(10, 30))
    vehicle_targets[vehicle_id] = end_edge.getID()

# Run the simulation
for step in range(1000):
    traci.simulationStep()
    for vehicle_id in vehicle_targets.keys():
        current_edge = traci.vehicle.getRoadID(vehicle_id)
        target_edge = vehicle_targets[vehicle_id]
        if current_edge != target_edge:
            new_target = random.choice(valid_edges).getID()
            traci.vehicle.changeTarget(vehicle_id, new_target)
            vehicle_targets[vehicle_id] = new_target
        traci.vehicle.setSpeed(vehicle_id, random.uniform(10, 30))

# Close simulation
traci.close()
