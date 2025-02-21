import traci
import sumolib
import random

# Define paths
SUMO_BINARY = "sumo-gui"  # Use "sumo" for command-line mode
CONFIG_FILE = "configs/grid_simulation_config.sumocfg"

# Load the SUMO network
net = sumolib.net.readNet("configs/grid_network.net.xml")
edges = [edge.getID() for edge in net.getEdges()]

# Start SUMO
traci.start([SUMO_BINARY, "-c", CONFIG_FILE])

# Function to add a vehicle
def add_vehicle(vehicle_id):
    start_edge = random.choice(edges)
    end_edge = random.choice(edges)
    
    while start_edge == end_edge:  # Ensure start and end are different
        end_edge = random.choice(edges)

    route_id = f"route_{vehicle_id}"
    
    # Define a new route
    traci.route.add(route_id, [start_edge, end_edge])
    
    # Add the vehicle
    traci.vehicle.add(vehicle_id, route_id)
    
    # Set a random speed mode
    traci.vehicle.setSpeedMode(vehicle_id, 31)  # Default SUMO speed mode
    traci.vehicle.setSpeed(vehicle_id, random.uniform(5, 15))  # Random speed

# Spawn 10 vehicles
for i in range(1):
    add_vehicle(f"vehicle_{i}")

# Run simulation
for step in range(500):  # Adjust steps as needed
    traci.simulationStep()
    
    # Optional: Print vehicle positions
    for i in range(10):
        vehicle_id = f"vehicle_{i}"
        if vehicle_id in traci.vehicle.getIDList():
            pos = traci.vehicle.getPosition(vehicle_id)
            print(f"{vehicle_id} is at {pos}")

# Close the simulation
traci.close()
