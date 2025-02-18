import random
import traci
import sumolib

# Load your network
net = sumolib.net.readNet("your_network.net.xml")

# Start SUMO simulation (replace with your own configuration file)
traci.start(["sumo", "-c", "your_simulation_config.sumocfg"])

# Create 10 random vehicles
for i in range(10):
    # Choose a random edge for the vehicle to start on
    start_edge = random.choice(net.getEdges())
    vehicle_id = f"vehicle_{i}"

    # Add the vehicle with a random route (start and destination edges)
    route = [start_edge.getID(), random.choice(net.getEdges()).getID()]
    traci.vehicle.add(vehicle_id, route[0])

    # Set random speed (optional)
    traci.vehicle.setSpeed(vehicle_id, random.uniform(10, 30))

# Run the simulation for a set amount of time
for step in range(1000):  # Adjust the number of steps as needed
    traci.simulationStep()
    # Move each vehicle to a random destination
    for i in range(10):
        vehicle_id = f"vehicle_{i}"
        if traci.vehicle.getRoadID(vehicle_id) != route[1]:
            traci.vehicle.changeTarget(vehicle_id, random.choice(net.getEdges()).getID())
        # Optional: random speed adjustment
        traci.vehicle.setSpeed(vehicle_id, random.uniform(10, 30))

# Close the simulation
traci.close()
