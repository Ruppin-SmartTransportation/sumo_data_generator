import sumolib
import traci
import networkx as nx
import matplotlib.pyplot as plt
import random

# Load SUMO network
net = sumolib.net.readNet("configs/grid_network.net.xml")

# Start TraCI and connect to SUMO
traci.start(["sumo", "-c", "configs/grid_simulation_config.sumocfg", "--step-length", "1"])

# Generate 10 random vehicles
for i in range(10):
    route_id = f"route_{i}"
    vehicle_id = f"V{i}"
    
    # Pick a random edge from the network
    random_edge = random.choice(net.getEdges()).getID()

    # Add vehicle and assign route
    traci.route.add(route_id, [random_edge])  # Create a route with one road
    traci.vehicle.add(vehicle_id, route_id)

# Step the simulation to let vehicles appear
for _ in range(10):
    traci.simulationStep()

# Create a directed graph
G = nx.DiGraph()

# Add junctions as nodes
junction_nodes = {}
for node in net.getNodes():
    G.add_node(node.getID(), pos=(node.getCoord()[0], node.getCoord()[1]))
    junction_nodes[node.getID()] = node.getCoord()

# Add road edges (between junctions)
road_edges = []
for edge in net.getEdges():
    start = edge.getFromNode().getID()
    end = edge.getToNode().getID()
    G.add_edge(start, end)
    road_edges.append((start, end))

# Extract vehicle positions and update edges
vehicles_on_edges = {}

# Get vehicle positions
vehicles = traci.vehicle.getIDList()
vehicle_nodes = {}
for vehicle in vehicles:
    x, y = traci.vehicle.getPosition(vehicle)  # Get vehicle coordinates
    edge = traci.vehicle.getRoadID(vehicle)  # Get the edge (road) where the vehicle is

    # Store vehicle info under the road it belongs to
    if edge not in vehicles_on_edges:
        vehicles_on_edges[edge] = []
    vehicles_on_edges[edge].append((vehicle, x, y))

# Sort vehicles by position on each edge to maintain order
vehicle_edges = []
for edge in vehicles_on_edges:
    vehicles_on_edges[edge].sort(key=lambda v: v[1])  # Sort by x-coordinate

    # Get the start and end junctions for this edge
    edge_obj = net.getEdge(edge)
    start_junction = edge_obj.getFromNode().getID()
    end_junction = edge_obj.getToNode().getID()

    prev_entity = start_junction  # Start connecting from the junction

    for vehicle, x, y in vehicles_on_edges[edge]:
        G.add_node(vehicle, pos=(x, y))  # Add vehicle node
        vehicle_nodes[vehicle] = (x, y)
        G.add_edge(prev_entity, vehicle)  # Connect previous entity (junction or vehicle) to this vehicle
        vehicle_edges.append((prev_entity, vehicle))
        prev_entity = vehicle  # Update previous entity

    # Finally, connect the last vehicle to the end junction
    G.add_edge(prev_entity, end_junction)
    vehicle_edges.append((prev_entity, end_junction))

# Extract node positions
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph
plt.figure(figsize=(10, 8))

# Draw road edges (junction-to-junction) in thick grey
nx.draw_networkx_edges(G, pos, edgelist=road_edges, edge_color='grey', width=2)

# Draw vehicle edges in blue
nx.draw_networkx_edges(G, pos, edgelist=vehicle_edges, edge_color='blue', width=1)

# Draw junctions (larger nodes)
nx.draw_networkx_nodes(G, pos, nodelist=list(junction_nodes.keys()), node_size=300, node_color='red', label="Junctions")

# Draw vehicles (smaller nodes)
nx.draw_networkx_nodes(G, pos, nodelist=list(vehicle_nodes.keys()), node_size=150, node_color='blue', label="Vehicles")

# Add labels
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("SUMO Network with Vehicles Positioned on Roads")
plt.legend()
plt.savefig("sumo_network.png")
plt.show()

# Close TraCI connection
traci.close()
