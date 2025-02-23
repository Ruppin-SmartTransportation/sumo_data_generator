import traci
import sumolib
import random
import networkx as nx
import matplotlib.pyplot as plt


def takeGraphSnapshot(net, step):
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
        try:
            edge_obj = net.getEdge(edge)
        except KeyError:
            continue
           
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
    plt.savefig(f"grid_simulation/snapshots/grid_snap_{step}.png")
    plt.show()

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

if __name__ == "__main__":
    
    # Define paths
    SUMO_BINARY = "sumo-gui"  # Use "sumo" for command-line mode
    CONFIG_FILE = "configs/grid/grid_simulation_config.sumocfg"
    NET_FILE = "configs/grid/grid_network.net.xml"

    # Define args
    NUM_VEHICLES = 10
    NUM_STEPS = 150
    VERBATIM = False
    SNAPSHOT_STEP_INTERVAL = 50

    # Load the SUMO network
    net = sumolib.net.readNet(NET_FILE)
    edges = [edge.getID() for edge in net.getEdges()]

    # Start SUMO
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])

    # Spawn vehicles
    for i in range(NUM_VEHICLES):
        add_vehicle(f"vehicle_{i}")

    # Run simulation
    for step in range(NUM_STEPS):  # Adjust steps as needed
        traci.simulationStep()
        
        # Optional: Print vehicle positions
        if VERBATIM:
            for i in range(NUM_VEHICLES):
                vehicle_id = f"vehicle_{i}"
                if vehicle_id in traci.vehicle.getIDList():
                    pos = traci.vehicle.getPosition(vehicle_id)
                    print(f"{vehicle_id} is at {pos}")
        
        if step % SNAPSHOT_STEP_INTERVAL == 0 :
            takeGraphSnapshot(net, step)

    # Close the simulation
    traci.close()
