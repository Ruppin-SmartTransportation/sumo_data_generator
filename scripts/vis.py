import matplotlib
# matplotlib.use("TkAgg")  # Use TkAgg backend (requires Tkinter)
import sumolib
import networkx as nx
import matplotlib.pyplot as plt

# Load SUMO network
net = sumolib.net.readNet("configs/grid_network.net.xml")

# Create a directed graph
G = nx.DiGraph()

# Add junctions as nodes
for node in net.getNodes():
    G.add_node(node.getID(), pos=(node.getCoord()[0], node.getCoord()[1]))

# Add edges (roads and lanes)
for edge in net.getEdges():
    from_node = edge.getFromNode().getID()
    to_node = edge.getToNode().getID()
    num_lanes = edge.getLaneNumber()

    # Add multiple edges to represent lanes
    for lane_idx in range(num_lanes):
        lane_id = f"{from_node}-{to_node}-lane{lane_idx}"
        G.add_edge(from_node, to_node, label=f"Lane {lane_idx+1}", weight=num_lanes)

# Extract node positions for visualization
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph
plt.figure(figsize=(10, 8))

# Draw roads with varying thickness based on lane count
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw(G, pos, with_labels=True, node_size=100, node_color='red', edge_color='gray', font_size=8, width=[w*0.5 for w in weights])

plt.title("SUMO Network with Lanes")
plt.savefig("sumo_network.png")
