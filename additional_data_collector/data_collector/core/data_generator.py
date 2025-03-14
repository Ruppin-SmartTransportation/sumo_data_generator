import os
import traci
import matplotlib.pyplot as plt
import csv
import numpy as np
import networkx as nx

class DataGenerator:
    def __init__(self, logger, export_data_directory):
        self.logger = logger
        self.export_data_directory = export_data_directory
        self.reset_files()

    def reset_files(self):
        """ Resets the export data content and content of the CSV files. """

        # Reset the export data directory
        if os.path.exists(self.export_data_directory):
            for file in os.listdir(self.export_data_directory):
                file_path = os.path.join(self.export_data_directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
        else:
            os.makedirs(self.export_data_directory)

        files_to_reset = [
            self.export_data_directory + "/junction_data.csv",
            self.export_data_directory + "/vehicle_data.csv"
        ]
        for file_path in files_to_reset:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                if "junction_data.csv" in file_path:
                    writer.writerow(['Step Number', 'Junction ID', 'X Coordinate', 'Y Coordinate', 'Junction Type', 'Vehicle Count'])
                elif "vehicle_data.csv" in file_path:
                    writer.writerow(['Step Number', 'Vehicle ID', 'Vehicle Type', 'X Coordinate', 'Y Coordinate', 'Speed'])

    def export_data(self, step_number, filtered_static_nodes):
        # pull all junctions and their outgoing edges
        junction_positions = {junction: traci.junction.getPosition(junction) for junction in filtered_static_nodes}

        # Call the new function to export network graph
        self.export_network_graph(step_number, junction_positions)
        # Call the new function to export data to CSV
        self.export_junction_data_to_csv(step_number, junction_positions)
        # Call the new function to export adjacency matrix
        self.export_junctions_adjacency_matrix(step_number, junction_positions)
        # Call the new function to export vehicle data
        self.export_vehicle_data(step_number)

    def export_network_graph(self, step_number, junction_positions):
        """ Exports the network graph as an image using matplotlib. """
        
        self.logger.log("📡 Exporting network graph...", "INFO", 
                        class_name="DataGenerator", function_name="export_network_graph")

        # create figure
        plt.figure(figsize=(12, 10))

        for junction, position in junction_positions.items():
            plt.scatter(position[0], position[1], s=100, c='red', edgecolors='black', zorder=5)

            # get vehicles near the junction
            vehicles_nearby = traci.junction.getContextSubscriptionResults(junction)
            vehicle_count = len(vehicles_nearby) if vehicles_nearby else 0
            plt.text(position[0], position[1] + 20, f"{junction}\nVehicles: {vehicle_count}", fontsize=12, ha='center', zorder=10, 
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            # get real edges connected to the junction
            incoming_edges = traci.junction.getIncomingEdges(junction)
            outgoing_edges = traci.junction.getOutgoingEdges(junction)
            all_edges = list(set(incoming_edges + outgoing_edges))
            real_edges = [edge for edge in all_edges if not edge.startswith(":")]

            # draw edges and count vehicles on them
            for edge in real_edges:
                outgoing_junction = traci.edge.getToJunction(edge)
                outgoing_position = traci.junction.getPosition(outgoing_junction)
                plt.plot([position[0], outgoing_position[0]], [position[1], outgoing_position[1]], 'gray', zorder=1)

                # get vehicles on the edge
                vehicles_on_edge = traci.edge.getLastStepVehicleNumber(edge)
                mid_x = (position[0] + outgoing_position[0]) / 2
                mid_y = (position[1] + outgoing_position[1]) / 2
                plt.text(mid_x, mid_y - 40, str(vehicles_on_edge), fontsize=10, ha='center', zorder=10, 
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # save and close
        plt.title("SUMO Network Graph")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)

        export_network_dir = self.export_data_directory + "/network_graph"
        # Create the directory only if it does not exist
        if not os.path.exists(export_network_dir):
            os.makedirs(export_network_dir)
        plt.savefig(f"{self.export_data_directory}/network_graph/step_{step_number}.png")
        plt.close()

        self.logger.log("✅ Network graph exported successfully as 'network_graph.png'", "INFO", 
                        class_name="DataGenerator", function_name="export_network_graph")

    def export_junction_data_to_csv(self, step_number, junction_positions):
        """ Exports junction positions, types, and vehicle counts to a CSV file. """
        csv_file_path = self.export_data_directory + "/junction_data.csv"
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as csv_file:  # Change mode to 'a'
            fieldnames = ['Step Number', 'Junction ID', 'X Coordinate', 'Y Coordinate', 'Junction Type', 'Vehicle Count']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for junction, position in junction_positions.items():
                vehicles_nearby = traci.junction.getContextSubscriptionResults(junction)
                vehicle_count = len(vehicles_nearby) if vehicles_nearby else 0
                junction_type = traci.junction.getParameter(junction, "type")  # Use getParameter to get the junction type
                writer.writerow({
                    'Step Number': step_number,
                    'Junction ID': junction,
                    'X Coordinate': position[0],
                    'Y Coordinate': position[1],
                    'Junction Type': junction_type,
                    'Vehicle Count': vehicle_count
                })

        self.logger.log(f"✅ Junction data exported successfully to '{csv_file_path}'", "INFO", 
                        class_name="DataGenerator", function_name="export_to_csv")

    def export_junctions_adjacency_matrix(self, step_number, junction_positions):
        """ Exports the adjacency matrix of the network and draws the graph. """
        junction_ids = list(junction_positions.keys())
        num_junctions = len(junction_ids)
        adjacency_matrix = np.zeros((num_junctions, num_junctions), dtype=int)

        for i, junction in enumerate(junction_ids):
            outgoing_edges = traci.junction.getOutgoingEdges(junction)
            for edge in outgoing_edges:
                outgoing_junction = traci.edge.getToJunction(edge)
                if outgoing_junction in junction_ids:
                    j = junction_ids.index(outgoing_junction)
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1  # Since the graph is undirected

        # Save adjacency matrix to CSV
        csv_file_path = f"{self.export_data_directory}/adjacency_matrix/step_{step_number}.csv"
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([''] + junction_ids)
            for i, row in enumerate(adjacency_matrix):
                writer.writerow([junction_ids[i]] + list(row))

        self.logger.log(f"✅ Adjacency matrix exported successfully to '{csv_file_path}'", "INFO", 
                        class_name="DataGenerator", function_name="export_adjacency_matrix")

        # Create the graph using networkx
        G = nx.Graph()
        for i, junction in enumerate(junction_ids):
            G.add_node(junction, pos=junction_positions[junction])
        for i in range(num_junctions):
            for j in range(i + 1, num_junctions):
                if adjacency_matrix[i, j] == 1:
                    G.add_edge(junction_ids[i], junction_ids[j])

        # Draw the graph
        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(12, 10))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='red', edge_color='gray', font_size=10, font_color='black')
        plt.title("SUMO Network Adjacency Matrix Graph")
        
        graph_file_path = f"{self.export_data_directory}/adjacency_matrix/step_{step_number}.png"
        directory = os.path.dirname(graph_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(graph_file_path)
        plt.close()

        self.logger.log(f"✅ Adjacency matrix graph exported successfully to '{graph_file_path}'", "INFO", 
                        class_name="DataGenerator", function_name="export_adjacency_matrix")

    def export_vehicle_data(self, step_number):
        """ Exports vehicle data to a CSV file. """
        csv_file_path = self.export_data_directory + "/vehicle_data.csv"
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as csv_file:  # Change mode to 'a'
            fieldnames = ['Step Number', 'Vehicle ID', 'Vehicle Type', 'X Coordinate', 'Y Coordinate', 'Speed']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                position = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                writer.writerow({
                    'Step Number': step_number,
                    'Vehicle ID': vehicle_id,
                    'Vehicle Type': vehicle_type,
                    'X Coordinate': f"{position[0]:.3f}",
                    'Y Coordinate': f"{position[1]:.3f}",
                    'Speed': f"{speed:.3f}"
                })

        self.logger.log(f"✅ Vehicle data exported successfully to '{csv_file_path}'", "INFO", 
                        class_name="DataGenerator", function_name="export_vehicle_data")