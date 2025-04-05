import os
import traci
import matplotlib.pyplot as plt
import csv
import numpy as np
import networkx as nx

class DataGenerator:
    def __init__(self, logger, export_data_directory, time_tag="default"):
        self.logger = logger
        self.export_data_directory = export_data_directory
        self.time_tag = time_tag
        self.reset_files()

    def reset_files(self):
        """ Resets the export data content and content of the CSV files. """

        # Create export folder if needed
        os.makedirs(self.export_data_directory, exist_ok=True)

        # Reset vehicle and junction CSV files with time tag
        files_to_reset = [
            f"{self.export_data_directory}/junction_data_{self.time_tag}.csv",
            f"{self.export_data_directory}/vehicle_data_{self.time_tag}.csv"
        ]
        for file_path in files_to_reset:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                if "junction_data" in file_path:
                    writer.writerow(['Step Number', 'Junction ID', 'X Coordinate', 'Y Coordinate', 'Junction Type', 'Vehicle Count'])
                elif "vehicle_data" in file_path:
                    writer.writerow(['Step Number', 'Vehicle ID', 'Vehicle Type', 'X Coordinate', 'Y Coordinate', 'Speed'])

    def export_data(self, step_number, filtered_static_nodes):
        junction_positions = {junction: traci.junction.getPosition(junction) for junction in filtered_static_nodes}
        self.export_junction_data_to_csv(step_number, junction_positions)
        self.export_vehicle_data_to_csv(step_number)

        if step_number % 10 == 0:
            self.export_network_graph(step_number, junction_positions)
            self.export_junctions_adjacency_matrix(step_number, junction_positions)

    def export_network_graph(self, step_number, junction_positions):
        self.logger.log("📡 Exporting network graph...", "INFO",
                        class_name="DataGenerator", function_name="export_network_graph")

        plt.figure(figsize=(12, 10))

        for junction, position in junction_positions.items():
            plt.scatter(position[0], position[1], s=100, c='red', edgecolors='black', zorder=5)
            vehicles_nearby = traci.junction.getContextSubscriptionResults(junction)
            vehicle_count = len(vehicles_nearby) if vehicles_nearby else 0
            plt.text(position[0], position[1] + 20, f"{junction}\nVehicles: {vehicle_count}", fontsize=12, ha='center', zorder=10,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            incoming_edges = traci.junction.getIncomingEdges(junction)
            outgoing_edges = traci.junction.getOutgoingEdges(junction)
            all_edges = list(set(incoming_edges + outgoing_edges))
            real_edges = [edge for edge in all_edges if not edge.startswith(":")]

            for edge in real_edges:
                outgoing_junction = traci.edge.getToJunction(edge)
                outgoing_position = traci.junction.getPosition(outgoing_junction)
                plt.plot([position[0], outgoing_position[0]], [position[1], outgoing_position[1]], 'gray', zorder=1)
                vehicles_on_edge = traci.edge.getLastStepVehicleNumber(edge)
                mid_x = (position[0] + outgoing_position[0]) / 2
                mid_y = (position[1] + outgoing_position[1]) / 2
                plt.text(mid_x, mid_y - 40, str(vehicles_on_edge), fontsize=10, ha='center', zorder=10,
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        plt.title("SUMO Network Graph")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)

        export_network_dir = f"{self.export_data_directory}/network_graph_{self.time_tag}"
        os.makedirs(export_network_dir, exist_ok=True)
        plt.savefig(f"{export_network_dir}/step_{step_number}.png")
        plt.close()

        self.logger.log("✅ Network graph exported successfully.", "INFO",
                        class_name="DataGenerator", function_name="export_network_graph")

    def export_junction_data_to_csv(self, step_number, junction_positions):
        csv_file_path = f"{self.export_data_directory}/junction_data_{self.time_tag}.csv"

        with open(csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['Step Number', 'Junction ID', 'X Coordinate', 'Y Coordinate', 'Junction Type', 'Vehicle Count']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            for junction, position in junction_positions.items():
                vehicles_nearby = traci.junction.getContextSubscriptionResults(junction)
                vehicle_count = len(vehicles_nearby) if vehicles_nearby else 0
                junction_type = traci.junction.getParameter(junction, "type")
                writer.writerow({
                    'Step Number': step_number,
                    'Junction ID': junction,
                    'X Coordinate': position[0],
                    'Y Coordinate': position[1],
                    'Junction Type': junction_type,
                    'Vehicle Count': vehicle_count
                })

        self.logger.log(f"✅ Junction data exported to '{csv_file_path}'", "INFO",
                        class_name="DataGenerator", function_name="export_junction_data_to_csv")

    def export_junctions_adjacency_matrix(self, step_number, junction_positions):
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
                    adjacency_matrix[j, i] = 1

        dir_path = f"{self.export_data_directory}/adjacency_matrix_{self.time_tag}"
        os.makedirs(dir_path, exist_ok=True)
        csv_file_path = f"{dir_path}/step_{step_number}.csv"

        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([''] + junction_ids)
            for i, row in enumerate(adjacency_matrix):
                writer.writerow([junction_ids[i]] + list(row))

        self.logger.log(f"✅ Adjacency matrix exported to '{csv_file_path}'", "INFO",
                        class_name="DataGenerator", function_name="export_junctions_adjacency_matrix")

        G = nx.Graph()
        for i, junction in enumerate(junction_ids):
            G.add_node(junction, pos=junction_positions[junction])
        for i in range(num_junctions):
            for j in range(i + 1, num_junctions):
                if adjacency_matrix[i, j] == 1:
                    G.add_edge(junction_ids[i], junction_ids[j])

        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(12, 10))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='red', edge_color='gray')
        plt.title("SUMO Network Adjacency Matrix Graph")
        plt.savefig(f"{dir_path}/step_{step_number}.png")
        plt.close()

        self.logger.log(f"✅ Adjacency matrix graph saved to '{dir_path}'", "INFO",
                        class_name="DataGenerator", function_name="export_junctions_adjacency_matrix")

    def export_vehicle_data_to_csv(self, step_number):
        csv_file_path = f"{self.export_data_directory}/vehicle_data_{self.time_tag}.csv"

        with open(csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['Step Number', 'Vehicle ID', 'Vehicle Type', 'X Coordinate', 'Y Coordinate', 'Speed']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

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

        self.logger.log(f"✅ Vehicle data exported to '{csv_file_path}'", "INFO",
                        class_name="DataGenerator", function_name="export_vehicle_data_to_csv")
