import os
import traci
import matplotlib.pyplot as plt
import csv
import numpy as np
import networkx as nx
import xml.etree.ElementTree as ET

class DataGenerator:
    def __init__(self, logger):
        self.logger = logger
        self.reset_files()
        self.grid_net_xml_path = r"C:\Users\Matan\project_SmartTransportationRuppin\sumo_data_generator\additional_data_collector\data_collector\sumo_config\my_3x3_grid.net.xml"
        self.load_junction_types_from_netxml(self.grid_net_xml_path)
        self.export_fixed_road_edges = True
    
    def load_junction_types_from_netxml(self, net_xml_path):
        """
        Parses the .net.xml file to retrieve junction types.
        """
        junction_types = {}
        tree = ET.parse(net_xml_path)
        root = tree.getroot()

        for junction in root.findall("junction"):
            junction_id = junction.get("id")
            junction_type = junction.get("type")
            if junction_id and junction_type:
                junction_types[junction_id] = junction_type

        self.logger.log(f"🔹 Junction types loaded from {net_xml_path}: {junction_types}", "DEBUG"
                        , class_name="DataGenerator", function_name="load_junction_types_from_netxml")

        self.junction_types = junction_types
    
    def reset_files(self):
        """ Resets the export data content and content of the CSV files. """

        # Reset the export data directory
        if os.path.exists("export_data"):
            for file in os.listdir("export_data"):
                file_path = os.path.join("export_data", file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
        else:
            os.makedirs("export_data")

        files_to_reset = [
            "export_data/junction_data.csv",
            "export_data/vehicle_data.csv",
            "export_data/fixed_road_edges_data.csv",
        ]
        for file_path in files_to_reset:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                if "junction_data.csv" in file_path:
                    writer.writerow(['Step Number', 'Junction ID', 'Junction Type', 'X Coordinate', 'Y Coordinate', 'Number of Vehicles', 'Average Speed', 'Congestion Level', 'Traffic Light State'])
                elif "vehicle_data.csv" in file_path:
                    writer.writerow(['Step Number', 'Vehicle ID', 'Vehicle Type', 'X Coordinate', 'Y Coordinate', 'Length Dimention', 'Width Dimention' ,'Speed', 'Acceleration' , 'Route ID', 'Route Edges', 'Lane ID', 'Lane Position', 'Lane Index', 'Changing Lane', 'Left Signal', 'Right Signal', 'Leader ID', 'Leader Distance', 'Driving Status', 'Is Near Exit'])
                elif "fixed_road_edges_data.csv" in file_path:
                    writer.writerow(['Edge ID <> Lane ID', 'Source', 'Destination', 'Length', 'Speed Limit', 'Current Traffic Flow', 'Road Type'])


    def export_data(self, step_number, filtered_static_nodes):
        # pull all junctions and their outgoing edges
        junction_positions = {junction: traci.junction.getPosition(junction) for junction in filtered_static_nodes}

        # Call the new function to export data to CSV
        self.export_junction_data_to_csv(step_number, junction_positions)
        # Call the new function to export vehicle data
        self.export_vehicle_data_to_csv(step_number)

        self.export_fixed_road_edges_data_to_csv()

        # Call the new function to export network graph
        # self.export_network_graph(step_number, junction_positions)
        # Call the new function to export adjacency matrix
        # self.export_junctions_adjacency_matrix(step_number, junction_positions)

    def export_fixed_road_edges_data_to_csv(self):
        """ Exports fixed road edges data to a CSV file. """
        if not self.export_fixed_road_edges:
            return
        self.export_fixed_road_edges = False

        csv_file_path = "export_data/fixed_road_edges_data.csv"
        file_exists = os.path.isfile(csv_file_path)

        tree = ET.parse(self.grid_net_xml_path)
        root = tree.getroot()

        file_exists = os.path.isfile(csv_file_path)
        fieldnames = ['Edge ID <> Lane ID', 'Source', 'Destination', 'Length', 'Speed Limit', 'Current Traffic Flow', 'Road Type']

        with open(csv_file_path, mode='a', newline='') as csv_file:  # Append mode ('a')
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            edges_from_xml = root.findall('edge')
            
            self.logger.log(f"Total edges found in XML: {len(edges_from_xml)}", "INFO",
                            class_name="DataGenerator", function_name="export_fixed_road_edges_data_to_csv")
            
            # Filter out road edges based, and filter out internal edges
            road_edges = [edge for edge in edges_from_xml if edge.get('function') != 'internal']
            self.logger.log(f"Total ***road edges*** found in XML: {len(road_edges)}", "INFO",
                            class_name="DataGenerator", function_name="export_fixed_road_edges_data_to_csv")

            for edge in road_edges:
                edge_id = edge.get('id')
                source = edge.get('from')
                destination = edge.get('to')
                road_type = edge.get('') 
                
                lanes = edge.findall('lane')

                for lane in lanes: 
                    # each edge can have multiple lanes - "Had Nativi / Du Nativi ...."
                    # todo - disccuss what to do with this multi -lanes
                    lane_id = lane.get('id')
                    length = float(lane.get('length', 0))  
                    speed_limit = float(lane.get('speed', 0)) 
                    road_type = "highway" if speed_limit * 3.6 >= 80 else "local road" if speed_limit * 3.6 >= 50 else "residential/street"
                    
                    # Current traffic flow is not available in the XML, so we set it to 0.0
                    current_traffic_flow = 0.0

                    # Write data to CSV
                    writer.writerow({
                        'Edge ID <> Lane ID': f"{edge_id} <> {lane_id}",
                        'Source': source,
                        'Destination': destination,
                        'Length': length,
                        'Speed Limit': speed_limit,
                        'Road Type': road_type,
                        'Current Traffic Flow': current_traffic_flow
                    })

        self.logger.log(f"✅ Fixed road edges data exported successfully to '{csv_file_path}'", "INFO",
                        class_name="DataGenerator", function_name="export_fixed_road_edges_data_to_csv")

    def export_junction_data_to_csv(self, step_number, junction_positions):
        """ Exports junction positions, types, and vehicle counts to a CSV file. """
        csv_file_path = "export_data/junction_data.csv"
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as csv_file:  # Change mode to 'a'
            fieldnames = ['Step Number', 'Junction ID', 'Junction Type', 'X Coordinate', 'Y Coordinate', 'Number of Vehicles', 'Average Speed', 'Congestion Level', 'Traffic Light State']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for junction, position in junction_positions.items():
                
                junction_type = self.junction_types.get(junction, "Unknown")
                vehicles_nearby = traci.junction.getContextSubscriptionResults(junction)
                number_of_vehicles = len(vehicles_nearby) if vehicles_nearby else 0
                average_speed_in_junction = self.calculate_average_speed_at_junction(junction)
                congestion_level = self.calculate_congestion_level_at_junction(junction, average_speed_in_junction)
                traffic_light_state = traci.trafficlight.getRedYellowGreenState(junction) if junction_type == "traffic_light" else "N/A"
                writer.writerow({
                    'Step Number': step_number,
                    'Junction ID': junction,
                    'Junction Type': junction_type,
                    'X Coordinate': position[0],
                    'Y Coordinate': position[1],
                    'Number of Vehicles': number_of_vehicles,
                    'Average Speed': average_speed_in_junction,
                    'Congestion Level': congestion_level,
                    'Traffic Light State': traffic_light_state
                })

        self.logger.log(f"✅ Junction data exported successfully to '{csv_file_path}'", "INFO", 
                        class_name="DataGenerator", function_name="export_to_csv")

    def calculate_congestion_level_at_junction(self, junction_id, average_speed_in_junction):
        """ Calculates the congestion level at a junction based on the number of vehicles. """
        speed_limit = self.get_max_speed_at_junction(junction_id)
        if speed_limit == 0:
            return "Error"
        
        # Compute the speed ratio (percentage of speed limit)
        speed_ratio = (average_speed_in_junction / speed_limit) * 100

        # Determine congestion level based on predefined thresholds
        if speed_ratio >= 70:
            return "Low Congestion"  # Free flow of traffic
        elif 30 <= speed_ratio < 70:
            return "Medium Congestion"  # Noticeable slowdown
        else:
            return "High Congestion"  # Heavy traffic / Traffic jam
    
    def get_max_speed_at_junction(self, junction_id):
        """ 
        Retrieves the maximum speed of edges connected to a junction.
        Uses available methods to determine the speed limit.
        """

        # Getting all edges connected to the junction
        incoming_edges = traci.junction.getIncomingEdges(junction_id)
        outgoing_edges = traci.junction.getOutgoingEdges(junction_id)
        all_edges = set(incoming_edges + outgoing_edges)
        
        # Collecting the maximum speeds available from these edges
        max_speeds = []
        for edge in all_edges:
            try:
                # Try retrieving a relevant speed attribute (e.g., current max speed)
                # If getMaxSpeed is unavailable, use another related method or adjust accordingly
                # speed = float(traci.edge.get(('speed', 0)))  # Ensure this is valid in your setup
                speed = traci.edge.getMaxSpeed(edge)
                max_speeds.append(speed)
            except AttributeError:
                # Handle the case if getMaxSpeed is unavailable
                # todo: Add alternative method to retrieve speed
                self.logger.log(f"⚠️ AttributeError: getMaxSpeed not available for edge {edge}", "WARNING", 
                                class_name="DataGenerator", function_name="get_max_speed_at_junction")
                pass

        # Return the highest speed found (if any) or 0.0 if not found
        return max(max_speeds) if max_speeds else 0.0


    def calculate_average_speed_at_junction(self, junction_id):
        vehicles_nearby = traci.junction.getContextSubscriptionResults(junction_id)

        if not vehicles_nearby:
            return 0.0

        speeds = [traci.vehicle.getSpeed(vehicle_id) for vehicle_id in vehicles_nearby.keys()]
        
        average_speed = sum(speeds) / len(speeds)
        return average_speed

    
    def export_vehicle_data_to_csv(self, step_number):
        """ Exports vehicle data to a CSV file. """
        csv_file_path = "export_data/vehicle_data.csv"
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as csv_file:  # Change mode to 'a'
            fieldnames = ['Step Number', 'Vehicle ID', 'Vehicle Type', 'X Coordinate', 'Y Coordinate', 'Length Dimention', 'Width Dimention' ,'Speed', 'Acceleration', 'Route ID', 'Route Edges', 'Lane ID', 'Lane Position', 'Lane Index', 'Changing Lane', 'Left Signal', 'Right Signal', 'Leader ID', 'Leader Distance', 'Driving Status', 'Is Near Exit']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                position = traci.vehicle.getPosition(vehicle_id)
                length = traci.vehicle.getLength(vehicle_id)
                width = traci.vehicle.getWidth(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                acceleration = traci.vehicle.getAcceleration(vehicle_id)
                route_id = traci.vehicle.getRouteID(vehicle_id)
                route_edges = traci.vehicle.getRoute(vehicle_id)
                
                # Lane information
                lane_id = traci.vehicle.getLaneID(vehicle_id)  # Current lane identifier
                lane_position = traci.vehicle.getLanePosition(vehicle_id)  # Position within the lane
                # Lane change status
                lane_index = traci.vehicle.getLaneIndex(vehicle_id)  # Which lane the vehicle is in - 0, 1, 2, etc. 0 means the leftmost lane
                intended_lane = traci.vehicle.getBestLanes(vehicle_id)  # Whether there is an intention to change lanes
                changing_lane = len(intended_lane) > 0  # Whether the vehicle is in the process of lane change calculation
                # Turn signal status
                signals = traci.vehicle.getSignals(vehicle_id)  # Numeric code representing the turn signals
                left_signal = bool(signals & 1)  # Whether the left turn signal is on
                right_signal = bool(signals & 2)  # Whether the right turn signal is on
                # Distance to the leading vehicle
                leader_info = traci.vehicle.getLeader(vehicle_id)  # ID and distance of the vehicle ahead
                leader_id = leader_info[0] if leader_info else None
                leader_distance = leader_info[1] if leader_info else float('inf')
                # Driving status
                driving_status = "Accelerating" if acceleration > 0 else "Decelerating" if acceleration < 0 else "Cruising"
                # Proximity to lane merges or exits
                edge_id = traci.vehicle.getRoadID(vehicle_id)  # Current road identifier
                is_near_exit = edge_id in route_edges[-2:]  # Whether the vehicle is near the end of its route

                writer.writerow({
                    'Step Number': step_number,
                    'Vehicle ID': vehicle_id,
                    'Vehicle Type': vehicle_type,
                    'X Coordinate': f"{position[0]:.3f}",
                    'Y Coordinate': f"{position[1]:.3f}",
                    'Length Dimention': f"{length:.3f}",
                    'Width Dimention': f"{width:.3f}",
                    'Speed': f"{speed:.3f}",
                    'Acceleration': f"{acceleration:.3f}",
                    'Route ID': route_id,
                    'Route Edges': route_edges,
                    'Lane ID': lane_id,
                    'Lane Position': f"{lane_position:.3f}",
                    'Lane Index': lane_index,
                    'Changing Lane': changing_lane,
                    'Left Signal': left_signal,
                    'Right Signal': right_signal,
                    'Leader ID': leader_id,
                    'Leader Distance': f"{leader_distance:.3f}" if leader_id else "N/A",
                    'Driving Status': driving_status,
                    'Is Near Exit': is_near_exit
                })

        self.logger.log(f"✅ Vehicle data exported successfully to '{csv_file_path}'", "INFO", 
                        class_name="DataGenerator", function_name="export_vehicle_data_to_csv")
    
    
    #####################################################################
    # The following function is not used in the current implementation. It is kept for future use.

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
            number_of_vehicles = len(vehicles_nearby) if vehicles_nearby else 0
            plt.text(position[0], position[1] + 20, f"{junction}\nVehicles: {number_of_vehicles}", fontsize=12, ha='center', zorder=10, 
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

        export_data_directory = "export_data/network_graph"
        # Create the directory only if it does not exist
        if not os.path.exists(export_data_directory):
            os.makedirs(export_data_directory)
        plt.savefig(f"export_data/network_graph/step_{step_number}.png")
        plt.close()

        self.logger.log("✅ Network graph exported successfully as 'network_graph.png'", "INFO", 
                        class_name="DataGenerator", function_name="export_network_graph")
        

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
            csv_file_path = f"export_data/adjacency_matrix/step_{step_number}.csv"
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
            
            graph_file_path = f"export_data/adjacency_matrix/step_{step_number}.png"
            directory = os.path.dirname(graph_file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(graph_file_path)
            plt.close()

            self.logger.log(f"✅ Adjacency matrix graph exported successfully to '{graph_file_path}'", "INFO", 
                            class_name="DataGenerator", function_name="export_adjacency_matrix")
