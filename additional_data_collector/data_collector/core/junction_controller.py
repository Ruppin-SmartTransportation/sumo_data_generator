import traci
import networkx as nx
import matplotlib.pyplot as plt

class JunctionController:
    """ Handles operations related to junctions (static nodes) in the SUMO simulation. """

    def __init__(self, logger):
        self.logger = logger

    def get_all_junctions(self):
        """ Retrieves all static junctions in the network. """
        junctions = traci.junction.getIDList()
        self.logger.log(f"📍 Static Junctions Retrieved: {junctions}", "INFO",
                        class_name="JunctionController", function_name="get_all_junctions")
        return junctions

    def subscribe_to_junctions(self):
        """ Subscribes all junctions to track vehicles in their vicinity. """
        for junction_id in traci.junction.getIDList():
            traci.junction.subscribeContext(
                junction_id,
                traci.constants.CMD_GET_VEHICLE_VARIABLE,  # vehicle information
                20.0  # radius in meters to track vehicles around the junction
            )

    def get_junction_info(self, junction_id):
        """ Retrieves detailed information about a specific junction. """
        info = {}

        # position of the junction
        position = traci.junction.getPosition(junction_id)
        info["Position"] = f"({position[0]:.2f}, {position[1]:.2f})"

        # count of vehicles in the junction
        vehicles_nearby = traci.junction.getContextSubscriptionResults(junction_id)
        vehicle_count = len(vehicles_nearby) if vehicles_nearby else 0
        info["Vehicles in Junction"] = vehicle_count

        # traffic light state
        if junction_id in traci.trafficlight.getIDList():
            light_state = traci.trafficlight.getRedYellowGreenState(junction_id)
            info["Traffic Light State"] = light_state
        else:
            info["Traffic Light State"] = "No Traffic Light"

        # finding all edges connected to the junction
        incoming_edges = traci.junction.getIncomingEdges(junction_id)
        outgoing_edges = traci.junction.getOutgoingEdges(junction_id)
        all_edges = list(set(incoming_edges + outgoing_edges))

        # internal edges are edges that are not connected to any other 
        real_edges = [edge for edge in all_edges if not edge.startswith(":")]
        internal_edges = [edge for edge in all_edges if edge.startswith(":")]

        info["Connected Edges"] = real_edges
        info["Internal Edges"] = internal_edges

        # finding all lanes connected to the junction
        real_lanes = []
        internal_lanes = []

        for edge in real_edges:
            lane_count = traci.edge.getLaneNumber(edge)
            for i in range(lane_count):
                lane_id = f"{edge}_{i}"
                real_lanes.append(lane_id)

        for edge in internal_edges:
            lane_count = traci.edge.getLaneNumber(edge)
            for i in range(lane_count):
                lane_id = f"{edge}_{i}"
                internal_lanes.append(lane_id)

        info["Connected Lanes"] = real_lanes
        info["Internal Lanes"] = internal_lanes

        # # if true - export the network graph
        # if export_graph:
        #     self.export_network_graph()

        return info

    def log_all_junctions_info(self):
        """ Logs detailed information about all junctions in the simulation. """
        junctions = self.get_all_junctions()
        for junction_id in junctions:
            junction_info = self.get_junction_info(junction_id)
            log_message = f"🔹 Junction {junction_id} | Position: {junction_info['Position']} | " \
                          f"Vehicles: {junction_info['Vehicles in Junction']} | " \
                          f"Traffic Light: {junction_info['Traffic Light State']} | " \
                          f"Edges: {junction_info['Connected Edges']} | " \
                          f"Internal Edges: {junction_info['Internal Edges']} | " \
                          f"Lanes: {junction_info['Connected Lanes']} | " \
                          f"Internal Lanes: {junction_info['Internal Lanes']}"
            self.logger.log(log_message, "INFO",
                            class_name="JunctionController", function_name="log_all_junctions_info")

    def export_network_graph(self, step_number, _filtered_static_nodes):
        """ Exports the network graph as an image using matplotlib. """
        self.logger.log("📡 Exporting network graph...", "INFO", 
                        class_name="JunctionController", function_name="export_network_graph")

        # create figure
        plt.figure(figsize=(12, 10))

        # pull all junctions and their outgoing edges
        junction_positions = {junction: traci.junction.getPosition(junction) for junction in _filtered_static_nodes}

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
        plt.savefig(f"network_graph_step_{step_number}.png")
        plt.close()

        self.logger.log("✅ Network graph exported successfully as 'network_graph.png'", "INFO", 
                        class_name="JunctionController", function_name="export_network_graph")
