import pandas as pd
import traci
import xml.etree.ElementTree as ET
import sumolib
import math
import h5py
import numpy as np
import csv
import pandas as pd
from pyproj import Proj
from scipy.spatial import KDTree
import pickle

def load_pems_data(file_path):
    """Load PeMS sensor metadata CSV/XML."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xml'):
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = []
        # index,sensor_id,latitude,longitude
        for station in root.findall('station'):
            sid = station.get('index')
            lat = float(station.find('latitude').text)
            lon = float(station.find('longitude').text)
            data.append({'sensor_id': sid, 'latitude': lat, 'longitude': lon})
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format. Use CSV or XML.")

def load_adjacency_matrix(file_path):
    """Load adjacency matrix from a pickle (.pkl) file."""
    with open(file_path, 'rb') as f:
        adj_matrix = pickle.load(f, encoding='latin1')
    return np.array(adj_matrix[2], dtype=np.float32)

def map_sensors_to_edges(sensor_file="convertors/sensor_graph/graph_sensor_locations.csv",
                            edges_file = "configs/metrla/pems_edges_updated.edg.xml",
                                output_file="configs/metrla/sensor_to_edge.csv"):
    # Define UTM projection (adjust zone if needed)
    proj = Proj(proj="utm", zone=10, ellps="WGS84", datum="WGS84")

    # Load sensor data
    sensor_df = pd.read_csv(sensor_file)

    # Convert sensor GPS (lat, lon) to UTM (x, y)
    sensor_df["utm_x"], sensor_df["utm_y"] = proj(sensor_df["longitude"].values, sensor_df["latitude"].values)

    # Load SUMO edges
    tree = ET.parse(edges_file)
    root = tree.getroot()

    edges = []  # Store edge data
    edge_positions = []  # Store edge start points for KDTree

    for edge in root.findall("edge"):
        from_node = edge.get("from")
        to_node = edge.get("to")

        if edge.get("x1") is None or edge.get("y1") is None:
            continue  # Skip edges with missing coordinates
        
        from_x, from_y = proj(float(edge.get("x1")), float(edge.get("y1")))
        to_x, to_y = proj(float(edge.get("x2")), float(edge.get("y2")))
        edges.append((edge.get("id"), from_x, from_y, to_x, to_y))
        edge_positions.append((from_x, from_y))  # Use from-node as reference for KDTree

    # Build KDTree for nearest neighbor search
    kdtree = KDTree(edge_positions)

    # Find nearest SUMO edge for each sensor
    sensor_to_edge = []
    for _, row in sensor_df.iterrows():
        print("Edges list:", edges)  # Ensure edges exist
        print("Index:", idx)         # Check which index is failing
        print("Total edges:", len(edges))
        if len(edges) > 0:
            print("First edge:", edges[0])  # See what edges are being retrieved
        else:
            print("No edges found for this sensor!")

        utm_x, utm_y = row["utm_x"], row["utm_y"]
        dist, idx = kdtree.query((utm_x, utm_y))  # Get nearest edge index
        closest_edge = edges[idx][0]  # Get edge ID
        sensor_to_edge.append((row["sensor_id"], closest_edge))

    # Save mapping to CSV
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor_id", "edge_id"])
        writer.writerows(sensor_to_edge)

    print(f"Mapping saved to {output_file}")

def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def project_point_onto_lane(x, y, lane):
    """Finds the closest position along the lane for a given (x, y) coordinate."""
    min_distance = float("inf")
    projected_position = 0
    lane_length = lane.getLength()
    
    shape = lane.getShape()  # List of (x, y) tuples
    
    for i in range(len(shape) - 1):
        x1, y1 = shape[i]
        x2, y2 = shape[i + 1]
        
        # Project (x, y) onto the segment (x1, y1) -> (x2, y2)
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx ** 2 + dy ** 2
        
        if length_sq == 0:  # Avoid division by zero
            continue
        
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
        proj_x, proj_y = x1 + t * dx, y1 + t * dy
        distance = euclidean_distance(x, y, proj_x, proj_y)
        
        if distance < min_distance:
            min_distance = distance
            projected_position = t * lane_length  # Position along the lane
    
    # Ensure the position is within lane bounds
    projected_position = max(0, min(projected_position, lane_length - 1))
    return projected_position

def is_valid_departure_edge(edge):
    """Returns True if edge is valid for vehicle departure, else False."""
    try:
        # Exclude internal edges (junctions)
        if edge.getFunction() == "internal":
            return False
        
        return True  
    
    except KeyError:
        return False  # Edge does not exist
    
def convert_and_map_sensors(input_csv, output_csv, sumo_net):
    """
    Converts latitude/longitude in a CSV file to SUMO x, y coordinates,
    finds the closest edge manually if needed, and ensures valid lane positions.
    """
    df = pd.read_csv(input_csv)

    traci.start(["sumo", "--net-file", sumo_net, "--start"], label="converter")
    conn = traci.getConnection("converter")
    
    net = sumolib.net.readNet(sumo_net)

    sumo_x, sumo_y, sumo_edges, lane_positions = [], [], [], []

    for _, row in df.iterrows():
        # Convert lat/lon to SUMO (x, y)
        x, y = conn.simulation.convertGeo(row['longitude'], row['latitude'], fromGeo=True)
        sumo_x.append(x)
        sumo_y.append(y)

        # Find nearest edge
        neighboring_edges = net.getNeighboringEdges(x, y)
        if neighboring_edges:
            valid_edges = [e for e in neighboring_edges if is_valid_departure_edge(e[0])]
            closest_edge = min(valid_edges, key=lambda e: e[1])[0]
        else:
            # Fallback: manually find closest edge
            min_distance = float("inf")
            closest_edge = None
            for edge in net.getEdges():
                if False == is_valid_departure_edge(edge):
                    continue
                for px, py in edge.getShape():
                    dist = euclidean_distance(x, y, px, py)
                    if dist < min_distance:
                        min_distance = dist
                        closest_edge = edge

        if closest_edge:
            edge_id = closest_edge.getID()
            lane = closest_edge.getLanes()[0]  # Get first lane
            lane_length = lane.getLength()

            # Compute projected position
            position_on_lane = project_point_onto_lane(x, y, lane)
            
            sumo_edges.append(edge_id)
            lane_positions.append(position_on_lane)
        else:
            sumo_edges.append("N/A")
            lane_positions.append("N/A")

    traci.close()

    # Save results
    df['sumo_x'] = sumo_x
    df['sumo_y'] = sumo_y
    df['sumo_edge'] = sumo_edges
    df['lane_position'] = lane_positions
    df.to_csv(output_csv, index=False)
    print(f"Sensor mapping saved to {output_csv}")


def generate_detector_xml(sensor_csv, output_xml):
    """
    Generate a SUMO additional file (.add.xml) to add sensors (loop detectors) to edges.
    
    Parameters:
    - sensor_csv: CSV file with sensor_id, sumo_edge, sumo_x, sumo_y.
    - output_xml: Path to save the .add.xml file.
    """
    df = pd.read_csv(sensor_csv)

    # Create XML root
    root = ET.Element("additional")

    for _, row in df.iterrows():
        sensor_id = f"sensor_{row['sensor_id']}"
        edge_id = row['sumo_edge']
        pos = row['lane_position']  # Position along the edge (adjustable)
        
        detector = ET.SubElement(root, "e1Detector", {
            "id": sensor_id,
            "lane": f"{edge_id}_0",  # Assuming the first lane of the edge
            "pos": str(pos),
            "freq": "60",  # Sample every 60s
            "file": "detector_output.xml"
        })

    # Save XML file
    tree = ET.ElementTree(root)
    tree.write(output_xml)
    print(f"Sensor detectors saved to {output_xml}")

def load_and_prepare_data(file_path):

    with h5py.File(file_path, "r") as f:
        timestamps = f["df"]["axis1"][:]  # Assuming axis1 holds timestamps
        # Ensure timestamps are integers
        timestamps = timestamps.astype(int)
        # Convert from nanoseconds to seconds
        timestamps = timestamps // 1_000_000_000
        
        # Convert to datetime
        timestamps = pd.to_datetime(timestamps, unit="s")
        
        columns = f["df"]["axis0"][:]  # Should be axis0, not axis1
        columns = columns.astype(int)
        
        traffic_data = f["df"]["block0_values"][:]  # (num_samples, num_sensors)

    # Create the DataFrame
    df = pd.DataFrame(traffic_data, index=timestamps, columns=columns)
    # Replace 0 values with NaN to avoid interpolating valid zero-speed scenarios
    df.replace(0, np.nan, inplace=True)

    # Interpolate missing values (e.g., linear interpolation along the time axis)
    df.interpolate(method="linear", inplace=True)

    # Fill any remaining NaNs (e.g., forward-fill or backward-fill)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    print(df[:10][:3])

    return df


def generate_sumo_trips(metr_df, sensor_mapping_file, output_file="metr_la_trips.rou.xml", chunk_size=10000):
    """Generates a SUMO .rou.xml file with trips based on METR-LA speed data."""
    # Load sensor-edge mapping
    sensor_mapping = pd.read_csv(sensor_mapping_file)

    # Initialize XML structure
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="0.8", decel="4.5", length="5", maxSpeed="25")

    # Process data in chunks
    timestamps = metr_df.index
    for i in range(0, len(timestamps), chunk_size):
        chunk = metr_df.iloc[i : i + chunk_size]
        chunk_long = chunk.reset_index().melt(id_vars=["index"], var_name="sensor_id", value_name="speed")
        chunk_long.rename(columns={"index": "timestamp"}, inplace=True)

        # Convert time & speed
        chunk_long["sumo_time"] = chunk_long["timestamp"].dt.hour * 3600 + chunk_long["timestamp"].dt.minute * 60
        chunk_long["speed"] = chunk_long["speed"] * 0.44704  # Convert mph to m/s

        # Merge with sensor mapping
        chunk_long = chunk_long.merge(sensor_mapping, on="sensor_id", how="left")
        chunk_long.dropna(subset=["sumo_edge"], inplace=True)  # Remove unmatched sensors

        # Add vehicles to XML
        for _, row in chunk_long.iterrows():
            edge_id = str(row["sumo_edge"])
            depart_time = str(int(row["sumo_time"]))
            speed = str(row["speed"])

            vehicle = ET.SubElement(root, "vehicle", id=f"veh_{depart_time}_{edge_id}", type="car", depart=depart_time)
            route = ET.SubElement(vehicle, "route", edges=edge_id)
            ET.SubElement(vehicle, "param", key="speed", value=speed)

    # Save XML
    tree = ET.ElementTree(root)
    tree.write(output_file)
    print(f"✅ SUMO trip file saved: {output_file}")


def process_metr_la_in_chunks(file_path, sensor_mapping_file, output_file="metr_la_trips.rou.xml", chunk_size=5000):
    """Processes METR-LA h5 in chunks to prevent high memory usage and generates SUMO trips."""

    # Load sensor mapping (small file, safe to load fully)
    sensor_mapping = pd.read_csv(sensor_mapping_file)

    # Initialize XML structure
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="0.8", decel="4.5", length="5", maxSpeed="25")

    with h5py.File(file_path, "r") as f:
        timestamps = f["df"]["axis1"][:]  # Raw timestamps
        timestamps = (timestamps // 1_000_000_000).astype(int)  # Convert ns to seconds
        timestamps = pd.to_datetime(timestamps, unit="s")

        columns = f["df"]["axis0"][:].astype(int)  # Sensor IDs (edge IDs)
        num_rows = f["df"]["block0_values"].shape[0]  # Total time steps

        # Process in chunks
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)

            # Read a chunk from the dataset
            chunk_data = f["df"]["block0_values"][start:end, :]
            df_chunk = pd.DataFrame(chunk_data, index=timestamps[start:end], columns=columns)

            # Replace 0 values with NaN to avoid misinterpreting zero-speed scenarios
            df_chunk.replace(0, np.nan, inplace=True)

            # Interpolate missing values (row-wise)
            df_chunk.interpolate(method="linear", inplace=True)
            df_chunk.fillna(method="ffill", inplace=True)
            df_chunk.fillna(method="bfill", inplace=True)

            # Reshape data for merging
            chunk_long = df_chunk.reset_index().melt(id_vars=["index"], var_name="sensor_id", value_name="speed")
            chunk_long.rename(columns={"index": "timestamp"}, inplace=True)

            # Convert time to SUMO format (seconds of the day)
            chunk_long["sumo_time"] = chunk_long["timestamp"].dt.hour * 3600 + chunk_long["timestamp"].dt.minute * 60
            chunk_long["speed"] = chunk_long["speed"] * 0.44704  # Convert mph to m/s

            # Merge only the chunk with sensor mapping
            chunk_long = chunk_long.merge(sensor_mapping, on="sensor_id", how="left")
            chunk_long.dropna(subset=["sumo_edge"], inplace=True)  # Remove unmatched sensors

            # Write vehicles to XML without keeping them in memory
            for _, row in chunk_long.iterrows():
                edge_id = str(row["sumo_edge"])
                depart_time = str(int(row["sumo_time"]))
                speed = str(row["speed"])

                vehicle = ET.SubElement(root, "vehicle", id=f"veh_{depart_time}_{edge_id}", type="car", depart=depart_time)
                route = ET.SubElement(vehicle, "route", edges=edge_id)
                ET.SubElement(vehicle, "param", key="speed", value=speed)

            print(f"Processed chunk {start}-{end}")

    # Save XML to disk
    tree = ET.ElementTree(root)
    tree.write(output_file)
    print(f"✅ SUMO trip file saved: {output_file}")

def process_metr_la_lazy(file_path, sensor_mapping_file, output_file="metr_la_trips.rou.xml"):
    """Processes METR-LA h5 dataset efficiently without loading it all into memory."""
    
    sensor_mapping = {}
    with open(sensor_mapping_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensor_mapping[int(row["sensor_id"])] = row

    # Store a counter for each (timestamp, edge_id) pair
    vehicle_counters = {}

    with open(output_file, "w") as f_out:
        f_out.write('<routes>\n<vType id="car" accel="0.8" decel="4.5" length="5" maxSpeed="25"/>\n')

        with h5py.File(file_path, "r") as f:
            timestamps = (f["df"]["axis1"][:] // 1_000_000_000).astype(int)
            first_timestamp = timestamps[0]  # Normalize timestamps
            columns = f["df"]["axis0"][:].astype(int)
            num_rows = f["df"]["block0_values"].shape[0]

            for i in range(num_rows):
                row_data = f["df"]["block0_values"][i, :]
                sumo_time = timestamps[i] - first_timestamp

                for sensor_id, speed in zip(columns, row_data):
                    if sensor_id in sensor_mapping and speed > 0:
                        sensor_info = sensor_mapping[sensor_id]
                        edge_id = sensor_info["sumo_edge"]
                        speed_mps = speed * 0.44704  

                        # Create a unique counter for this timestamp + edge
                        key = (sumo_time, edge_id)
                        vehicle_counters[key] = vehicle_counters.get(key, 0) + 1
                        veh_id = f"veh_{sumo_time}_{edge_id}_{vehicle_counters[key]}"  # Make it unique

                        # Write SUMO trip directly to file (streaming)
                        f_out.write(f'<vehicle id="{veh_id}" type="car" depart="{sumo_time}">\n')
                        f_out.write(f'    <route edges="{edge_id}"/>\n')
                        f_out.write(f'    <param key="speed" value="{speed_mps}"/>\n</vehicle>\n')

                if i % 100 == 0:
                    print(f"Processed timestep {i}/{num_rows}")  # Status update

        f_out.write("</routes>\n")

    print(f"✅ Fixed! SUMO trip file saved: {output_file}")

# Run function

if __name__ == "__main__":
    # Filepath to processed traffic dataset
    input = "convertors/sensor_graph/graph_sensor_locations.csv"  
    output = "convertors/sensor_graph/graph_sensor_locations_coo.csv"  
    net = "convertors/sensor_graph/osm.net.xml"
    # Load and normalize
    # convert_and_map_sensors(input, output, net)


    generate_detector_xml(output, "convertors/sensor_graph/sensors.add.xml")
    filepath = "convertors/Datasets/METR-LA.h5"  

    # Load and normalize
    # df = load_and_prepare_data(filepath)

    process_metr_la_lazy(filepath, "convertors/sensor_graph/graph_sensor_locations_coo.csv")
