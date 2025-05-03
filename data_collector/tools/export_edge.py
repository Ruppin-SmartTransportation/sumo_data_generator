import xml.etree.ElementTree as ET

def extract_edges_from_netxml(file_path):
    """
    Receives the path to a SUMO net.xml file
    and returns a list of all edge IDs.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    edges = []
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        # Skip internal edges (e.g., those starting with ':' or other types if needed)
        if not edge_id.startswith(':'):
            edges.append(edge_id)
    
    return edges

# Example usage:
edges_list = extract_edges_from_netxml(r"C:\Users\Matan\project_SmartTransportationRuppin\sumo_data_generator\additional_data_collector\data_collector\sumo_config\turgi-land.net.xml")

# Write edges to a text file
output_file_path = r"C:\Users\Matan\project_SmartTransportationRuppin\sumo_data_generator\additional_data_collector\data_collector\tools\edges_list.txt"
with open(output_file_path, 'w') as file:
    for edge in edges_list:
        file.write(edge + '\n')
