# filter_edges.py

def filter_edges(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith("edge:"):
                outfile.write(line[len("edge:"):])  # Remove "edge:"

if __name__ == "__main__":
    zoneA = r"C:\Users\Matan\project_SmartTransportationRuppin\sumo_data_generator\turgi-land\network\zoneA.txt" 
    output_file_zoneA = "zoneA_edges.txt"
    filter_edges(zoneA, output_file_zoneA)
    zoneB = r"C:\Users\Matan\project_SmartTransportationRuppin\sumo_data_generator\turgi-land\network\zoneB.txt"
    output_file_zoneB = "zoneB_edges.txt"
    filter_edges(zoneB, output_file_zoneB)
    zoneC = r"C:\Users\Matan\project_SmartTransportationRuppin\sumo_data_generator\turgi-land\network\zoneC.txt"
    output_file_zoneC = "zoneC_edges.txt" 
    filter_edges(zoneC, output_file_zoneC)

    