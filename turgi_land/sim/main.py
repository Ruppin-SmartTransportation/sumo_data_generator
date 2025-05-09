from sumolib import net
from ..graph.entities import SimManager
import json

if __name__ == "__main__":
    # Load SUMO network
    net = net.readNet("turgi_land/net/turgi-land.net.xml")

    # Define zone file map
    zone_file_map = {
        "A": "turgi_land/net/zoneA.txt",
        "B": "turgi_land/net/zoneB.txt",
        "C": "turgi_land/net/zoneC.txt",
        "H": "turgi_land/net/Hwys.txt"
    }

    # Initialize and load simulation
    sim = SimManager(net, zone_file_map)
    sim.load_zones()

    # Summary output
    print(f"Zones loaded: {len(sim.db.zones)}")
    for zid, zone in sim.db.zones.items():
        print(f"Zone {zid}: {len(zone.edges)} roads, {len(zone.junctions)} junctions")

    with open("turgi_land/sim/simulation.config.json") as f:
        config = json.load(f)
    sim.populate_vehicles_from_config(config)

    sim.print_vehicle_statistics()



