from sumolib import net
from ..graph.entities import SimManager
import json
import traci
import traci.constants as tc
import os
import sys


sumo_binary = "sumo-gui"

# Paths
net_path = "turgi_land/net/turgi-land.net.xml"
sumo_cfg_path = "turgi_land/net/turgi-land.sumocfg"
config_path = "turgi_land/sim/simulation.config.json"


if __name__ == "__main__":
    # Load SUMO network
    net = net.readNet(net_path)
    with open(config_path) as f:
        config = json.load(f)

    # Initialize and load simulation
    sim = SimManager(net)
    sim.load_zones()
    sim.populate_vehicles_from_config(config)

    # Summary output
    print(f"Zones loaded: {len(sim.db.zones)}")
    for zid, zone in sim.db.zones.items():
        print(f"Zone {zid}: {len(zone.edges)} roads, {len(zone.junctions)} junctions")
    sim.print_vehicle_statistics()


    # Launch SUMO GUI with TraCI
    sumo_binary = "sumo-gui"
    sumo_cmd = [sumo_binary, "-c", sumo_cfg_path, "--start"]

    traci.start(sumo_cmd)

    # Add vehicles to SUMO
    for vehicle in sim.db.vehicles.values():
        route_id = f"route_{vehicle.id}"
        traci.route.add(routeID=route_id, edges=[vehicle.current_edge])

        traci.vehicle.add(
            vehID=vehicle.id,
            routeID=route_id,
            typeID=vehicle.vehicle_type,
            depart=0,
            departPos=vehicle.current_position,
            departSpeed=0,
            departLane="0"
        )
        lane_id = vehicle.current_edge + "_0"
        traci.vehicle.moveTo(vehicle.id, lane_id, vehicle.current_position)
        
    print(f"Injected {len(sim.db.vehicles)} vehicles.")
    input("Press Enter to close simulation...")
    traci.close()


