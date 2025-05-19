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

    sim.schedule_from_config(config)



    # Launch SUMO GUI with TraCI
    sumo_cmd = [sumo_binary, "-c", sumo_cfg_path, "--start"]

    traci.start(sumo_cmd)

    # Step the simulation forward
    step = 0
    num_weeks = config["vehicle_generation"]["simulation_weeks"]
    seconds_in_day = 86400
    seconds_in_week = seconds_in_day * 7
    limit = num_weeks * seconds_in_week
    while step < limit:
        traci.simulationStep()
        sim.dispatch(step, traci)
        
        for vid in sim.get_vehicles_in_route():
            vehicle = sim.db.vehicles[vid]
            if vehicle.status == "in_route":
                current_edge = traci.vehicle.getRoadID(vid)
                vehicle.current_edge = current_edge
                vehicle.current_position = traci.vehicle.getLanePosition(vid)
                vehicle.current_speed = traci.vehicle.getSpeed(vid)
                vehicle.current_lane = traci.vehicle.getLaneID(vid)
                vehicle.current_x, vehicle.current_y = traci.vehicle.getPosition(vid)

                if current_edge == vehicle.current_destination_edge:
                    vehicle.status = "parked"
                    vehicle.current_edge = vehicle.destinations[vehicle.current_destination_name]["edge"]
                    vehicle.current_position = vehicle.destinations[vehicle.current_destination_name]["position"]
                    print(f"Vehicle {vehicle.id} arrived at destination {vehicle.current_destination_name} at {sim.convert_seconds_to_time(step)}.")
        step += 1
    # input("Press Enter to close simulation...")
    traci.close()

def add_all_vehicles(sim, traci):
    """
    Add all vehicles to the simulation.
    """
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

