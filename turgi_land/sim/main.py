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
    sumo_cmd = [sumo_binary, "-c", sumo_cfg_path, "--start"]

    traci.start(sumo_cmd)

    zone_a = sim.db.get_zone("A")
    depart_time = 0
    depart_offset = 1000

    for vid in zone_a.current_vehicles:
        vehicle = sim.db.get_vehicle(vid)

        # if vehicle.is_stagnant:
        #     continue  # Skip stagnant vehicles

        work_dest = vehicle.destinations["work"]
        route_id = f"route_to_work_{vehicle.id}"
        try:
            traci.route.add(routeID=route_id, edges=[vehicle.current_edge, work_dest["edge"]])

            traci.vehicle.add(
                vehID=vehicle.id,
                routeID=route_id,
                typeID=vehicle.vehicle_type,
                depart=0 if depart_time <= depart_offset else (depart_time-depart_offset)//3,
                departPos=vehicle.current_position,
                departSpeed=0,
                departLane="0"
            )
            depart_time += 1
            if vehicle.is_stagnant:
                traci.vehicle.setColor(vehicle.id, (255, 255, 255))  # White for stagnant vehicles

            # lane_id = vehicle.current_edge + "_0"
            # traci.vehicle.moveTo(vehicle.id, lane_id, vehicle.current_position)

            # traci.vehicle.changeTarget(vehicle.id, work_dest["edge"])
            vehicle.status = "in_route"

            # print(f"[INFO] Vehicle {vehicle.id} en route to work at {work_dest['edge']}")

        except traci.exceptions.TraCIException as e:
            print(f"[ERROR] Failed to add or route vehicle {vehicle.id}: {e}")
    
    zone_c = sim.db.get_zone("C")
    depart_time = 0
    for vid in zone_c.current_vehicles:
        vehicle = sim.db.get_vehicle(vid)

        # if vehicle.is_stagnant:
        #     continue  # Skip stagnant vehicles

        work_dest = vehicle.destinations["work"]
        route_id = f"route_to_work_{vehicle.id}"
        try:
            traci.route.add(routeID=route_id, edges=[vehicle.current_edge, work_dest["edge"]])

            traci.vehicle.add(
                vehID=vehicle.id,
                routeID=route_id,
                typeID=vehicle.vehicle_type,
                depart=0 if depart_time <= depart_offset else (depart_time-depart_offset)//3,
                departPos=vehicle.current_position,
                departSpeed=0,
                departLane="0"
            )
            depart_time += 1
            if vehicle.is_stagnant:
                traci.vehicle.setColor(vehicle.id, (255, 255, 255))  # White for stagnant vehicles

            # lane_id = vehicle.current_edge + "_0"
            # traci.vehicle.moveTo(vehicle.id, lane_id, vehicle.current_position)

            # traci.vehicle.changeTarget(vehicle.id, work_dest["edge"])
            vehicle.status = "in_route"

            # print(f"[INFO] Vehicle {vehicle.id} en route to work at {work_dest['edge']}")

        except traci.exceptions.TraCIException as e:
            print(f"[ERROR] Failed to add or route vehicle {vehicle.id}: {e}")
    

    # Step the simulation forward
    for step in range(100000000000):
        traci.simulationStep()
        for vehicle in sim.db.vehicles.values():
            if vehicle.status == "in_route":
                current_edge = traci.vehicle.getRoadID(vehicle.id)
                dest_edge = vehicle.destinations["work"]["edge"]

            if current_edge == dest_edge:
                vehicle.status = "parked"
                print(f"Vehicle {vehicle.id} arrived at work.")
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

