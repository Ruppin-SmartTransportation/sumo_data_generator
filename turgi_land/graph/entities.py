import re
import random

class Junction:
    """
    Represents a fixed junction point in the traffic network.
    Holds incoming and outgoing road connections and basic spatial metadata.
    """
    def __init__(self, junction_id, x=0.0, y=0.0, junc_type="priority", zone=None):
        self.id = junction_id
        self.x = x
        self.y = y
        self.type = junc_type
        self.zone = zone

        self.incoming_roads = set()  # Set of incoming road IDs
        self.outgoing_roads = set()  # Set of outgoing road IDs

    def add_incoming(self, road_id):
        self.incoming_roads.add(road_id)

    def add_outgoing(self, road_id):
        self.outgoing_roads.add(road_id)

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "type": self.type,
            "zone": self.zone,
            "incoming": sorted(self.incoming_roads),
            "outgoing": sorted(self.outgoing_roads)
        }


class Road:
    """
    Represents a road (edge) connecting two junctions.
    Includes static properties such as speed, length, and lane count.
    """
    def __init__(self, road_id, from_junction, to_junction, speed=13.89, length=100.0, num_lanes=1, zone=None):
        self.id = road_id
        self.from_junction = from_junction
        self.to_junction = to_junction
        self.speed = speed
        self.length = length
        self.num_lanes = num_lanes
        self.zone = zone  # Zone label (e.g. 'A', 'B', 'C', or 'H')

        # Optional attributes (future use):
        self.vehicles_on_road = set()
        self.density = 0.0  # Computed as vehicles / (length * num_lanes)

    def set_density(self):
        if self.length > 0 and self.num_lanes > 0:
            self.density = len(self.vehicles_on_road) / (self.length * self.num_lanes)
        else:
            self.density = 0.0

    def add_vehicle(self, vehicle_id):
        """
        Adds a vehicle ID to the road and updates density.
        """
        self.vehicles_on_road.add(vehicle_id)
        self.set_density()

    def remove_vehicle(self, vehicle_id):
        """
        Removes a vehicle ID from the road and updates density.
        """
        self.vehicles_on_road.discard(vehicle_id)
        self.set_density()

    def get_density(self):
        """
        Returns the current density of the road.
        """
        return self.density
    

    def to_dict(self):
        return {
            "id": self.id,
            "from": self.from_junction,
            "to": self.to_junction,
            "speed": self.speed,
            "length": self.length,
            "num_lanes": self.num_lanes,
            "zone": self.zone,
            "density": self.density
        }


class Vehicle:
    """
    Represents a dynamic vehicle in the simulation.
    Tracks position, movement, physical characteristics, and zone associations.
    """
    def __init__(
        self,
        vehicle_id,
        vehicle_type,
        current_edge,
        current_position=0.0,
        speed=0.0,
        acceleration=0.0,
        route=None,
        length=4.5,
        width=1.8,
        height=1.5,
        max_speed=33.33,
        current_x=None,
        current_y=None,
        current_zone=None,
        color='green',
        status="parked",
        is_stagnant=False
    ):
        self.id = vehicle_id
        self.vehicle_type = vehicle_type
        self.current_edge = current_edge
        self.current_position = current_position
        self.speed = speed
        self.acceleration = acceleration
        self.route = route if route else []

        self.length = length
        self.width = width
        self.height = height
        self.max_speed = max_speed

        self.current_x = current_x
        self.current_y = current_y

        self.origin_edge = current_edge
        self.origin_position = current_position
        self.origin_location = (current_x, current_y) if current_x is not None and current_y is not None else None

        self.origin_zone = current_zone
        self.current_zone = current_zone

        self.color = color
        self.status = status  # e.g., "moving", "parked"
        self.is_stagnant = is_stagnant  # True if vehicle is not tracked by the model

    def update_state(self, current_edge, current_position, speed, acceleration, current_x=None, current_y=None, current_zone=None):
        self.current_edge = current_edge
        self.current_position = current_position
        self.speed = speed
        self.acceleration = acceleration
        if current_x is not None and current_y is not None:
            self.current_x = current_x
            self.current_y = current_y
        if current_zone is not None:
            self.current_zone = current_zone

    def to_dict(self):
        return {
            "id": self.id,
            "edge": self.current_edge,
            "position": self.current_position,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "route": self.route,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "max_speed": self.max_speed,
            "x": self.current_x,
            "y": self.current_y,
            "origin_zone": self.origin_zone,
            "origin_location": self.origin_location,
            "origin_edge": self.origin_edge,
            "origin_position": self.origin_position,
            "current_zone": self.current_zone
        }


class Zone:
    """
    Represents a traffic zone (e.g., 'A', 'B', 'C', 'H').
    Tracks all edges and junctions belonging to the zone,
    as well as vehicles that originated or are currently located in the zone.
    """
    def __init__(self, zone_id, description=None):
        self.id = zone_id
        self.description = description  # Optional textual description of the zone
        self.edges = set()
        self.junctions = set()
        self.original_vehicles = set()  # Vehicles that originated here
        self.current_vehicles = set()   # Vehicles currently here

    def add_edge(self, edge_id):
        self.edges.add(edge_id)

    def add_junction(self, junction_id):
        self.junctions.add(junction_id)

    def add_original_vehicle(self, vehicle_id):
        self.original_vehicles.add(vehicle_id)

    def add_current_vehicle(self, vehicle_id):
        self.current_vehicles.add(vehicle_id)

    def remove_current_vehicle(self, vehicle_id):
        self.current_vehicles.discard(vehicle_id)

    def get_random_edge(self):
        import random
        return random.choice(list(self.edges)) if self.edges else None

    def get_random_junction(self):
        import random
        return random.choice(list(self.junctions)) if self.junctions else None

    def get_random_vehicle(self):
        import random
        return random.choice(list(self.original_vehicles)) if self.original_vehicles else None

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "edges": sorted(self.edges),
            "junctions": sorted(self.junctions),
            "original_vehicles": sorted(self.original_vehicles),
            "current_vehicles": sorted(self.current_vehicles)
        }


# Keep these at the bottom so that entity classes are defined first
class DataBase:
    """
    Centralized store for all simulation entities:
    roads, junctions, vehicles, and zones.
    SimManager uses this to read/write state.
    """
    def __init__(self):
        self.roads = {}       # edge_id -> Road
        self.junctions = {}   # junction_id -> Junction
        self.vehicles = {}    # vehicle_id -> Vehicle
        self.zones = {}       # zone_id -> Zone

    def add_road(self, road):
        self.roads[road.id] = road

    def add_junction(self, junction):
        self.junctions[junction.id] = junction

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.id] = vehicle

    def update_vehicle(self, vehicle_id, **kwargs):
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            return
        vehicle.update_state(
            kwargs.get("current_edge", vehicle.current_edge),
            kwargs.get("current_position", vehicle.current_position),
            kwargs.get("speed", vehicle.speed),
            kwargs.get("acceleration", vehicle.acceleration),
            current_x=kwargs.get("current_x", vehicle.current_x),
            current_y=kwargs.get("current_y", vehicle.current_y),
            current_zone=kwargs.get("current_zone", vehicle.current_zone)
        )

    def update_junction(self, junction_id, incoming_roads=None, outgoing_roads=None):
        junction = self.get_junction(junction_id)
        if not junction:
            return
        if incoming_roads is not None:
            junction.incoming_roads = set(incoming_roads)
        if outgoing_roads is not None:
            junction.outgoing_roads = set(outgoing_roads)

    def update_road(self, road_id, vehicles_on_road=None):
        road = self.get_road(road_id)
        if not road:
            return
        if vehicles_on_road is not None:
            road.vehicles_on_road = vehicles_on_road
            road.set_density()

    def add_zone(self, zone):
        self.zones[zone.id] = zone

    def get_road(self, road_id):
        return self.roads.get(road_id)

    def get_junction(self, junction_id):
        return self.junctions.get(junction_id)

    def get_vehicle(self, vehicle_id):
        return self.vehicles.get(vehicle_id)

    def get_zone(self, zone_id):
        return self.zones.get(zone_id)
    
    def print_zone_statistics(self):
        print("\n--- Simulation Zone Statistics ---")
        for zid, zone in self.zones.items():
            print(f"\nZone {zid}:")

            # Roads by lane count
            lane_counts = {}
            for eid in zone.edges:
                road = self.get_road(eid)
                lane_counts[road.num_lanes] = lane_counts.get(road.num_lanes, 0) + 1
            total_roads = sum(lane_counts.values())
            print(f"  Roads: {total_roads}")
            for lanes, count in sorted(lane_counts.items()):
                print(f"    {count} with {lanes} lane(s)")

            # Junctions and traffic lights
            total_junctions = len(zone.junctions)
            num_tls = sum(1 for jid in zone.junctions if self.get_junction(jid).type == "traffic_light")
            print(f"  Junctions: {total_junctions} ({num_tls} traffic lights)")

            # Vehicles
            vehicle_type_counts = {}
            stagnant_count = 0
            for vid in zone.current_vehicles:
                vehicle = self.get_vehicle(vid)
                vehicle_type_counts[vehicle.vehicle_type] = vehicle_type_counts.get(vehicle.vehicle_type, 0) + 1
                if vehicle.is_stagnant:
                    stagnant_count += 1

            total_vehicles = len(zone.current_vehicles)
            print(f"  Vehicles: {total_vehicles} ({stagnant_count} stagnant)")
            for vtype, count in vehicle_type_counts.items():
                print(f"    {count} {vtype}")

        print("\n total vehicles in simulation:", len(self.vehicles))
        print("\n-----------------------------------\n")


class SimManager:
    """
    Manages the simulation process using IDs only,
    delegating storage and state to the DataBase.
    """
    def __init__(self, net, zone_file_map):
        self.net = net
        self.zone_file_map = zone_file_map  # {"A": "zoneA.txt", ...}
        self.db = DataBase()

    def load_zones(self):
        zone_objects = {}

        # Collect edges by zone attribute
        for edge in self.net.getEdges():
            zone_attr = edge.getParam("zone")
            if not zone_attr:
                continue
            zone_id = zone_attr.upper()
            if zone_id not in zone_objects:
                zone_objects[zone_id] = Zone(zone_id)
                print(f"Zone {zone_id} created.")

            road = Road(
                road_id=edge.getID(),
                from_junction=edge.getFromNode().getID(),
                to_junction=edge.getToNode().getID(),
                speed=edge.getSpeed(),
                length=edge.getLength(),
                num_lanes=len(edge.getLanes()),
                zone=zone_id
            )
            self.db.add_road(road)
            zone_objects[zone_id].add_edge(road.id)
            # print(f"Road {road.id} added to zone {zone_id}.")

        # Collect junctions by zone attribute
        for junction in self.net.getNodes():
            zone_attr = junction.getParam("zone")
            if not zone_attr:
                continue
            zone_id = zone_attr.upper()
            if zone_id not in zone_objects:
                zone_objects[zone_id] = Zone(zone_id)

            junc = Junction(
                junction_id=junction.getID(),
                x=junction.getCoord()[0],
                y=junction.getCoord()[1],
                junc_type=junction.getType(),
                zone=zone_id
            )
            self.db.add_junction(junc)
            zone_objects[zone_id].add_junction(junc.id)

        for zone in zone_objects.values():
            self.db.add_zone(zone)


    def extract_edges_from_file(self, filepath):
        edge_ids = set()
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("junction:"):
                    continue  # Skip junction declarations

                match = re.search(r'from([^\s]+)to([^\s]+)', line)
                if match:
                    edge_ids.add(match.group(1).split('_')[0])
                    edge_ids.add(match.group(2).split('_')[0])
                elif line.startswith("edge:"):
                    edge_ids.add(line.split("edge:")[1].strip().split('_')[0])
                else:
                    edge_ids.add(line.split()[0].split('_')[0])

        return list(edge_ids)

    def populate_vehicles_from_config(self, config):
        total_vehicles = config["vehicle_generation"]["total_num_vehicles"]
        print(f"Total vehicles to generate: {total_vehicles}")
        zone_alloc = config["vehicle_generation"]["zone_allocation"]
        vehicle_types = config["vehicle_generation"]["vehicle_types"]

        vehicle_id_counter = 0

        # Track per-zone assignments
        active_zone_ids = [zid for zid in zone_alloc if zid.lower() not in ("h", "stagnant")]
        num_zones = len(active_zone_ids)

        sum_vehicles = 0
        # Calculate total stagnant vehicles
        stagnant_per_zone = {}
        total_stagnant = 0
        if "stagnant" in zone_alloc:
            stagnant_pct = zone_alloc["stagnant"]["percentage"]
            total_stagnant = round((stagnant_pct / 100) * total_vehicles)
            base_stag = total_stagnant // num_zones
            extra_stag = total_stagnant % num_zones
            for i, zid in enumerate(active_zone_ids):
                stagnant_per_zone[zid] = base_stag + (1 if i < extra_stag else 0)
                sum_vehicles += stagnant_per_zone[zid]
        # Calculate total vehicles in other zones
        active_zone_vehicle_counts = {}
        for zid in zone_alloc:
            if zid.lower() == "stagnant":
                continue
            if zid.upper() == "H":
                continue
            percentage = zone_alloc[zid]["percentage"]
            num_zone_vehicles = round((percentage / 100) * total_vehicles)
            active_zone_vehicle_counts[zid] = num_zone_vehicles
            sum_vehicles += num_zone_vehicles
    
        
        while total_vehicles != sum_vehicles:
            # Randomly select a zone
            zone_id = random.choice(active_zone_ids)
            active_zone_vehicle_counts[zone_id] += 1
            sum_vehicles += 1
            print(f"Adding vehicle to zone {zone_id}.")

        print(f"Total sum_vehicles vehicles: {sum_vehicles}")
        

        for zone_id, zone_cfg in zone_alloc.items():
            if zone_id.lower() == "stagnant":
                continue
            if zone_id.upper() == "H":
                continue  # Skip highway zone
            num_zone_vehicles = active_zone_vehicle_counts[zone_id] + stagnant_per_zone.get(zone_id, 0)
            type_distribution = zone_cfg["vehicle_type_distribution"]

            # Get eligible roads in this zone
            zone = self.db.get_zone(zone_id)
            eligible_roads = [eid for eid in zone.edges if self.db.get_road(eid).num_lanes == 1]
            if not eligible_roads:
                continue

            # Allocate vehicles across types
            type_allocations = {
                vtype: round((vperc / 100) * num_zone_vehicles)
                for vtype, vperc in type_distribution.items()
            }
            
            vehicle_specs = [
                (vtype, vehicle_types[vtype].copy())
                for vtype, count in type_allocations.items()
                for _ in range(count)
                ]
            random.shuffle(vehicle_specs)

            per_road = len(vehicle_specs) // len(eligible_roads)
            overflow = len(vehicle_specs) % len(eligible_roads)

            vehicle_iter = iter(vehicle_specs)
            created_vehicle_ids = []

            for i, road_id in enumerate(eligible_roads):
                vehicles_on_road = per_road + (1 if i < overflow else 0)
                road = self.db.get_road(road_id)
                net_edge = self.net.getEdge(road_id)
                length = road.length
                spacing = length / (vehicles_on_road + 1)

                for j in range(vehicles_on_road):
                    try:
                        vtype, vcfg = next(vehicle_iter)
                    except StopIteration:
                        break

                    pos = (j + 1) * spacing
                    x, y = net_edge.getLane(0).getShape()[0]

                    vehicle = Vehicle(
                        vehicle_id=f"veh_{vehicle_id_counter}",
                        vehicle_type=vtype,
                        current_zone=zone_id,
                        current_edge=road_id,
                        current_position=pos,
                        current_x=x,
                        current_y=y,
                        length=vcfg["length"],
                        width=vcfg["width"],
                        height=vcfg["height"],
                        color=vcfg["color"],
                        status="parked",
                        is_stagnant=False
                    )

                    self.db.add_vehicle(vehicle)
                    road.add_vehicle(vehicle.id)
                    zone.add_original_vehicle(vehicle.id)
                    zone.add_current_vehicle(vehicle.id)

                    vehicle_id_counter += 1

                    # print(f"Added {vtype} vehicle {vehicle.id} in zone {zone_id} on road {road_id} at position {pos:.2f}")

            # Randomly convert some vehicles in this zone to stagnant
            zone_vehicle_ids = list(zone.current_vehicles)
            if 'stagnant' in zone_alloc:
                num_to_convert = stagnant_per_zone.get(zone_id, 0)
                to_convert = random.sample(zone_vehicle_ids, min(num_to_convert, len(zone_vehicle_ids)))
                for vid in to_convert:
                    v = self.db.get_vehicle(vid)
                    v.is_stagnant = True
                    v.color = "purple"
                    # print(f"Converted vehicle {vid} to stagnant in zone {zone_id}")


    def print_vehicle_statistics(self):
        self.db.print_zone_statistics()
