import traci
import random

class VehicleController:
    """ Controls vehicles in the SUMO simulation. """
    def __init__(self, logger):
        self.logger = logger
        self.fastest_vehicle = None
        self.fastest_speed = 0
        self.fastest_step = 0

    def get_active_vehicles(self):
        """ Retrieves the list of all active vehicles in the simulation. """
        return traci.vehicle.getIDList()

    def update_vehicle_speed(self, vehicle_id, speed):
        """ Updates the speed of a specific vehicle. """
        try:
            traci.vehicle.setSpeed(vehicle_id, speed)
            self.logger.log(f"🚗 Vehicle {vehicle_id} speed set to {speed} m/s", "INFO", "blue",
                            class_name="VehicleController", function_name="update_vehicle_speed")
        except traci.TraCIException:
            self.logger.log(f"⚠️ Error: Unable to update speed for vehicle {vehicle_id}", "ERROR", "red",
                            class_name="VehicleController", function_name="update_vehicle_speed")

    def change_vehicle_lane(self, vehicle_id, step):
        """ Moves vehicle to another lane after 50 steps. """
        if vehicle_id and step == 50:
            try:
                traci.vehicle.changeLane(vehicle_id, 1, 5)
                self.logger.log(f"🔄 Vehicle {vehicle_id} changed to lane 1", "INFO", "magenta",
                                class_name="VehicleController", function_name="change_vehicle_lane")
            except traci.TraCIException:
                self.logger.log(f"⚠️ Error: Unable to change lane for vehicle {vehicle_id}", "ERROR", "red",
                                class_name="VehicleController", function_name="change_vehicle_lane")

    def log_vehicle_info(self):
        """ Logs detailed vehicle info. """
        vehicles = traci.vehicle.getIDList()
        if vehicles:
            for v_id in vehicles:
                position = traci.vehicle.getPosition(v_id)
                speed = traci.vehicle.getSpeed(v_id)
                lane = traci.vehicle.getLaneIndex(v_id)
                self.logger.log(f"🚙 Vehicle {v_id}: Position ({position[0]:.3f}, {position[1]:.3f}), Speed {speed:.3f} m/s, Lane {lane}", "INFO",
                                class_name="VehicleController", function_name="log_vehicle_info")
        else:
            self.logger.log("⚠️ No vehicles detected in the simulation!", "WARNING", "red",
                            class_name="VehicleController", function_name="log_vehicle_info")

    def track_fastest_vehicle(self, step):
        """ Tracks the fastest vehicle in the simulation for each step. """
        vehicles = self.get_active_vehicles()
        current_fastest_vehicle = None
        current_fastest_speed = 0
        for v_id in vehicles:
            speed = traci.vehicle.getSpeed(v_id)
            if speed > current_fastest_speed:
                current_fastest_speed = speed
                current_fastest_vehicle = v_id

        # Check if this step has a faster vehicle than previously recorded
        if current_fastest_speed > self.fastest_speed:
            self.fastest_speed = current_fastest_speed
            self.fastest_vehicle = current_fastest_vehicle
            self.fastest_step = step

    def get_fastest_vehicle_summary(self):
        """ Returns the summary of the fastest vehicle recorded. """
        return self.fastest_vehicle, self.fastest_speed, self.fastest_step