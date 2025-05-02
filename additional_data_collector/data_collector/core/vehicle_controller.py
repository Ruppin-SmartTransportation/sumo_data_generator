import traci
import random

class VehicleController:
    """ Controls vehicles in the SUMO simulation. """
    def __init__(self, logger):
        self.logger = logger

    def init_vehicles_network(self):
        """ Initializes the vehicle network. """
        self.logger.log("🚗 Vehicle network initialized", "INFO", "blue",
                        class_name="VehicleController", function_name="init_vehicles_network")
        
        

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
