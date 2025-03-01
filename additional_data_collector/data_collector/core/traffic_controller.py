import traci

class TrafficController:
    """ Controls traffic lights in the SUMO simulation. """
    def __init__(self, logger):
        self.logger = logger
        # Retrieve the list of all traffic lights
        self.traffic_lights = traci.trafficlight.getIDList()
        self.logger.log(f"Detected Traffic Lights: {self.traffic_lights}", "INFO", "yellow",
                        class_name="TrafficController", function_name="__init__")

    def update_traffic_light(self, step, phase_duration):
        """ Changes the traffic light phase every {phase_duration} steps for all traffic lights. """
        # Update all traffic lights every 20 steps
        if step % phase_duration == 0:
            for tl_id in self.traffic_lights:
                try:
                    # Get the current phase and switch to the next one
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    new_phase = (current_phase + 1) % 4
                    traci.trafficlight.setPhase(tl_id, new_phase)

                    # Log the phase change
                    self.logger.log(f"🚦 Traffic light {tl_id} changed to phase {new_phase}", "INFO", "cyan",
                                    class_name="TrafficController", function_name="update_traffic_light")

                except traci.TraCIException as e:
                    self.logger.log(f"⚠️ Error updating traffic light {tl_id}: {e}", "ERROR", "red",
                                    class_name="TrafficController", function_name="update_traffic_light")
